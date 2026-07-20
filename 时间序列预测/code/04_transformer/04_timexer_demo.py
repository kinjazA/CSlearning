"""
TimeXer 完整学习示例
====================
两套实现，按需选择：

  A. PyTorch 从零构建 — 深入理解"外生变量增强 Transformer"的内部机制
  B. 合成数据训练 + 预测 + 可视化（包含外生变量）

参考文献：
  Wang et al. "TimeXer: Empowering Transformers for Time Series Forecasting
  with Exogenous Variables" (NeurIPS 2024)

环境依赖：
  pip install torch numpy pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 0. 合成带外生变量的时间序列数据
# ============================================================================

def make_timeseries_with_exogenous(num_samples=1000, L=96, H=96,
                                    C_en=2, C_ex=3, seed=42):
    """生成内生变量 + 外生变量的时间序列数据。

    场景：电力需求预测
      内生变量 (C_en=2):
        - var0: 电力需求（要预测的目标）
        - var1: 价格（与需求相关）

      外生变量 (C_ex=3):
        - exo0: 温度（已知未来7天预报）
        - exo1: 节假日标识（已知）
        - exo2: 历史同期均值（已知）

    返回:
      X_en: (N, L, C_en) - 内生变量历史
      X_ex: (N, L+H, C_ex) - 外生变量（历史+未来）
      Y: (N, H, C_en) - 内生变量未来（预测目标）
    """
    rng = np.random.default_rng(seed)
    total_len = L + H
    total_points = num_samples + total_len
    t = np.arange(total_points)

    # ---- 外生变量（已知历史+未来）----
    # exo0: 温度（日周期 + 长期趋势）
    temperature = 20 + 10*np.sin(2*np.pi*t/24) + 5*np.sin(2*np.pi*t/365) \
                  + rng.normal(0, 1, total_points)

    # exo1: 节假日（每周末=1，平日=0）
    holiday = ((t % 7) >= 5).astype(float)

    # exo2: 历史同期均值（滞后7天的滑动平均，模拟"去年同期"）
    historical_avg = np.convolve(
        np.concatenate([np.zeros(7), rng.normal(50, 10, total_points)]),
        np.ones(7)/7, mode='same'
    )[:total_points]

    # ---- 内生变量（只有历史，未来未知）----
    # var0: 电力需求（受温度、节假日影响）
    demand = np.zeros(total_points)
    demand[0] = 50 + rng.normal(0, 5)
    for i in range(1, total_points):
        # 基础AR模式
        ar_component = 0.7 * demand[i-1]
        # 温度影响：温度高 → 需求高（空调）
        temp_effect = 0.5 * (temperature[i] - 20)
        # 节假日影响：假日 → 需求低
        holiday_effect = -8 * holiday[i]
        # 历史模式
        hist_effect = 0.2 * historical_avg[i]

        demand[i] = ar_component + temp_effect + holiday_effect \
                   + 0.1 * hist_effect + rng.normal(0, 2)

    # var1: 价格（与需求正相关，但有延迟）
    price = 0.3 * demand + 0.5 * np.roll(demand, 1) + rng.normal(0, 1, total_points)

    # 标准化
    demand = (demand - demand.mean()) / demand.std()
    price = (price - price.mean()) / price.std()
    temperature = (temperature - temperature.mean()) / temperature.std()
    historical_avg = (historical_avg - historical_avg.mean()) / historical_avg.std()

    # ---- 滑动窗口切片 ----
    X_en_list, X_ex_list, Y_list = [], [], []

    for start in range(num_samples):
        # 内生变量：只有历史
        x_en = np.stack([
            demand[start:start+L],
            price[start:start+L]
        ], axis=-1)  # (L, 2)

        # 外生变量：历史 + 未来
        x_ex = np.stack([
            temperature[start:start+total_len],
            holiday[start:start+total_len],
            historical_avg[start:start+total_len]
        ], axis=-1)  # (L+H, 3)

        # 目标：内生变量的未来
        y = np.stack([
            demand[start+L:start+total_len],
            price[start+L:start+total_len]
        ], axis=-1)  # (H, 2)

        X_en_list.append(x_en)
        X_ex_list.append(x_ex)
        Y_list.append(y)

    return np.array(X_en_list), np.array(X_ex_list), np.array(Y_list)


# ============================================================================
# A. PyTorch 从零构建 — 完整 TimeXer
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PatchEmbedding(nn.Module):
    """将时间序列切分成 patch 并嵌入"""
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x: (B, L, C_en)
        返回: (B, C_en*N_patch, d_model)
        """
        B, L, C = x.shape

        # 转置: (B, C, L)
        x = x.transpose(1, 2)

        # Unfold: (B, C, L) → (B, C, N_patch, P)
        N_patch = L // self.patch_len
        x = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)

        # Reshape: (B, C, N_patch, P) → (B, C*N_patch, P)
        x = x.reshape(B, C * N_patch, self.patch_len)

        # Embed: (B, C*N_patch, P) → (B, C*N_patch, d_model)
        x = self.projection(x)

        return x, N_patch


class SeriesEmbedding(nn.Module):
    """将外生变量的整条序列嵌入为一个 token"""
    def __init__(self, seq_len, d_model):
        super().__init__()
        # 直接投影：(seq_len,) → (d_model,)
        self.projection = nn.Linear(seq_len, d_model)

    def forward(self, x):
        """
        x: (B, L+H, C_ex)
        返回: (B, C_ex, d_model)
        """
        B, T, C = x.shape

        # 转置: (B, C_ex, L+H)
        x = x.transpose(1, 2)

        # 投影: (B, C_ex, L+H) → (B, C_ex, d_model)
        x = self.projection(x)

        return x


class PatchTransformerEncoder(nn.Module):
    """Patch-level Transformer Encoder（用于内生变量）"""
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        x: (B, C_en*N_patch, d_model)
        返回: (B, C_en*N_patch, d_model)
        """
        return self.encoder(x)


class SeriesTransformerEncoder(nn.Module):
    """Series-level Transformer Encoder（用于外生变量）"""
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        x: (B, C_ex, d_model)
        返回: (B, C_ex, d_model)
        """
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """Cross-Attention: 内生变量 attend to 外生变量"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, z_en, z_ex):
        """
        z_en: (B, C_en*N_patch, d_model) - 内生变量表示
        z_ex: (B, C_ex, d_model) - 外生变量表示
        返回: (B, C_en*N_patch, d_model), attention_weights
        """
        # Cross-Attention
        attn_output, attn_weights = self.cross_attn(
            query=z_en,
            key=z_ex,
            value=z_ex
        )

        # 残差 + LN
        z = self.norm1(z_en + attn_output)

        # FFN + 残差 + LN
        z = self.norm2(z + self.ffn(z))

        return z, attn_weights


class PredictionHead(nn.Module):
    """将 patch-level 表示投影到预测"""
    def __init__(self, d_model, n_vars, n_patches, pred_len):
        super().__init__()
        self.n_vars = n_vars
        self.n_patches = n_patches

        # 方案1：全局池化 + 投影
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, z):
        """
        z: (B, C_en*N_patch, d_model)
        返回: (B, pred_len, C_en)
        """
        B = z.shape[0]

        # Reshape: (B, C_en*N_patch, d_model) → (B, C_en, N_patch, d_model)
        z = z.reshape(B, self.n_vars, self.n_patches, -1)

        # 对每个变量，池化所有 patch
        z = z.mean(dim=2)  # (B, C_en, d_model)

        # 投影到预测长度: (B, C_en, d_model) → (B, C_en, pred_len)
        output = self.projection(z)

        # 转置: (B, C_en, pred_len) → (B, pred_len, C_en)
        output = output.transpose(1, 2)

        return output


class TimeXer(nn.Module):
    """TimeXer: 外生变量增强的时间序列预测模型"""
    def __init__(self,
                 seq_len=96,          # 回溯长度
                 pred_len=96,         # 预测长度
                 enc_in=7,            # 内生变量数量
                 exo_in=5,            # 外生变量数量
                 d_model=128,         # 模型维度
                 n_heads=8,           # 注意力头数
                 e_layers_en=2,       # 内生编码器层数
                 e_layers_ex=1,       # 外生编码器层数
                 patch_len=16,        # Patch 长度
                 dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.exo_in = exo_in
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len

        # 内生变量 Patch 编码
        self.patch_embedding = PatchEmbedding(patch_len, d_model)
        self.patch_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in * self.n_patches, d_model)
        )
        self.patch_encoder = PatchTransformerEncoder(
            d_model, n_heads, e_layers_en, dropout
        )

        # 外生变量 Series 编码
        self.series_embedding = SeriesEmbedding(seq_len + pred_len, d_model)
        self.series_pos_embedding = nn.Parameter(
            torch.randn(1, exo_in, d_model)
        )
        self.series_encoder = SeriesTransformerEncoder(
            d_model, n_heads, e_layers_ex, dropout
        )

        # Cross-Attention 融合
        self.cross_attention = CrossAttentionFusion(d_model, n_heads, dropout)

        # 预测头
        self.prediction_head = PredictionHead(
            d_model, enc_in, self.n_patches, pred_len
        )

    def forward(self, x_en, x_ex):
        """
        x_en: (B, L, C_en) - 内生变量历史
        x_ex: (B, L+H, C_ex) - 外生变量（历史+未来）

        返回:
          predictions: (B, H, C_en)
          attn_weights: (B, C_en*N_patch, C_ex)
        """
        # 1. 内生变量 Patch 编码
        patch_tokens, n_patches = self.patch_embedding(x_en)  # (B, C_en*N_patch, d_model)
        patch_tokens = patch_tokens + self.patch_pos_embedding
        z_en = self.patch_encoder(patch_tokens)

        # 2. 外生变量 Series 编码
        series_tokens = self.series_embedding(x_ex)  # (B, C_ex, d_model)
        series_tokens = series_tokens + self.series_pos_embedding
        z_ex = self.series_encoder(series_tokens)

        # 3. Cross-Attention 融合
        z_fused, attn_weights = self.cross_attention(z_en, z_ex)

        # 4. 预测
        predictions = self.prediction_head(z_fused)

        return predictions, attn_weights


# ============================================================================
# B. 训练与评估
# ============================================================================

def train_timexer(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cpu'):
    """训练 TimeXer 模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for x_en, x_ex, y in train_loader:
            x_en, x_ex, y = x_en.to(device), x_ex.to(device), y.to(device)

            optimizer.zero_grad()
            predictions, _ = model(x_en, x_ex)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_en, x_ex, y in val_loader:
                x_en, x_ex, y = x_en.to(device), x_ex.to(device), y.to(device)
                predictions, _ = model(x_en, x_ex)
                loss = criterion(predictions, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def visualize_results(model, x_en, x_ex, y_true, device='cpu', var_names=None):
    """可视化预测结果 + Cross-Attention 权重"""
    model.eval()

    with torch.no_grad():
        x_en = torch.FloatTensor(x_en).unsqueeze(0).to(device)
        x_ex = torch.FloatTensor(x_ex).unsqueeze(0).to(device)
        predictions, attn_weights = model(x_en, x_ex)

    predictions = predictions.squeeze(0).cpu().numpy()
    attn_weights = attn_weights.squeeze(0).cpu().numpy()
    x_en = x_en.squeeze(0).cpu().numpy()

    n_vars = predictions.shape[1]
    if var_names is None:
        var_names = [f'Var {i}' for i in range(n_vars)]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 预测结果（每个内生变量）
    for i in range(n_vars):
        ax = fig.add_subplot(gs[i, 0])

        # 历史
        ax.plot(range(len(x_en)), x_en[:, i],
                label='History', color='gray', linewidth=1.5)

        # 真实值
        ax.plot(range(len(x_en), len(x_en) + len(y_true)), y_true[:, i],
                label='True Future', color='blue', linewidth=1.5)

        # 预测值
        ax.plot(range(len(x_en), len(x_en) + len(predictions)), predictions[:, i],
                label='Prediction', color='red', linewidth=1.5, linestyle='--')

        ax.axvline(x=len(x_en), color='black', linestyle=':', alpha=0.5)
        ax.set_title(f'{var_names[i]} Prediction', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Cross-Attention 热力图
    ax_attn = fig.add_subplot(gs[:, 1])

    im = ax_attn.imshow(attn_weights, aspect='auto', cmap='YlOrRd')
    ax_attn.set_xlabel('Exogenous Variables', fontsize=12)
    ax_attn.set_ylabel('Endogenous Patches (time →)', fontsize=12)
    ax_attn.set_title('Cross-Attention Weights\n外生变量对不同时间段的影响',
                      fontsize=13, fontweight='bold')

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax_attn)
    cbar.set_label('Attention Weight', fontsize=11)

    # 标注外生变量名称
    exo_names = ['Temperature', 'Holiday', 'Historical Avg']
    ax_attn.set_xticks(range(len(exo_names)))
    ax_attn.set_xticklabels(exo_names, rotation=45, ha='right')

    plt.suptitle('TimeXer: Prediction Results & Cross-Attention Analysis',
                 fontsize=14, fontweight='bold', y=0.995)

    return fig


# ============================================================================
# C. 主函数 - 完整演示
# ============================================================================

def main():
    print("=" * 70)
    print("TimeXer 完整演示")
    print("=" * 70)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 1. 生成数据
    print("\n[1] 生成带外生变量的时间序列数据...")
    X_en, X_ex, Y = make_timeseries_with_exogenous(
        num_samples=1000, L=96, H=96, C_en=2, C_ex=3, seed=42
    )
    print(f"  内生变量 X_en: {X_en.shape} (历史)")
    print(f"  外生变量 X_ex: {X_ex.shape} (历史+未来)")
    print(f"  目标变量 Y: {Y.shape} (未来)")

    # 划分训练/验证集
    train_size = int(0.8 * len(X_en))
    X_en_train, X_en_val = X_en[:train_size], X_en[train_size:]
    X_ex_train, X_ex_val = X_ex[:train_size], X_ex[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_en_train),
        torch.FloatTensor(X_ex_train),
        torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_en_val),
        torch.FloatTensor(X_ex_val),
        torch.FloatTensor(Y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. 构建模型
    print("\n[2] 构建 TimeXer 模型...")
    model = TimeXer(
        seq_len=96,
        pred_len=96,
        enc_in=2,      # 2个内生变量
        exo_in=3,      # 3个外生变量
        d_model=128,
        n_heads=4,
        e_layers_en=2,
        e_layers_ex=1,
        patch_len=16,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 3. 训练
    print("\n[3] 开始训练...")
    train_losses, val_losses = train_timexer(
        model, train_loader, val_loader,
        epochs=50, lr=1e-4, device=device
    )

    # 4. 可视化训练曲线
    print("\n[4] 可视化训练曲线...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('TimeXer Training Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('timexer_training.png', dpi=150)
    print("  保存训练曲线: timexer_training.png")

    # 5. 预测并可视化
    print("\n[5] 预测并可视化结果...")
    idx = 0  # 选择一个测试样本
    fig = visualize_results(
        model,
        X_en_val[idx],
        X_ex_val[idx],
        Y_val[idx],
        device=device,
        var_names=['Demand', 'Price']
    )
    plt.savefig('timexer_prediction.png', dpi=150, bbox_inches='tight')
    print("  保存预测结果: timexer_prediction.png")

    # 6. 消融实验：对比有/无外生变量
    print("\n[6] 消融实验: 对比有无外生变量的影响...")

    # 评估完整版 TimeXer
    model.eval()
    mse_with_exo = 0
    with torch.no_grad():
        for x_en, x_ex, y in val_loader:
            x_en, x_ex, y = x_en.to(device), x_ex.to(device), y.to(device)
            pred, _ = model(x_en, x_ex)
            mse_with_exo += F.mse_loss(pred, y).item()
    mse_with_exo /= len(val_loader)

    # 评估无外生变量版本（外生变量置零）
    mse_without_exo = 0
    with torch.no_grad():
        for x_en, x_ex, y in val_loader:
            x_en, x_ex, y = x_en.to(device), x_ex.to(device), y.to(device)
            x_ex_zero = torch.zeros_like(x_ex)  # 外生变量置零
            pred, _ = model(x_en, x_ex_zero)
            mse_without_exo += F.mse_loss(pred, y).item()
    mse_without_exo /= len(val_loader)

    improvement = (mse_without_exo - mse_with_exo) / mse_without_exo * 100

    print(f"\n  结果对比:")
    print(f"  ├─ 有外生变量 MSE:  {mse_with_exo:.4f}")
    print(f"  ├─ 无外生变量 MSE:  {mse_without_exo:.4f}")
    print(f"  └─ 精度提升:        {improvement:.2f}%")

    print("\n" + "=" * 70)
    print("演示完成！生成的图片:")
    print("  - timexer_training.png: 训练曲线")
    print("  - timexer_prediction.png: 预测结果 + Cross-Attention 热力图")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
