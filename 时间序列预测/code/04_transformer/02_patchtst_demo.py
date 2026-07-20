"""
PatchTST 完整学习示例
=====================
两套内容，按需使用：

  A. PyTorch 从零构建 — 深入理解 Patching + Channel-Independent
  B. 合成数据训练 + 预测 + 可视化 + 消融实验

参考文献：
  Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting
  with Transformers" (ICLR 2023)

环境依赖：
  pip install torch numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成多变量时间序列数据
# ============================================================================

def make_multivariate_ts(num_samples=1000, L=96, H=96, C=4, seed=42):
    """生成含跨变量依赖的多变量时间序列。

    模拟 4 个物理量：
      var0: 温度（AR(2) + 季节）
      var1: 湿度（依赖温度 + 自噪声）
      var2: 气压（独立 AR(1)）
      var3: 风速（依赖湿度 + 气压）

    返回 X: (N, L, C), Y: (N, H, C)
    """
    rng = np.random.default_rng(seed)
    total_len = L + H
    total_points = num_samples + total_len
    t = np.arange(total_points) / 50

    # var0: 温度
    v0 = np.zeros(total_points)
    v0[:2] = 20 + rng.normal(0, 1, 2)
    for i in range(2, total_points):
        v0[i] = 0.7*v0[i-1] - 0.2*v0[i-2] + 5*np.sin(2*np.pi*i/200) + rng.normal(0, 0.5)
    v0 = (v0 - v0.mean()) / v0.std()

    # var1: 湿度（受温度反向影响）
    v1 = -0.6*v0 + 0.3*np.sin(2*np.pi*t/300) + rng.normal(0, 0.3, total_points)

    # var2: 气压（独立 AR(1)）
    v2 = np.zeros(total_points)
    v2[0] = rng.normal(0, 1)
    for i in range(1, total_points):
        v2[i] = 0.9*v2[i-1] + rng.normal(0, 0.2)

    # var3: 风速（受湿度和气压影响）
    v3 = 0.3*v1 + 0.4*v2 + rng.normal(0, 0.4, total_points)

    # 滑动窗口切片
    data = np.stack([v0, v1, v2, v3], axis=-1)
    X, Y = [], []
    for start in range(num_samples):
        X.append(data[start:start+L])
        Y.append(data[start+L:start+total_len])
    return np.array(X), np.array(Y)


# ============================================================================
# A. PyTorch 从零构建 — PatchTST
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PatchEmbedding(nn.Module):
    """时间序列 → patch 嵌入"""
    def __init__(self, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, N_patch, P) → (B, C, N_patch, d_model)"""
        return self.dropout(self.projection(x))


class PatchTSTEncoder(nn.Module):
    """标准 Transformer Encoder"""
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """x: (B*C, N_patch, d_model) → (B*C, N_patch, d_model)"""
        return self.encoder(x)


class FlattenHead(nn.Module):
    """Flatten + Linear 预测头"""
    def __init__(self, d_model, n_patches, pred_len, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(dropout),
            nn.Linear(n_patches * d_model, pred_len)
        )

    def forward(self, x):
        """x: (B*C, N_patch, d_model) → (B*C, pred_len)"""
        return self.head(x)


class PatchTST(nn.Module):
    """PatchTST: A Time Series is Worth 64 Words

    参数:
        enc_in: 变量数 C
        seq_len: 回溯长度 L
        pred_len: 预测长度 H
        patch_len: patch 大小 P
        stride: patch 步长（默认=patch_len 不重叠）
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Encoder 层数
        use_channel_independent: 是否变量独立（默认True）
    """
    def __init__(self,
                 enc_in=7,
                 seq_len=96,
                 pred_len=96,
                 patch_len=16,
                 stride=None,
                 d_model=128,
                 n_heads=16,
                 n_layers=3,
                 dropout=0.1,
                 use_channel_independent=True):
        super().__init__()

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride if stride is not None else patch_len
        self.use_channel_independent = use_channel_independent

        # 计算 patch 数量
        self.n_patches = (seq_len - patch_len) // self.stride + 1

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(patch_len, d_model, dropout)

        # 可学习位置编码
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )

        # Transformer Encoder
        self.encoder = PatchTSTEncoder(d_model, n_heads, n_layers, dropout)

        # 预测头（每个变量独立）
        if use_channel_independent:
            self.heads = nn.ModuleList([
                FlattenHead(d_model, self.n_patches, pred_len, dropout)
                for _ in range(enc_in)
            ])
        else:
            self.head = FlattenHead(d_model, self.n_patches * enc_in, pred_len, dropout)

    def forward(self, x):
        """
        x: (B, L, C)
        返回: (B, H, C)
        """
        B, L, C = x.shape

        # 1. Patching: (B, L, C) → (B, C, N_patch, P)
        x = x.permute(0, 2, 1)             # (B, C, L)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, C, N_patch, P)

        # 2. Patch Embedding: (B, C, N_patch, P) → (B, C, N_patch, d_model)
        x = self.patch_embedding(x)

        # 3. 位置编码
        x = x + self.pos_encoding[:, :self.n_patches, :]

        if self.use_channel_independent:
            # 4a. Channel-Independent: 逐个变量编码
            # 合并 B 和 C 维度: (B, C, N_patch, d_model) → (B*C, N_patch, d_model)
            x = x.reshape(B * C, self.n_patches, -1)

            # Transformer Encoder
            x = self.encoder(x)  # (B*C, N_patch, d_model)

            # 分开变量: (B*C, N_patch, d_model) → (B, C, N_patch, d_model)
            x = x.reshape(B, C, self.n_patches, -1)

            # 每个变量用独立的预测头
            outputs = []
            for i in range(C):
                pred_i = self.heads[i](x[:, i, :, :])  # (B, pred_len)
                outputs.append(pred_i)

            output = torch.stack(outputs, dim=-1)  # (B, pred_len, C)

        else:
            # 4b. Channel-Mixing: 所有变量的patch混合
            x = x.reshape(B, C * self.n_patches, -1)
            x = self.encoder(x)
            output = self.head(x)  # (B, pred_len * C)
            output = output.reshape(B, self.pred_len, C)

        return output


# ============================================================================
# B. 训练与评估
# ============================================================================

def train_patchtst(model, train_loader, val_loader, epochs=50, lr=1e-4,
                   patience=10, device='cpu'):
    """训练 PatchTST"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_patchtst.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')

    return train_losses, val_losses


def visualize_predictions(model, X, Y, var_idx=0, device='cpu',
                          sample_idx=0, var_name='Variable'):
    """可视化单个变量的预测"""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X[sample_idx:sample_idx+1]).to(device)
        pred = model(x).squeeze(0).cpu().numpy()

    history = X[sample_idx, :, var_idx]
    true_future = Y[sample_idx, :, var_idx]
    pred_future = pred[:, var_idx]
    L = len(history)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(L), history, label='History', color='gray', linewidth=1.5)
    ax.plot(range(L, L+len(true_future)), true_future,
            label='True', color='blue', linewidth=1.5)
    ax.plot(range(L, L+len(pred_future)), pred_future,
            label='Prediction', color='red', linewidth=1.5, linestyle='--')
    ax.axvline(x=L, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'{var_name} Prediction', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================================
# C. 主函数 - 完整演示
# ============================================================================

def main():
    print("=" * 65)
    print("PatchTST 完整演示")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 1. 生成数据
    print("\n[1] 生成多变量时间序列...")
    X, Y = make_multivariate_ts(num_samples=1000, L=96, H=96, C=4, seed=42)
    print(f"  X: {X.shape} (N, 96, 4)")
    print(f"  Y: {Y.shape} (N, 96, 4)")

    # 划分数据
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(Y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. 构建模型
    print("\n[2] 构建 PatchTST (P=16)...")
    model = PatchTST(
        enc_in=4,
        seq_len=96,
        pred_len=96,
        patch_len=16,
        d_model=128,
        n_heads=4,
        n_layers=2,
        use_channel_independent=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")
    print(f"  Patch 数量: {model.n_patches} (96/{model.patch_len})")

    # 3. 训练
    print("\n[3] 训练中...")
    train_losses, val_losses = train_patchtst(
        model, train_loader, val_loader,
        epochs=50, lr=1e-4, patience=10, device=device
    )

    # 4. 消融实验：对比不同 P 值
    print("\n[4] 消融实验: 不同 patch 长度对比...")
    p_values = [4, 8, 16, 24]
    ablation_results = {}

    for P in p_values:
        print(f"  测试 P={P}...")
        m = PatchTST(
            enc_in=4, seq_len=96, pred_len=96,
            patch_len=P, d_model=64, n_heads=2, n_layers=1  # 小模型快速测试
        ).to(device)

        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 快速训练 20 epoch
        for epoch in range(20):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = m(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 评估
        m.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(m(x), y).item()
        val_loss /= len(val_loader)

        n_tokens = 96 // P
        ablation_results[P] = {'loss': val_loss, 'n_tokens': n_tokens}
        print(f"    P={P}, token数={n_tokens}, Val MSE={val_loss:.4f}")

    # 5. 可视化
    print("\n[5] 生成可视化...")

    # 5a. 训练曲线
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('PatchTST Training Curve', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 5b. 消融实验
    P_list = list(ablation_results.keys())
    losses = [ablation_results[p]['loss'] for p in P_list]
    tokens = [ablation_results[p]['n_tokens'] for p in P_list]

    ax2.plot(P_list, losses, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Patch Length (P)')
    ax2.set_ylabel('Validation MSE', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(P_list, tokens, 's-', linewidth=2, markersize=8, color='#3498db')
    ax2_twin.set_ylabel('Number of Tokens (L/P)', color='#3498db')
    ax2_twin.tick_params(axis='y', labelcolor='#3498db')

    ax2.set_title('Ablation: Patch Length Effect', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig1.savefig('patchtst_training_and_ablation.png', dpi=150)
    print("  保存: patchtst_training_and_ablation.png")

    # 5c. 预测结果
    var_names = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed']
    fig2, axes = plt.subplots(2, 2, figsize=(14, 8))

    for i, ax in enumerate(axes.flat):
        history = X_val[0, :, i]
        true_future = Y_val[0, :, i]

        with torch.no_grad():
            x_in = torch.FloatTensor(X_val[0:1]).to(device)
            pred = model(x_in).squeeze(0).cpu().numpy()

        L = len(history)
        ax.plot(range(L), history, label='History', color='gray', linewidth=1.5)
        ax.plot(range(L, L+len(true_future)), true_future,
                label='True Future', color='blue', linewidth=1)
        ax.plot(range(L, L+len(pred)), pred[:, i],
                label='Prediction', color='red', linewidth=1, linestyle='--')
        ax.axvline(x=L, color='black', linestyle=':', alpha=0.5)
        ax.set_title(var_names[i], fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig2.suptitle('PatchTST: Multi-variable Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig('patchtst_predictions.png', dpi=150)
    print("  保存: patchtst_predictions.png")

    # 6. 打印总结
    best_P = min(ablation_results.items(), key=lambda x: x[1]['loss'])
    print(f"\n{'='*65}")
    print(f"实验结果总结:")
    print(f"  Best Patch Length: P={best_P[0]} (MSE={best_P[1]['loss']:.4f})")
    print(f"  最终 Val Loss: {val_losses[-1]:.4f}")
    print(f"  Token 数: {model.n_patches} (vs 逐点token的 96)")
    print(f"  Attention 复杂度降低: {96//model.n_patches}² = {(96//model.n_patches)**2}x")
    print(f"\n生成的图片:")
    print(f"  - patchtst_training_and_ablation.png: 训练曲线 + 消融实验")
    print(f"  - patchtst_predictions.png: 4个变量的预测结果")
    print(f"{'='*65}")

    plt.show()


if __name__ == "__main__":
    main()
