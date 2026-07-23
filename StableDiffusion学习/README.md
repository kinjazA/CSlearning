# Stable Diffusion 学习大纲

> 本大纲按 6 个阶段组织，从扩散模型基础原理到前沿架构，逐步深入。每个章节对应 `doc/` 下的笔记和 `code/` 下的代码示例。

---

## 第一阶段：扩散模型基础

| 章节 | 内容要点 | 笔记 | 代码 |
|------|---------|------|------|
| 1.1 | **生成模型概览** — GAN / VAE / Autoregressive / Diffusion 对比，扩散模型的直觉理解 | `doc/1.1-生成模型概览.md` | — |
| 1.2 | **DDPM 原理（上）** — 前向加噪过程，马尔可夫链，重参数化技巧 | `doc/1.2-DDPM前向过程.md` | `code/1.2-forward-diffusion.ipynb` |
| 1.3 | **DDPM 原理（下）** — 反向去噪过程，U-Net 预测噪声，损失函数推导 | `doc/1.3-DDPM反向过程.md` | `code/1.3-reverse-diffusion.ipynb` |
| 1.4 | **DDIM 与采样加速** — 非马尔可夫前向、确定性采样、跳步加速 | `doc/1.4-DDIM采样加速.md` | `code/1.4-ddim-sampling.ipynb` |
| 1.5 | **Score-based 视角** — Score Matching、SDE/ODE 统一框架、SDE Solvers | `doc/1.5-score-based-sde.md` | — |

## 第二阶段：Stable Diffusion 核心架构

| 章节 | 内容要点 | 笔记 | 代码 |
|------|---------|------|------|
| 2.1 | **SD 架构总览** — Latent Diffusion 设计动机，三大组件关系，pipeline 流程 | `doc/2.1-SD架构总览.md` | — |
| 2.2 | **VAE：潜空间压缩** — Encoder/Decoder 结构，KL 正则化，潜空间维度选择 | `doc/2.2-VAE潜空间.md` | `code/2.2-vae-latent.ipynb` |
| 2.3 | **U-Net 去噪网络** — 残差块、Self-Attention、时间嵌入、各层分工 | `doc/2.3-UNet去噪网络.md` | `code/2.3-unet-inspection.ipynb` |
| 2.4 | **CLIP 文本编码与 Cross-Attention** — CLIP 架构、文本嵌入、Classifier-Free Guidance | `doc/2.4-CLIP与条件控制.md` | `code/2.4-text-embedding.ipynb` |
| 2.5 | **噪声调度器 (Schedulers)** — DDPM / DDIM / PNDM / DPM-Solver / Euler 原理解析 | `doc/2.5-噪声调度器.md` | `code/2.5-schedulers-compare.ipynb` |

## 第三阶段：训练与微调

| 章节 | 内容要点 | 笔记 | 代码 |
|------|---------|------|------|
| 3.1 | **训练目标与损失函数** — 简化损失、噪声预测 vs 数据预测、SNR 加权 | `doc/3.1-训练目标与损失.md` | — |
| 3.2 | **数据集与标注** — LAION 数据集、图像-文本配对、自动标注 (BLIP, WD14) | `doc/3.2-数据集与标注.md` | — |
| 3.3 | **Dreambooth** — 先验保留损失、罕见 token、主体驱动生成 | `doc/3.3-Dreambooth.md` | `code/3.3-dreambooth/` |
| 3.4 | **Textual Inversion** — Embedding 空间优化、伪词概念注入 | `doc/3.4-TextualInversion.md` | `code/3.4-textual-inversion/` |
| 3.5 | **LoRA / QLoRA** — 低秩分解原理、rank 选择、Concepts vs Styles | `doc/3.5-LoRA.md` | `code/3.5-lora-training/` |

## 第四阶段：高级条件控制

| 章节 | 内容要点 | 笔记 | 代码 |
|------|---------|------|------|
| 4.1 | **Classifier-Free Guidance** — CFG 原理与推导、scale 调参、负面提示词 | `doc/4.1-CFG详解.md` | `code/4.1-cfg-experiments.ipynb` |
| 4.2 | **ControlNet** — Zero-Convolution、可训练副本、边缘/深度/姿态条件 | `doc/4.2-ControlNet.md` | `code/4.2-controlnet/` |
| 4.3 | **IP-Adapter** — 图像作为提示词、解耦 Cross-Attention、FaceID 变体 | `doc/4.3-IPAdapter.md` | `code/4.3-ipadapter.ipynb` |
| 4.4 | **Inpainting & Outpainting** — 蒙版策略、专用模型 vs 通用方法 | `doc/4.4-Inpainting.md` | `code/4.4-inpainting.ipynb` |

## 第五阶段：推理加速与部署

| 章节 | 内容要点 | 笔记 | 代码 |
|------|---------|------|------|
| 5.1 | **知识蒸馏** — LCM / SDXL-Turbo / TCD，一致性模型原理 | `doc/5.1-蒸馏加速.md` | `code/5.1-lcm-demo.ipynb` |
| 5.2 | **推理优化** — xFormers / FlashAttention / TensorRT / ONNX / VAE Tiling | `doc/5.2-推理优化.md` | `code/5.2-optimization-bench.ipynb` |
| 5.3 | **部署方案** — Diffusers / ComfyUI / Automatic1111 / 自建服务 | `doc/5.3-部署方案.md` | — |

## 第六阶段：前沿架构与生态

| 章节 | 内容要点 | 笔记 | 代码 |
|------|---------|------|------|
| 6.1 | **SDXL** — 双文本编码器、Refiner 模型、更高分辨率策略 | `doc/6.1-SDXL.md` | `code/6.1-sdxl-compare.ipynb` |
| 6.2 | **DiT (Diffusion Transformer)** — 用 Transformer 替代 U-Net，SD3 / FLUX 架构 | `doc/6.2-DiT与Transformer.md` | — |
| 6.3 | **FLUX** — Flow Matching、Rectified Flow、MMDiT 双流注意力 | `doc/6.3-FLUX.md` | — |
| 6.4 | **视频生成** — AnimateDiff / SVD / I2VGen-XL 运动模块原理 | `doc/6.4-视频生成.md` | — |
| 6.5 | **社区生态与工具链** — ComfyUI 节点、CivitAI 模型、HuggingFace Diffusers 最佳实践 | `doc/6.5-社区生态.md` | — |

---

## 学习路线建议

1. **入门路径**（无扩散基础）：1.1 → 1.2 → 1.3 → 2.1 → 2.3 → 2.4 → 动手跑通 SD 推理
2. **进阶路径**（已有基础）：3.5 LoRA → 4.2 ControlNet → 5.1 蒸馏 → 6.2 DiT
3. **研究向路径**：1.5 Score-based → 2.5 调度器 → 3.1 损失函数 → 6.3 FLUX

## 前置知识

- 概率论基础（高斯分布、条件概率、贝叶斯）
- 深度学习基础（CNN、Transformer、U-Net）
- PyTorch 基础（能读训练代码）
- （可选）扩散模型数学基础可配合 *Understanding Diffusion Models: A Unified Perspective* 论文阅读
