"""Central configuration for SEED-VII preprocessing, model, and training.

重构原则 (2026-05-27):
- 模型重点转向 EEGNet，Conformer 保留但非默认
- ICA 默认关闭且不在主 Pipeline 中使用
- 预处理输出为 per-subject .npz (20 个文件)，回写到 ModelScope 数据集
- 训练从 .npz 文件加载

更新 (2026-05-28):
- 新增 split_mode 参数，支持 "all"（全被试混合 trial-level 分割，原始行为）
  和 "per_subject"（每个被试独立 trial-level 分割，再合并）
"""
from __future__ import annotations
from typing import Dict

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
PREPROCESS_DEFAULTS: Dict[str, object] = {
    "fs": 200,                     # SEED-VII EEG_preprocessed 已 200Hz
    "window_seconds": 4.0,         # 4 秒窗口
    "step_seconds": 2.0,           # 50% 重叠
    "middle_ratio": 0.6,           # 每个 trial 取居中 60%
    "max_windows_per_trial": 60,   # 防长视频主导：每 clip 至多 N 个窗口
    "use_car": True,               # 平均参考
    "use_baseline_correct": True,  # 基线去均值
    "use_ica": False,              # ICA — 关闭，不在主 Pipeline 中使用
    "ica_components": 20,
    "ica_remove": 5,
    "per_channel_zscore": True,    # 按通道 z-score（优先方案）
    "eps": 1e-8,
    "save_float32": True,
}

# --------------------------------------------------------------------------
# EEGNet 模型配置（主力模型）
# --------------------------------------------------------------------------
EEGNET_CONFIG: Dict[str, object] = {
    "n_channels": 62,
    "n_timepoints": 800,           # 4s * 200Hz
    "n_classes": 7,                # SEED-VII 7 类情绪
    "F1": 8,                       # 时间滤波器数量
    "D": 2,                        # 深度乘子
    "F2": 16,                      # 通常 F2 = F1 * D
    "kernLength": 100,             # 200Hz raw EEG，0.5秒感受野
    "dropout": 0.5,                # 跨被试建议 0.5；被试内可 0.25
    "intensity_head_hidden": 64,
}

# --------------------------------------------------------------------------
# EEGConformer 配置（保留备用，≈0.75M 参数）
# --------------------------------------------------------------------------
CONFORMER_CONFIG: Dict[str, object] = {
    "n_channels": 62,
    "n_timepoints": 800,
    "n_classes": 7,
    "embed_dim": 40,
    "time_kernel": 20,
    "time_padding": 10,
    "pool_kernel": 60,
    "pool_stride": 12,
    "transformer_layers": 6,
    "transformer_heads": 10,
    "ffn_dim": 160,
    "head_hidden": 256,
    "dropout": 0.5,
    "intensity_head_hidden": 64,
}

def compute_token_count(n_timepoints: int, pool_kernel: int, pool_stride: int) -> int:
    return (n_timepoints - pool_kernel) // pool_stride + 1

CONFORMER_CONFIG["n_tokens"] = compute_token_count(
    int(CONFORMER_CONFIG["n_timepoints"]),
    int(CONFORMER_CONFIG["pool_kernel"]),
    int(CONFORMER_CONFIG["pool_stride"]),
)

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
TRAIN_DEFAULTS: Dict[str, object] = {
    "output_dir": "runs_seed_vii",
    "device": "auto",
    "amp": True,
    "seed": 42,
    "batch_size": 256,
    "num_workers": 2,
    # 优化器
    "optimizer": "adam",
    "lr": 2e-4,
    "min_lr": 1e-5,
    "beta1": 0.5,
    "beta2": 0.999,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # 训练阶段
    "pretrain_epochs": 10,
    "max_epochs": 200,
    "patience": 30,
    # 损失权重
    "alpha_cls_start": 1.0,
    "beta_reg_start": 0.5,
    "gamma_rank_start": 0.0,
    "gamma_rank_end": 0.8,
    "rank_warmup_epochs": 30,
    "enable_rank": False,
    "rank_margin": 0.05,
    "label_smoothing": 0.05,
    # 样本权重
    "sample_weight_mode": "continuous",
    "intensity_threshold": 0.5,
    "weak_sample_weight": 0.1,
    # ★ 分割模式
    # "all"        : 全被试 trial 混合后随机分割（原始行为，跨被试泛化）
    # "per_subject": 每个被试独立 trial-level 分割，再合并（被试内泛化）
    "split_mode": "all",
    # 划分比例
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "split_unit": "trial",
    # 续训 / 容错
    "save_interval": 1,
    "max_runtime_hours": 10.0,
    "save_last": True,
    "save_features": False,
    "feature_type": "projected",
    # DataLoader
    "pin_memory": False,
    "persistent_workers": False,
    # 模型选择 — 默认 EEGNet
    "model_type": "eegnet",        # ["eegnet", "conformer"]
    # 被试筛选
    "train_subjects": "",
    "val_subjects": "",
    "test_subjects": "",
    # 过拟合缓解
    "freeze_intensity_head": False,
}

# --------------------------------------------------------------------------
# ModelScope 数据集
# --------------------------------------------------------------------------
MODELSCOPE_DATASET_ID = "DEREKVERSE/SEED-VII"
NPZ_REPO_PREFIX = "preprocessed_npz"   # npz 文件在 dataset repo 中的路径前缀
