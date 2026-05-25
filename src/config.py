"""Central configuration for SEED-VII preprocessing, model, and training.

参考 PRIMOCOSMOS/EEG_encoder/config.py 的风格（参数量目标 0.7–0.8M）。
本配置面向 62ch / 200Hz / 4s 窗口（即 800 时间点），7 类情绪 + 强度回归双头。
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------
PREPROCESS_DEFAULTS: Dict[str, object] = {
    "fs": 200,                       # SEED-VII EEG_preprocessed 已 200Hz
    "window_seconds": 4.0,           # 4 秒窗口
    "step_seconds": 2.0,             # 50% 重叠
    "middle_ratio": 0.6,             # 每个 trial 取居中 60%
    "max_windows_per_trial": 60,     # 防长视频主导：每 clip 至多 N 个窗口（居中截取）
    "use_car": True,                 # 平均参考
    "use_baseline_correct": True,    # 基线去均值（默认用整段均值，避免依赖额外baseline段）
    "use_ica": True,                # ICA 较慢，默认关闭（如开启建议线下做一次）
    "ica_components": 20,
    "ica_remove": 5,
    "per_channel_zscore": True,      # 按通道 z-score（优先方案）
    "eps": 1e-8,
    "save_float32": True,
}

# ---------------------------------------------------------------------------
# EEG-Conformer 模型（目标 ~0.75M 参数；与 SEED-IV 参考一致：embed_dim=40, depth=6, heads=10, ffn=160）
# ---------------------------------------------------------------------------
CONFORMER_CONFIG: Dict[str, object] = {
    "n_channels": 62,
    "n_timepoints": 800,             # 4s * 200Hz
    "n_classes": 7,                  # SEED-VII 7 类情绪
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
    "intensity_head_hidden": 64,     # 强度回归头中间层
}


def compute_token_count(n_timepoints: int, pool_kernel: int, pool_stride: int) -> int:
    return (n_timepoints - pool_kernel) // pool_stride + 1


CONFORMER_CONFIG["n_tokens"] = compute_token_count(
    int(CONFORMER_CONFIG["n_timepoints"]),
    int(CONFORMER_CONFIG["pool_kernel"]),
    int(CONFORMER_CONFIG["pool_stride"]),
)

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS: Dict[str, object] = {
    "output_dir": "runs_seed_vii_conformer",
    "device": "auto",
    "amp": True,
    "seed": 42,
    "batch_size": 256,
    "num_workers": 6,

    # 优化器
    "optimizer": "adam",
    "lr": 2e-4,
    "min_lr": 1e-5,                  # 余弦退火下限
    "beta1": 0.5,
    "beta2": 0.999,
    "weight_decay": 0.0,
    "grad_clip": 1.0,

    # 训练阶段
    "pretrain_epochs": 10,           # 只开 L_cls 的预训练 epoch 数
    "max_epochs": 200,
    "patience": 30,                  # 早停

    # 损失权重（联合训练）
    "alpha_cls_start": 1.0,
    "beta_reg_start": 0.5,
    "gamma_rank_start": 0.0,         # 退化方案：先关掉 ranking
    "gamma_rank_end": 0.8,           # 若 enable_rank：逐步提升到 0.8
    "rank_warmup_epochs": 30,        # ranking 升温 epoch
    "enable_rank": False,            # 默认退化方案：False
    "rank_margin": 0.05,             # margin ranking 的 margin
    "label_smoothing": 0.05,         # 分类损失标签平滑

    # 样本权重 (s_i = 连续标签)
    "sample_weight_mode": "continuous",  # ["continuous", "threshold", "none"]
    "intensity_threshold": 0.5,           # threshold 模式阈值
    "weak_sample_weight": 0.1,            # 阈值以下样本的降权系数

    # 划分（trial-level）
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "split_unit": "trial",            # ["trial", "subject", "session"]

    # 续训 / 容错
    "save_interval": 1,
    "max_runtime_hours": 10.0,        # 一次最多 10 小时
    "save_last": True,
    "save_features": False,
    "feature_type": "projected",      # ["projected", "flatten"]
}
