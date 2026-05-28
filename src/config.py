"""Central configuration for SEED-VII preprocessing, model, and training."""
from __future__ import annotations
from typing import Dict

PREPROCESS_DEFAULTS: Dict[str, object] = {
    "fs": 200,
    "window_seconds": 4.0,
    "step_seconds": 2.0,
    "middle_ratio": 0.6,
    "max_windows_per_trial": 60,
    "use_car": True,
    "use_baseline_correct": True,
    "use_ica": False,
    "ica_components": 20,
    "ica_remove": 5,
    "per_channel_zscore": True,
    "eps": 1e-8,
    "save_float32": True,
}

EEGNET_CONFIG: Dict[str, object] = {
    "n_channels": 62,
    "n_timepoints": 800,
    "n_classes": 3,        # 3类聚合：正面/中性/负面
    "F1": 8,
    "D": 2,
    "F2": 16,
    "kernLength": 100,
    "dropout": 0.5,
    "intensity_head_hidden": 64,
}

CONFORMER_CONFIG: Dict[str, object] = {
    "n_channels": 62,
    "n_timepoints": 800,
    "n_classes": 3,        # 3类聚合：正面/中性/负面
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

TRAIN_DEFAULTS: Dict[str, object] = {
    "output_dir": "runs_seed_vii",
    "device": "auto",
    "amp": True,
    "seed": 42,
    "batch_size": 256,
    "num_workers": 2,
    "optimizer": "adam",
    "lr": 2e-4,
    "min_lr": 1e-5,
    "beta1": 0.5,
    "beta2": 0.999,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "pretrain_epochs": 10,
    "max_epochs": 200,
    "patience": 30,
    "alpha_cls_start": 1.0,
    "beta_reg_start": 0.5,
    "gamma_rank_start": 0.0,
    "gamma_rank_end": 0.8,
    "rank_warmup_epochs": 30,
    "enable_rank": False,
    "rank_margin": 0.05,
    "label_smoothing": 0.05,
    "sample_weight_mode": "continuous",
    "intensity_threshold": 0.5,
    "weak_sample_weight": 0.1,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "split_unit": "trial",
    "save_interval": 1,
    "max_runtime_hours": 10.0,
    "save_last": True,
    "save_features": False,
    "feature_type": "projected",
    "pin_memory": False,
    "persistent_workers": False,
    "model_type": "eegnet",
    "train_subjects": "",
    "val_subjects": "",
    "test_subjects": "",
    "freeze_intensity_head": False,
    "balance_train": True,   # 训练集 WeightedRandomSampler 类别均衡，默认开启
}

MODELSCOPE_DATASET_ID = "DEREKVERSE/SEED-VII"
NPZ_REPO_PREFIX = "preprocessed_npz"
