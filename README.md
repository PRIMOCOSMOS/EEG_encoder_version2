# SEED-VII EEGNet Encoder（情绪分类 + 强度回归 双头）

**重构版 (2026-05-27)**：模型重心转向 EEGNet，预处理 Pipeline 不含 ICA，输出 per-subject `.npz` 文件并回写到 ModelScope 数据集。

## 核心变化（相对前版）

| 维度 | 前版 | 本版 |
|------|------|------|
| 模型 | EEGConformer + EEGNet 双选 | **EEGNet 为主力**，Conformer 保留备用 |
| ICA | 可选但实现不完善 | **彻底关闭**，不在 Pipeline 中使用 |
| 预处理输出 | 单个巨型 `seed_vii.npz` | **20 个 per-subject `.npz`** |
| 数据回写 | 不回写 | **上传到 ModelScope `preprocessed_npz/`** |
| 旧 npz | — | **作废**，以新 per-subject npz 为准 |

## 预处理 Pipeline

```
原始 EEG (62通道, 200Hz, 已带通+陷波)
    ↓ 基线校正（减均值）
    ↓ 平均参考 CAR
    ↓ 居中 60% 裁剪
    ↓ 4秒窗口 50%重叠（每 clip 最多 60 个居中窗口）
    ↓ 按通道 z-score
    → {subject}.npz
```

## 目录结构

```
EEG_encoder_version2/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── config.py           # 所有默认超参
│   ├── labels.py           # SEED-VII 7 类情绪 + session_sequences
│   ├── preprocess.py       # 基线校正/CAR/裁剪/窗口/z-score（无 ICA）
│   ├── dataset.py          # per-subject npz 加载 + trial-level 划分
│   ├── model.py            # EEGNet (主力) + EEGConformer (备用) 双头模型
│   ├── losses.py           # CE+LS、MSE、Ranking Loss、加权组合
│   ├── trainer.py          # 训练/续训/早停/AMP/余弦退火/超时
│   ├── inference.py        # 编码器推理：导出 embedding/预测
│   └── ms_io.py            # ModelScope 上传/下载封装
├── scripts/
│   ├── preprocess_to_npz.py            # 预处理 .mat → per-subject .npz
│   ├── upload_npz_to_modelscope.py     # 上传 .npz 到 ModelScope
│   ├── download_npz_from_modelscope.py # 从 ModelScope 下载 .npz
│   ├── train.py                        # 训练入口
│   └── encode.py                       # 推理入口
└── notebooks/
    └── pipeline.ipynb      # 一键调用所有脚本的 Jupyter Notebook
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 预处理 20 个 .mat → 20 个 .npz
python scripts/preprocess_to_npz.py \
  --mat-dir /data/EEG_preprocessed \
  --output-dir /workspace/preprocessed \
  --save-info-dir /data/save_info \
  --compress

# 2) 上传 .npz 到 ModelScope
export MODELSCOPE_API_TOKEN=...
python scripts/upload_npz_to_modelscope.py \
  --npz-dir /workspace/preprocessed \
  --dataset DEREKVERSE/SEED-VII

# 3) 训练 EEGNet
python scripts/train.py \
  --data-dir /workspace/preprocessed \
  --output-dir /workspace/runs \
  --model-type eegnet \
  --device auto --amp \
  --max-runtime-hours 10

# 4) 续训
python scripts/train.py \
  --data-dir /workspace/preprocessed \
  --output-dir /workspace/runs \
  --resume --max-runtime-hours 10

# 5) 编码推理
python scripts/encode.py \
  --data-dir /workspace/preprocessed \
  --checkpoint /workspace/runs/best_encoder.pt \
  --output /workspace/encoded.npz
```

## 训练策略

1. **预训练** (前 10 epoch)：只开 `L_cls`（交叉熵 + 标签平滑）
2. **联合训练**：开 `L_cls + L_reg`（退化方案，`L_rank` 暂不开启）
3. **余弦退火**学习率，最小 `1e-5`
4. **10h 软超时** + 优雅保存
5. **样本权重**：连续标签 `s_i` 作为动态权重

## 损失函数

$$
\mathcal{L} = s_i \cdot [\alpha \mathcal{L}_{cls} + \beta \mathcal{L}_{reg}]
$$

退化方案（排序损失 `γ=0`），后续可升级为完整版。

## ModelScope 数据流

```
ModelScope Dataset: DEREKVERSE/SEED-VII
├── EEG_preprocessed/    ← 原始 .mat (已有)
├── save_info/           ← 连续标签 CSV (已有)
├── preprocessed_npz/    ← ★ 新增：per-subject .npz
│   ├── 1.npz
│   ├── 2.npz
│   └── ...20.npz
└── (旧 npz 文件作废)
```
