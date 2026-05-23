# SEED-VII EEG Conformer Encoder（情绪分类 + 强度回归 双头）

基于 SEED-VII 数据集（62通道 / 200Hz，已带通+陷波）的 EEG-Conformer 编码器。
参考实现：<https://github.com/PRIMOCOSMOS/EEG_encoder> （SEED-IV 版本，单 global model，≈0.7-0.8M 参数）。

本仓库严格遵循配套设计文档 `Design.md` 的所有原则：

1. **多分卷 zip 流式合并** —— 32 个 5.37GB 分卷顺序拼接为 ≈160GB zip；**不落盘完整 zip**，使用流式 reader 按需提取 `.mat`，处理完即丢弃，确保在 ModelScope 实例 100GB 持久化盘下不爆。
2. **预处理** —— 已带通+陷波的数据 → 基线校正 → CAR 平均参考 →（可选 ICA）→ **每个 trial 居中 60%** → 4 秒窗口 50% 重叠 → **按通道 z-score** → 标签处理（7 类情绪 + 来自 `save_info` 的连续强度 ∈ [0,1]）。
3. **先切分、后处理** —— 划分以 **trial(clip) 为单位**（按被试 × session × trial 索引），先确定 train/val/test 的 trial 列表，再在各自集合内独立做窗口化和归一化，杜绝数据泄漏。
4. **样本均衡** —— 每个 clip 仅采样固定数量的居中窗口，避免长视频主导。
5. **模型** —— EEG-Conformer：时间卷积 + 空间深度卷积 + 池化 → Transformer 编码 → **双头输出**（7 类分类 + Sigmoid 强度回归）。参数量目标 0.7–0.8M。
6. **损失** —— `L = s_i · (α·L_cls + β·L_reg) + γ·L_rank`，先实现**退化方案**（先 `α + β`，`γ=0` 关闭排序损失），并支持后期开启 ranking loss；权重调度 + 余弦退火，最小 lr `1e-5`。
7. **训练策略** —— 先**只开 L_cls 预训练**若干 epoch，再联合训练；样本权重 = 连续标签 `s_i`，可设阈值 τ=0.5 弱样本降权。
8. **运行容错** —— 周期断点保存 / `--resume` 续训 / `--max-runtime-hours` 软超时退出（默认 10h，含优雅保存）。
9. **上传** —— 利用 `modelscope.hub.api.HubApi`（或 `MsDataset.upload`）把合并后的 zip 上传回 <https://www.modelscope.cn/datasets/DEREKVERSE/SEED-VII/>。

## 目录结构

```
seed_vii_encoder/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml                  # （可选）yaml 配置覆盖
├── src/
│   ├── __init__.py
│   ├── config.py                     # 所有默认超参（预处理 / 模型 / 训练）
│   ├── labels.py                     # SEED-VII 7类情绪 + session_sequences
│   ├── zip_stream.py                 # 32 分卷 → 流式 zip reader（不落盘）
│   ├── preprocess.py                 # CAR/ICA/居中裁剪/窗口化/z-score
│   ├── dataset.py                    # trial-level 切分 + 4s 窗口 IterableDataset
│   ├── model.py                      # EEG-Conformer（双头：分类+强度回归）
│   ├── losses.py                     # CE+LS、MSE、Margin Ranking、加权组合
│   ├── trainer.py                    # 训练/续训/早停/AMP/余弦退火/超时
│   ├── inference.py                  # 编码器推理：导出 embedding/预测
│   └── ms_upload.py                  # ModelScope 上传封装
├── scripts/
│   ├── merge_volumes.py              # 32 分卷拼接（流式校验）
│   ├── preprocess_to_npz.py          # 流式预处理 → .npz（或分片）
│   ├── train.py                      # 训练 / 续训入口
│   ├── encode.py                     # 推理脚本
│   └── upload_to_modelscope.py       # 上传到 ModelScope
└── notebooks/
    └── pipeline.ipynb                # 一键调用 4 个脚本的 Jupyter Notebook
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 流式合并 32 分卷（不落盘完整 zip；只生成 manifest，按需读取）
python scripts/merge_volumes.py --volumes-dir /data/volumes --pattern 'SEED-VII.zip.*'

# 2) 流式预处理 → npz（可选分片，避免一次性吃内存）
python scripts/preprocess_to_npz.py \
  --volumes-dir /data/volumes --pattern 'SEED-VII.zip.*' \
  --save-info-dir /data/save_info \
  --output /workspace/preprocessed/seed_vii.npz

# 3) 训练（先预训练分类，再联合训练；10h 自动安全保存）
python scripts/train.py \
  --data /workspace/preprocessed/seed_vii.npz \
  --output-dir /workspace/runs_seed_vii \
  --device auto --amp --max-runtime-hours 10 \
  --pretrain-epochs 10 --max-epochs 200

# 续训
python scripts/train.py --data ... --output-dir /workspace/runs_seed_vii --resume

# 4) 编码导出
python scripts/encode.py \
  --data /workspace/preprocessed/seed_vii.npz \
  --checkpoint /workspace/runs_seed_vii/best_encoder.pt \
  --output /workspace/encoded.npz

# 5) 上传到 ModelScope（zip 也可大文件分块）
python scripts/upload_to_modelscope.py \
  --local-file /workspace/SEED-VII.zip \
  --dataset DEREKVERSE/SEED-VII \
  --path-in-repo data/SEED-VII.zip
```

也可直接打开 `notebooks/pipeline.ipynb` 一键运行。
