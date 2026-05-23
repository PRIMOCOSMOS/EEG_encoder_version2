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

## 强烈推荐的执行顺序（100GB 持久化盘安全版）

由于 ModelScope 数据集**不能直接挂载**到实例工作区，而 SEED-VII 分卷合计 ≈160GB
但实例只有 100GB 持久化盘，所以推荐**先在 ModelScope 上把 32 分卷 stream-merge 成
一个完整的 `SEED-VII.zip` 并 push 回数据集**，再做后续所有处理。

```bash
# Step 0: 流式合并 + 上传（任意时刻磁盘 ≤ 1 卷 ≈ 5.4GB + 16MB 内存）
export MODELSCOPE_API_TOKEN=...
python scripts/merge_and_upload.py   --dataset DEREKVERSE/SEED-VII   --pattern '*.zip.*'   --path-in-repo SEED-VII.zip   --scratch-dir /workspace/_merge_scratch

# 全过程状态写入 /workspace/_merge_scratch/_merge_upload_state.json
# 中断后再跑即续传；尤其 LFS dedup 让"已上传 blob"会自动识别 → 跳过 Stage 3
```

完成后，**所有下游脚本默认从 `SEED-VII.zip` 用 HTTP Range 流式读**（磁盘 0 字节）：

```bash
# Step 1: 流式预处理（Range 模式，零磁盘）
python scripts/preprocess_to_npz.py   --ms-single-zip DEREKVERSE/SEED-VII   --ms-single-zip-path SEED-VII.zip   --ms-save-info-include 'save_info/*_save_info.csv'   --ms-scratch-dir /workspace/_ms_volumes_cache   --output /workspace/preprocessed/seed_vii.npz   --val-ratio 0.1 --test-ratio 0.1 --split-unit trial   --max-windows-per-trial 60

# Step 2: 训练（同前）
python scripts/train.py --data /workspace/preprocessed/seed_vii.npz   --output-dir /workspace/runs --device auto --amp --max-runtime-hours 10

# Step 3: 推理（同前）
python scripts/encode.py --data /workspace/preprocessed/seed_vii.npz   --checkpoint /workspace/runs/best_encoder.pt --output /workspace/encoded.npz
```

### 资源占用上界

| 阶段 | 磁盘 | 内存 | 网络 |
|---|---|---|---|
| Step 0 (merge+upload) | ≤ 1 卷 ≈ 5.4 GB | ~16 MB | 下载 2×160GB + 上传 1×160GB |
| Step 1 (preprocess)   | **0 字节** + npz 输出 | ~256 MB Range LRU | 下载 ≈160 GB |
| Step 2 (train)        | npz + ckpt + log | GPU 显存 + npz | 0（本地） |
| Step 3 (encode)       | npz + 输出 | GPU 显存 + npz | 0（本地） |

### 三种数据源对比

| 模式 | 何时用 | 磁盘 | 备注 |
|---|---|---|---|
| **C. `--ms-single-zip`** (推荐) | 已完成 Step 0 | 0 字节 | HTTP Range；LRU 缓存命中率高 |
| **B. `--ms-dataset`** | 未合并 / 不想合并 | ≤ 2 卷 ≈ 11 GB | LazyConcatStream，按需下载 |
| **A. `--volumes-dir`** | 数据已在本地 | 0 额外 | 兼容性最佳 |

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


## ModelScope 模式（实例不能挂载数据集时使用）

ModelScope 上的数据集**不能直接挂载**到实例工作区，必须通过 `MsDataset` / `HubApi` /
`dataset_file_download` 拉取。本仓库为此提供了：

- `src/ms_download.py` — 列表、单文件下载、`save_info` 批量下载、**`StreamingVolumeFetcher`**（一卷下载-用完-即删）。
- `src/zip_stream.py::LazyConcatStream` — 按需切片下载 + LRU 释放（任意时刻磁盘 ≤ N 个分卷）。
- `src/dataset.py::iter_trials_from_modelscope(...)` — 直接对接 ModelScope，再复用既有预处理管线。
- `scripts/ms_fetch.py` — `list / fetch-info / fetch-one / fetch-volumes` 四个子命令。
- 现有 `scripts/merge_volumes.py` 和 `scripts/preprocess_to_npz.py` 都新增了 `--ms-dataset` 入口；
  本地 / 远端两种来源**互斥可选**，其余参数不变。

磁盘占用：默认 `--ms-max-resident-volumes 2` → 任意时刻最多 ≈ 11 GB（远低于 100 GB 持久化盘）。

```bash
# 登录（任选其一）
export MODELSCOPE_API_TOKEN=xxxxxxxx
# 或者：python -c "from modelscope.hub.api import HubApi; HubApi().login('xxxxxxxx')"

# 1) 列出 + 校验远端分卷（不下载）
python scripts/merge_volumes.py   --ms-dataset DEREKVERSE/SEED-VII --pattern '*.zip.*'   --scratch-dir /workspace/_ms_scratch

# 2) 流式预处理 → npz（远端按需拉取，处理一个删一个）
python scripts/preprocess_to_npz.py   --ms-dataset DEREKVERSE/SEED-VII --pattern '*.zip.*'   --ms-scratch-dir /workspace/_ms_scratch   --ms-max-resident-volumes 2   --ms-save-info-include 'save_info/*_save_info.csv'   --output /workspace/preprocessed/seed_vii.npz

# 3 / 4) 训练、续训、编码导出（同本地模式，从 npz 读取，无需关心数据源）
python scripts/train.py  --data /workspace/preprocessed/seed_vii.npz --output-dir /workspace/runs --device auto --amp --max-runtime-hours 10
python scripts/encode.py --data /workspace/preprocessed/seed_vii.npz --checkpoint /workspace/runs/best_encoder.pt --output /workspace/encoded.npz

# 5) 上传（同前）
python scripts/upload_to_modelscope.py --local-file /workspace/SEED-VII.zip   --dataset DEREKVERSE/SEED-VII --path-in-repo data/SEED-VII.zip
```

辅助命令：

```bash
# 列出远端所有 .zip.* 分卷
python scripts/ms_fetch.py --dataset DEREKVERSE/SEED-VII list --pattern '*.zip.*'

# 只下载 save_info CSV（小文件）
python scripts/ms_fetch.py --dataset DEREKVERSE/SEED-VII   fetch-info --local-dir /workspace/save_info   --include 'save_info/*_save_info.csv'

# 烟雾测试：流式下载 + 立即删除每个分卷
python scripts/ms_fetch.py --dataset DEREKVERSE/SEED-VII   fetch-volumes --pattern '*.zip.*' --scratch-dir /workspace/_ms_scratch --keep 1 --delete-after
```

## Kaggle 训练流水线

仓库附了一套独立的 Kaggle Notebook：

- **`notebooks/kaggle_pipeline.ipynb`** — Kaggle 端到端流水线（预处理→训练→续训→编码导出→打包）
- **`notebooks/KAGGLE_DATA_SETUP.md`** — Kaggle 三份资源的准备说明

### Kaggle 需要的三份资源

| # | 内容 | 来源 |
|---|---|---|
| 1 | 20 个 `*.mat`（SEED-VII `EEG_preprocessed/`） | **公开** Kaggle Dataset，你在 Notebook 里 Add Data |
| 2 | `*_save_info.csv`（连续强度标签） | **你自己上传**的私有 Kaggle Dataset |
| 3 | 本仓库 (`src/`, `scripts/`) | 上传为 Kaggle Dataset / git clone / 粘代码 |

两个数据集挂载到 `/kaggle/input/` 下各自独立的子目录；Notebook 的第一个 cell **暴露了所有路径变量**，并带**容错逻辑**：
- 找不到 `EEG_MAT_SUBDIR` 会递归扫整个 mat 数据集挂载点
- 找不到 `SAVE_INFO_SUBDIR` 会回退到根目录扫 `*_save_info.csv`

trigger_info CSV **不强制需要**——SEED-VII 的 `EEG_preprocessed/*.mat` 已经按 trial 切好了。

### Kaggle vs ModelScope 流水线对比

| 维度 | ModelScope 流水线 | Kaggle 流水线 |
|---|---|---|
| 数据源 | 远端 zip 分卷 / 合并 zip | **本地 .mat 目录** (`--mat-dir`) |
| 入口脚本 | `merge_and_upload.py` + `preprocess_to_npz.py` | `preprocess_to_npz.py --mat-dir ...` |
| Notebook | `notebooks/pipeline.ipynb` | `notebooks/kaggle_pipeline.ipynb` |
| 超时 | 用户自管 (默认 10h) | **8.5h**（Kaggle 9h 限制留 buffer） |
| 输出 | 任意路径 | 必须落 `/kaggle/working/` 才能下载 |
| 数据集挂载 | 手动 SDK 拉 | **Kaggle 自动**挂到 `/kaggle/input/` |

### 4 种数据源（统一通过 `preprocess_to_npz.py` 互斥参数选择）

| 优先级 | 参数 | 适用场景 |
|---|---|---|
| **D**（Kaggle） | `--mat-dir <dir>` | 本地 .mat 文件夹（已挂载） |
| C | `--ms-single-zip <ds>` | ModelScope 合并 zip（HTTP Range，零磁盘） |
| B | `--ms-dataset <ds>` | ModelScope 多分卷（LRU 缓存，≤2 卷磁盘） |
| A | `--volumes-dir <dir>` | 本地分卷目录 |



## 字节级分卷的正确性说明（用户的实际分卷方式）

用户的 32 个 `*.zip.NNN` 是对**原始 zip 文件**做字节级顺序切片（与 `split -b` / `dd` 等价），
而**不是** zip 自带的多卷格式（`.z01/.z02/.../.zip`）。本仓库的 `ConcatStream` /
`LazyConcatStream` 正是为这种场景设计的：把 N 个分卷伪装成一个连续可 seek 的字节流，
直接喂给 `zipfile.ZipFile`，让 ZIP64 central directory 自然定位每个 `.mat`。

### 已实测验证

`src/zip_stream.py` 与 `src/dataset.py::iter_trials_from_modelscope` 已通过以下场景的实测：

1. **本地分卷拼接读取**：构造 14 MB 合成 zip → `split -b` 切成 8 个分卷 →
   `ConcatStream` 重读 20 个 `.mat`，**SHA-256 全部一致**。
2. **远端按需流式读取**：把 8 个分卷放在"远端目录"，用 `LazyConcatStream`
   + `max_resident=2` 重新读取 20 个 mat。结果：
   - 每个分卷**只下载 1 次**（零 churn）；
   - 末分卷被 `pin_last=True` 预热（zipfile 找 EOCD 时不会反复抓尾）；
   - 跨分卷的 mat 用临时 pin 保护，读完即释放；
   - `zf.close()` + `stream.close()` 后 **scratch 目录被完全清空**。
3. **完整端到端流水线**：用 monkey-patch 模拟 ModelScope，跑
   `iter_trials_from_modelscope`，6 个虚拟 trial 全部正确产出且 scratch 被清空。

### 关键设计要点

- **`LazyConcatStream.pin(idx, fetch_now=True)`**：构造时自动 pin **最后一个分卷**，
  避免 `zipfile.ZipFile(stream)` 一打开就触发"下载尾卷→满 LRU→驱逐→下次再下"循环。
- **`locate_members_in_stream` + `schedule_members_by_part`**：把所有 mat 按"在
  concat 流中的物理位置"排序后再读，使分卷按 1→N 单调遍历，**LRU 每个分卷只 fault-in 一次**。
- **跨分卷 mat 的临时 pin**：发现某 mat 的 `header_offset` 与 `header+data` 落在不同分卷时，
  临时 pin 末端分卷直到读完。
- **`zipfile.ZipFile.close()` 不会关闭传入的 file-like**：所以
  `iter_trials_from_modelscope` 的 `finally:` 块在关闭 zip 后**显式调用
  `stream.close()`**，触发 scratch 目录清空。
- **磁盘上界**：`(1 pinned last + max_resident LRU + 1 current) × volume_size`。
  默认 `max_resident=2` → 任意时刻 ≤ 4 卷 ≈ **21 GB**，远低于 100 GB 持久化盘。
  紧张时可设 `--ms-max-resident-volumes 1`，上限 ≈ 16 GB。
