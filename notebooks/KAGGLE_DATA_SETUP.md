# Kaggle 数据集准备指南

Kaggle 流水线需要**两个**数据集和**一份**代码：

| # | 内容 | 来源 | 你做什么 |
|---|---|---|---|
| 1 | 20 个 SEED-VII `*.mat` (`EEG_preprocessed/`) | **公开** Kaggle Dataset | 在 Notebook 里 **Add Data** 添加 |
| 2 | `*_save_info.csv`（连续强度标签） | **你自己上传**的 Kaggle Dataset | 自己创建并上传 + Add Data |
| 3 | 本仓库（`src/`, `scripts/`） | 你自己 | A/B/C 三选一（见后） |

---

## 1. EEG 数据（公开 Dataset）

在 Kaggle 上搜索 SEED-VII 相关的 EEG_preprocessed 公开数据集，点 **Add Data** 添加到你的 Notebook。挂载后路径形如：

```
/kaggle/input/<slug>/EEG_preprocessed/
    ├── 1.mat
    ├── 2.mat
    ├── ...
    └── 20.mat
```

**在 Notebook 的 cell 0 改这两个变量**：

```python
EEG_MAT_DATASET_DIR = '/kaggle/input/<slug>'            # 公开 dataset 的挂载点
EEG_MAT_SUBDIR      = 'EEG_preprocessed'                # mat 在 dataset 内的子目录
```

Notebook 有**容错逻辑**：如果 `EEG_MAT_SUBDIR` 找不到 `.mat`，会递归扫整个挂载点找第一个含 `.mat` 的目录。所以即使你写错子目录名通常也能正常工作。

---

## 2. save_info 数据集（你自己上传）

### 2.1 在本地准备目录

```
my-seed-vii-save-info/                     ← 这是你 Kaggle Dataset 的根
└── save_info/
    ├── 1_20221001_1_save_info.csv
    ├── 1_20221001_2_save_info.csv
    ├── 1_20221001_3_save_info.csv
    ├── 1_20221001_4_save_info.csv
    ├── 2_20221002_1_save_info.csv
    ├── ...
    └── 20_<date>_4_save_info.csv

  共 20 被试 × 4 session = 80 个 CSV
```

文件名严格按 `<subjectID>_<date>_<sessionID>_save_info.csv`。其中 `_save_info.csv` 是后缀（用来与 `_trigger_info.csv` 区分），本流水线只关心 `_save_info.csv`。

### 2.2 上传成 Kaggle Dataset

最快的方式是网页上传：

1. https://www.kaggle.com/datasets → **+ New Dataset**
2. 拖入 `my-seed-vii-save-info/save_info/` 整个文件夹
3. 起个 slug，比如 `seed-vii-save-info`，**Visibility 选 Private**

或用 `kaggle` CLI：

```bash
cd my-seed-vii-save-info
kaggle datasets init -p .
# 编辑 dataset-metadata.json 里的 title/id
kaggle datasets create -p .
```

### 2.3 挂到 Notebook

在 Notebook 侧栏 **Add Data** 搜索你刚上传的 Dataset，添加。挂载后：

```
/kaggle/input/seed-vii-save-info/save_info/*_save_info.csv
```

**在 Notebook 的 cell 0 改这两个变量**：

```python
SAVE_INFO_DATASET_DIR = '/kaggle/input/seed-vii-save-info'
SAVE_INFO_SUBDIR      = 'save_info'
```

容错：如果 CSV 直接在 dataset 根（没 `save_info/` 子目录），脚本也会自动识别。

---

## 3. 仓库代码（三选一）

### A. 上传仓库为 Kaggle Dataset（**推荐，最省事**）

```bash
cd seed_vii_encoder
kaggle datasets init -p .
# 改 metadata 里的 id 为 <your-username>/seed-vii-encoder
kaggle datasets create -p .
```

Notebook → Add Data → 加入这个 dataset。挂载后路径：

```
/kaggle/input/seed-vii-encoder/<your repo contents>
```

Notebook 默认的 `REPO_CANDIDATES` 已经覆盖了这种情况，无需修改。

### B. 在 Notebook 里 `git clone`

Notebook → Settings → Internet → **ON**，然后 cell 0 设：

```python
REPO_GIT_URL = 'https://github.com/<你的用户名>/seed_vii_encoder.git'
```

仓库会被 clone 到 `/kaggle/working/seed_vii_encoder/`。

### C. 直接把代码粘进 Notebook

每个 `src/*.py` 和 `scripts/*.py` 都开一个 cell 写文件。不推荐，难维护。

---

## 4. 一份完整的 cell 0 示例配置

```python
# 仓库
REPO_CANDIDATES = [
    '/kaggle/input/seed-vii-encoder/seed_vii_encoder',
    '/kaggle/input/seed-vii-encoder',
    '/kaggle/working/seed_vii_encoder',
]
REPO_GIT_URL = ''

# EEG mat（公开 dataset）
EEG_MAT_DATASET_DIR = '/kaggle/input/seed-vii-eeg-preprocessed'
EEG_MAT_SUBDIR      = 'EEG_preprocessed'

# save_info（你的私有 dataset）
SAVE_INFO_DATASET_DIR = '/kaggle/input/seed-vii-save-info'
SAVE_INFO_SUBDIR      = 'save_info'

# 输出
WORK    = pathlib.Path('/kaggle/working/seed_vii_work')
NPZ_OUT = str(WORK / 'preprocessed' / 'seed_vii.npz')
RUN_DIR = str(WORK / 'runs_seed_vii')
ENC_OUT = str(WORK / 'encoded' / 'embeddings.npz')

# 训练
MAX_RUNTIME_HOURS = 8.5
BATCH_SIZE        = 64
PRETRAIN_EPOCHS   = 10
MAX_EPOCHS        = 200
ENABLE_RANKING    = False
```

---

## 5. save_info CSV 容错说明

`src/dataset.py::_parse_save_info_csv` 自动识别以下几种 CSV 形式：

- 单列 `score` 表头 + 20 行 0-1 数字 ← 推荐
- 表头里有 `score / intensity / rating / feedback` 列 → 取该列
- 无表头 + 单列 20 行数字 → 直接取
- 单行 × 20 列 → 横排也行
- 0-5 / 0-100 量纲会自动归一到 [0,1]

如果某个 trial 没有对应 save_info 条目，使用默认值 `1.0`。

---

## 6. 关于 `*_trigger_info.csv`

trigger_info 给了 trial 的起止时间。本流水线**不需要**它，因为 SEED-VII 的 `EEG_preprocessed/*.mat` 已经按 trial 切好了。你的 save_info 数据集**可以包含也可以不包含** trigger_info CSV，预处理脚本会自动忽略。

如果你只想上传 save_info 不上传 trigger_info，更省空间。
