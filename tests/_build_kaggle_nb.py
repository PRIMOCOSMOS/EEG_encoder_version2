"""Build the Kaggle pipeline notebook programmatically (so JSON is always valid).

Run once to regenerate `notebooks/kaggle_pipeline.ipynb`.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = []

cells.append(md("""# SEED-VII EEG-Conformer · Kaggle 训练流水线

**适用环境：Kaggle Notebook (T4×2 / P100 / TPU)**

Kaggle 上数据集会**自动挂载到 `/kaggle/input/<dataset-slug>/`**（只读），不需要 ModelScope SDK / zip 流式合并那一套。

## 数据来源（两个独立的 Kaggle Dataset）

| 用途 | 来源 | 你需要做的事 |
|---|---|---|
| **EEG 数据**（20 个 `*.mat`）| **公开** Kaggle Dataset（已存在） | 在 Notebook 里 **Add Data** 搜索并添加 |
| **连续标签** (`*_save_info.csv`) | **你自己上传**的 Kaggle Dataset | 自己上传 + Add Data |

`trigger_info.csv` 文件不需要——SEED-VII 的 `EEG_preprocessed/*.mat` 已经按 trial 切好了。

## 流程

0. 定位仓库代码 + 配置两个数据集路径（**第一个 cell 必改**）
1. 安装依赖 + 检测 GPU
2. 预处理：mat → npz（trial-level 切分；居中 60% + 4s 窗口 + 50% 重叠 + 按通道 z-score）
3. 训练（先分类预训练 → 联合训练；**8.5 小时软超时**避免被 Kaggle 强杀）
4. 断点续训（同一 cell 重新跑即可）
5. 编码导出 embedding
6. 把产物打包到 `/kaggle/working/`（关闭 session 后可下载）

> 严格遵循 `Design.md` 所有原则；模型 ≈ 0.778M 参数。"""))

cells.append(md("""## 0. 仓库代码 + 数据路径配置（**两份数据各自独立配置**）

仓库代码三选一：
- A. 把仓库本身上传成第三个 Kaggle Dataset（推荐，离线）
- B. 用 `git clone` 临时拉（需要在 Notebook Settings → Internet 打开网络）
- C. 把 `src/` 和 `scripts/` 直接粘进 Kaggle Notebook 同目录"""))

cells.append(code('''import os, sys, subprocess, json, pathlib, shutil, time

# ============================================================================
#  必改：把这三个路径换成 Kaggle Add Data 后的实际路径
# ============================================================================

# A. 仓库代码位置
REPO_CANDIDATES = [
    '/kaggle/input/seed-vii-encoder/seed_vii_encoder',
    '/kaggle/input/seed-vii-encoder',
    '/kaggle/working/seed_vii_encoder',
]
REPO_GIT_URL = ''   # 留空跳过；填上则在 REPO_CANDIDATES 都失败时 git clone

# B. EEG mat 数据集（公开 Kaggle Dataset 挂载点）
#    示例 slug: 'seed-vii-eeg-preprocessed'
#    挂载点:    /kaggle/input/<slug>/...
#    把 EEG_MAT_DATASET 改成你 Add Data 后的实际目录名。
EEG_MAT_DATASET_DIR = '/kaggle/input/<eeg-public-dataset-slug>'
#    .mat 文件相对 EEG_MAT_DATASET_DIR 的子目录（公开数据集通常是
#    'EEG_preprocessed' 或直接在根目录；脚本会自动递归搜索）。
EEG_MAT_SUBDIR      = 'EEG_preprocessed'

# C. save_info 数据集（你自己上传的，默认放在 save_info/ 子目录里）
#    示例：/kaggle/input/seed-vii-save-info/save_info/<...>.csv
SAVE_INFO_DATASET_DIR = '/kaggle/input/<your-save-info-dataset-slug>'
SAVE_INFO_SUBDIR      = 'save_info'

# D. 输出路径（Kaggle 关闭后只有 /kaggle/working 能下载）
WORK    = pathlib.Path('/kaggle/working/seed_vii_work'); WORK.mkdir(parents=True, exist_ok=True)
NPZ_OUT = str(WORK / 'preprocessed' / 'seed_vii.npz')
RUN_DIR = str(WORK / 'runs_seed_vii')
ENC_OUT = str(WORK / 'encoded' / 'embeddings.npz')

# E. 训练超参
MAX_RUNTIME_HOURS = 8.5     # Kaggle 9h 限制，留 30 分钟兜底
BATCH_SIZE        = 64
PRETRAIN_EPOCHS   = 10
MAX_EPOCHS        = 200
ENABLE_RANKING    = False   # 退化方案：先关 ranking loss

# ============================================================================
#  自动定位 & 校验
# ============================================================================

def _find_repo():
    for cand in REPO_CANDIDATES:
        if (pathlib.Path(cand) / 'scripts' / 'preprocess_to_npz.py').is_file():
            return pathlib.Path(cand).resolve()
    if REPO_GIT_URL:
        target = pathlib.Path('/kaggle/working/seed_vii_encoder')
        if not target.exists():
            print(f'Cloning {REPO_GIT_URL} -> {target}')
            subprocess.check_call(['git', 'clone', '--depth', '1', REPO_GIT_URL, str(target)])
        return target.resolve()
    raise SystemExit('Repo not found. Add it as a Kaggle dataset OR set REPO_GIT_URL.')


def _resolve_mat_dir(root: str, subdir: str) -> str:
    """容错：如果 subdir 存在用 subdir；否则在 root 下递归找含 *.mat 的目录。"""
    root_p = pathlib.Path(root)
    if not root_p.exists():
        raise SystemExit(f'EEG dataset not mounted: {root}\\n'
                          'In the Notebook sidebar → Add Data → search & add the dataset.')
    candidate = root_p / subdir
    if candidate.is_dir() and list(candidate.rglob('*.mat')):
        return str(candidate)
    # fallback: anywhere under root containing *.mat
    for d in [root_p, *root_p.rglob('*')]:
        if d.is_dir() and any(p.suffix.lower() == '.mat' for p in d.iterdir() if p.is_file()):
            return str(d)
    raise SystemExit(f'No *.mat under {root}; check EEG_MAT_SUBDIR or dataset layout.')


def _resolve_save_info_dir(root: str, subdir: str) -> str:
    """容错：CSV 可能在 subdir/ 或直接在 root/。"""
    root_p = pathlib.Path(root)
    if not root_p.exists():
        raise SystemExit(f'save_info dataset not mounted: {root}\\n'
                          'In the Notebook sidebar → Add Data → search & add your own dataset.')
    candidate = root_p / subdir
    if candidate.is_dir() and list(candidate.rglob('*_save_info.csv')):
        return str(candidate)
    if list(root_p.rglob('*_save_info.csv')):
        return str(root_p)
    raise SystemExit(f'No *_save_info.csv under {root} or {root}/{subdir}')


REPO          = _find_repo()
MAT_DIR       = _resolve_mat_dir(EEG_MAT_DATASET_DIR, EEG_MAT_SUBDIR)
SAVE_INFO_DIR = _resolve_save_info_dir(SAVE_INFO_DATASET_DIR, SAVE_INFO_SUBDIR)

print('REPO          =', REPO)
print('MAT_DIR       =', MAT_DIR)
print('SAVE_INFO_DIR =', SAVE_INFO_DIR)

# Sanity counts
n_mat  = len(list(pathlib.Path(MAT_DIR).rglob('*.mat')))
n_save = len(list(pathlib.Path(SAVE_INFO_DIR).rglob('*_save_info.csv')))
n_trig = len(list(pathlib.Path(SAVE_INFO_DIR).rglob('*_trigger_info.csv')))
print(f'\\n.mat files: {n_mat} (expect 20)')
print(f'save_info CSVs: {n_save} (expect 4 × 20 = 80 for full SEED-VII)')
print(f'trigger_info CSVs (not used by this pipeline): {n_trig}')
assert n_mat > 0, 'No mat files found — check EEG_MAT_DATASET_DIR / EEG_MAT_SUBDIR.'
assert n_save > 0, 'No save_info CSVs found — check SAVE_INFO_DATASET_DIR / SAVE_INFO_SUBDIR.'


def run(cmd, env=None):
    print('$', ' '.join(map(str, cmd)))
    res = subprocess.run([str(x) for x in cmd], cwd=str(REPO),
                         env={**os.environ, **(env or {})})
    if res.returncode != 0:
        raise SystemExit(f'cmd failed with code {res.returncode}')

PY = sys.executable
print('\\npython =', PY)'''))

cells.append(md("## 1. 安装依赖（Kaggle 大多数包已预装，缺啥补啥）"))

cells.append(code('''subprocess.run([PY, '-m', 'pip', 'install', '--quiet',
                '-r', str(REPO / 'requirements.txt')])

import torch
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(),
      'devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}] {torch.cuda.get_device_name(i)}')'''))

cells.append(md("""## 2. 预处理：本地 mat 目录 → 单个 npz

- **trial-level 切分**：先按 (subject, session, trial) 决定 train/val/test，再各自做窗口化和按通道 z-score
- 每个 clip 取**居中 60%** → 4 秒窗口 → 50% 重叠
- 每 clip 最多保留 60 个窗口（防长视频主导）
- 连续强度 ∈ [0,1] 来自 `*_save_info.csv`"""))

cells.append(code('''run([PY, 'scripts/preprocess_to_npz.py',
     '--mat-dir', MAT_DIR,
     '--mat-pattern', '*.mat',
     '--save-info-dir', SAVE_INFO_DIR,
     '--output', NPZ_OUT,
     '--val-ratio', '0.1',
     '--test-ratio', '0.1',
     '--split-unit', 'trial',
     '--max-windows-per-trial', '60'])

import numpy as np
z = np.load(NPZ_OUT, allow_pickle=True)
print('keys:', list(z.files))
print('X:', z['X'].shape, z['X'].dtype)
print('y:', z['y'].shape, '   unique labels:', sorted(set(z['y'].tolist())))
print('s:', z['s'].shape, '   range:', float(z['s'].min()), '-', float(z['s'].max()))
for split in ('train', 'val', 'test'):
    key = f'split_{split}'
    if key in z.files:
        print(f'  {split}: {len(z[key])} windows')'''))

cells.append(md("""## 3. 训练（带 8.5h 软超时 + 周期断点）

- 先用**仅分类损失**预训练 10 epoch
- 再开启**联合损失**：分类 + 强度回归（默认 ranking 关闭，退化方案）
- 余弦退火 lr → 1e-5
- 周期断点 `train_state.pt`，进程被 Kaggle 杀掉也不丢"""))

cells.append(code('''cmd = [PY, 'scripts/train.py',
       '--data', NPZ_OUT,
       '--output-dir', RUN_DIR,
       '--device', 'auto', '--amp',
       '--pretrain-epochs', PRETRAIN_EPOCHS,
       '--max-epochs', MAX_EPOCHS,
       '--batch-size', BATCH_SIZE,
       '--lr', '2e-4', '--min-lr', '1e-5',
       '--sample-weight-mode', 'continuous',
       '--max-runtime-hours', MAX_RUNTIME_HOURS]
if ENABLE_RANKING:
    cmd += ['--enable-rank', '--gamma-rank-end', '0.8']
run(cmd)'''))

cells.append(md("""## 4. 续训（被超时打断后用）

**直接重跑这个 cell 即可从断点继续**——`train_state.pt` 自动加载，新 session 再吃 8.5 小时。"""))

cells.append(code('''RESUME = False     # 改成 True 即跑续训
if RESUME:
    cmd = [PY, 'scripts/train.py',
           '--data', NPZ_OUT,
           '--output-dir', RUN_DIR,
           '--device', 'auto', '--amp',
           '--resume',
           '--max-runtime-hours', MAX_RUNTIME_HOURS]
    if ENABLE_RANKING:
        cmd += ['--enable-rank', '--gamma-rank-end', '0.8']
    run(cmd)
else:
    print('Skip resume (set RESUME=True if you need it).')'''))

cells.append(md("## 5. 编码导出"))

cells.append(code('''best_encoder = pathlib.Path(RUN_DIR) / 'best_encoder.pt'
assert best_encoder.is_file(), f'No checkpoint at {best_encoder}'
run([PY, 'scripts/encode.py',
     '--data', NPZ_OUT,
     '--checkpoint', str(best_encoder),
     '--output', ENC_OUT,
     '--feature-type', 'projected',
     '--device', 'auto', '--amp'])
import numpy as np
z = np.load(ENC_OUT, allow_pickle=True)
print('output keys:', list(z.files))
for k in z.files:
    obj = z[k]
    try:
        print(f'  {k}: shape={obj.shape}, dtype={obj.dtype}')
    except Exception:
        print(f'  {k}: {obj}')'''))

cells.append(md("""## 6. 打包产物到 /kaggle/working（关闭后可下载）

Kaggle 关闭 session 后只有 `/kaggle/working/` 里的文件能保留并下载。这里把：
- 最佳模型 / 编码器 / 训练状态
- 训练日志 / 训练摘要
- 编码后 embedding

打成一个 zip 放在 `/kaggle/working/` 根目录。"""))

cells.append(code('''import zipfile, glob
out_zip = pathlib.Path('/kaggle/working/seed_vii_outputs.zip')
to_pack = []
for pat in [
    f'{RUN_DIR}/best_model.pt',
    f'{RUN_DIR}/best_encoder.pt',
    f'{RUN_DIR}/last_model.pt',
    f'{RUN_DIR}/train_state.pt',
    f'{RUN_DIR}/summary.json',
    f'{RUN_DIR}/train_config.json',
    f'{RUN_DIR}/train.log',
    ENC_OUT,
]:
    to_pack.extend(glob.glob(pat))

with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for f in to_pack:
        zf.write(f, arcname=pathlib.Path(f).relative_to('/kaggle/working').as_posix())

size_mb = out_zip.stat().st_size / 1024 / 1024
print(f'packed {len(to_pack)} files -> {out_zip} ({size_mb:.1f} MB)')
print('Save & Run All -> the zip is in the notebook Output tab for download.')'''))

cells.append(md("""## (可选) 7. 把 checkpoint 推回 ModelScope

如果想把 Kaggle 训出来的 checkpoint 同步到 ModelScope dataset / model repo："""))

cells.append(code('''PUSH_TO_MS   = False
MS_TOKEN     = ''                                # 粘贴你的 SDK token
MS_DATASET   = 'YOUR_NAMESPACE/YOUR_DATASET'
PATH_IN_REPO = 'kaggle_runs/best_encoder.pt'

if PUSH_TO_MS:
    os.environ['MODELSCOPE_API_TOKEN'] = MS_TOKEN
    run([PY, 'scripts/upload_to_modelscope.py',
         '--local-file', str(pathlib.Path(RUN_DIR) / 'best_encoder.pt'),
         '--dataset', MS_DATASET,
         '--path-in-repo', PATH_IN_REPO])'''))

nb.cells = cells
out_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'kaggle_pipeline.ipynb'
nbf.write(nb, str(out_path))
print(f'OK -> {out_path}')
