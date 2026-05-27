"""ModelScope dataset I/O utilities.

统一封装 ModelScope SDK 的下载和上传操作：
- download_one_file: 下载数据集中的单个文件
- upload_file_to_dataset: 上传文件到数据集
- upload_folder_to_dataset: 上传文件夹到数据集
- list_dataset_files: 列出数据集中的文件

API 参考 (2025/2026 版本 modelscope SDK):
- HubApi().login(token)
- HubApi().upload_file(path_or_fileobj, path_in_repo, repo_id, repo_type='dataset')
- HubApi().upload_folder(folder_path, path_in_repo, repo_id, repo_type='dataset')
- HubApi().get_dataset_file_url(file_name, dataset_name, namespace, revision)
- modelscope.hub.file_download.model_file_download(repo_id, file_path, repo_type='dataset')
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def login(token: Optional[str] = None) -> None:
    """Login to ModelScope using token or env MODELSCOPE_API_TOKEN."""
    from modelscope.hub.api import HubApi
    token = token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "No ModelScope token. Set MODELSCOPE_API_TOKEN env or pass token=..."
        )
    api = HubApi()
    api.login(token)
    print("[OK] ModelScope login successful")


def download_dataset_file(
    dataset_id: str,
    file_path: str,
    local_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
) -> str:
    """Download a single file from a ModelScope dataset.

    Uses modelscope.hub.file_download.model_file_download with repo_type='dataset'.
    Falls back to HubApi.get_dataset_file_url + requests if needed.

    Returns the local file path.
    """
    token = token or os.environ.get("MODELSCOPE_API_TOKEN", "")

    # Method 1: Use the SDK's built-in download function (preferred)
    try:
        from modelscope.hub.file_download import model_file_download
        local_path = model_file_download(
            model_id=dataset_id,
            file_path=file_path,
            revision=revision,
            local_dir=local_dir,
            repo_type='dataset',
        )
        print(f"[OK] Downloaded {dataset_id}/{file_path} -> {local_path}")
        return str(local_path)
    except Exception as e1:
        print(f"[INFO] SDK download failed ({e1}), trying fallback...")

    # Method 2: HubApi.get_dataset_file_url + requests
    import requests
    from modelscope.hub.api import HubApi

    api = HubApi()
    if token:
        api.login(token)

    parts = dataset_id.split("/")
    namespace = parts[0]
    ds_name = parts[1] if len(parts) > 1 else dataset_id

    url = api.get_dataset_file_url(
        file_name=file_path,
        dataset_name=ds_name,
        namespace=namespace,
        revision=revision,
    )

    # Determine local filename
    local = Path(local_dir) / Path(file_path).name
    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists() and local.stat().st_size > 0:
        print(f"[SKIP] {local.name} already exists ({local.stat().st_size} bytes)")
        return str(local)

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(local, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[OK] Downloaded {dataset_id}/{file_path} -> {local}")
    return str(local)


def upload_file_to_dataset(
    local_file: str,
    dataset_id: str,
    path_in_repo: str,
    token: Optional[str] = None,
    commit_message: str = "upload via SDK",
) -> None:
    """Upload one file to a ModelScope dataset repo.

    dataset_id : 'NAMESPACE/DATASET' e.g. 'DEREKVERSE/SEED-VII'
    path_in_repo: target path inside the dataset, e.g. 'preprocessed_npz/1.npz'
    """
    local = Path(local_file)
    if not local.is_file():
        raise FileNotFoundError(local)

    from modelscope.hub.api import HubApi
    api = HubApi()
    tk = token or os.environ.get("MODELSCOPE_API_TOKEN")
    if tk:
        api.login(tk)

    # Try modern API first (>= 1.10)
    if hasattr(api, "upload_file"):
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        print(f"[OK] Uploaded {local.name} -> {dataset_id}:{path_in_repo}")
        return

    # Fallback: MsDataset.upload
    from modelscope.msdatasets import MsDataset
    namespace, dataset_name = dataset_id.split("/", 1)
    MsDataset.upload(
        object_name=path_in_repo,
        local_file_path=str(local),
        dataset_name=dataset_name,
        namespace=namespace,
    )
    print(f"[OK] Uploaded {local.name} -> {dataset_id}:{path_in_repo}")


def upload_folder_to_dataset(
    local_dir: str,
    dataset_id: str,
    path_in_repo: str = "",
    token: Optional[str] = None,
    commit_message: str = "upload folder via SDK",
    allow_patterns: Optional[List[str]] = None,
) -> None:
    """Upload a whole folder to a ModelScope dataset repo."""
    local = Path(local_dir)
    if not local.is_dir():
        raise NotADirectoryError(local)

    from modelscope.hub.api import HubApi
    api = HubApi()
    tk = token or os.environ.get("MODELSCOPE_API_TOKEN")
    if tk:
        api.login(tk)

    if hasattr(api, "upload_folder"):
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=commit_message,
            allow_patterns=allow_patterns,
        )
        print(f"[OK] Uploaded folder -> {dataset_id}:{path_in_repo or '/'}")
        return

    # Fallback: iterate files
    import fnmatch
    from modelscope.msdatasets import MsDataset
    namespace, dataset_name = dataset_id.split("/", 1)
    for f in sorted(local.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(local).as_posix()
        if allow_patterns and not any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
            continue
        target = f"{path_in_repo.rstrip('/')}/{rel}" if path_in_repo else rel
        MsDataset.upload(
            object_name=target,
            local_file_path=str(f),
            dataset_name=dataset_name,
            namespace=namespace,
        )
        print(f"[OK] Uploaded {rel}")
