"""ModelScope upload helpers.

参考 https://github.com/modelscope/modelscope/issues/985 + `modelscope.hub.api`。
两种途径：
    A) HubApi.upload_file / upload_folder（推荐，类似 huggingface_hub 风格，支持大文件）
    B) MsDataset.upload（旧 API，按 namespace/dataset_name + object_name）

我们优先用 HubApi；若环境版本不带 upload_file，则回退到 MsDataset.upload。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def login(token: Optional[str] = None) -> None:
    """Login to ModelScope. Token can be passed or from env MODELSCOPE_API_TOKEN."""
    from modelscope.hub.api import HubApi
    token = token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "ModelScope token not provided. Set MODELSCOPE_API_TOKEN env var or pass token=..."
        )
    HubApi().login(token)


def upload_file_to_dataset(
    local_file: str,
    dataset_id: str,
    path_in_repo: str,
    token: Optional[str] = None,
    commit_message: str = "upload via SDK",
) -> None:
    """Upload one file to a ModelScope **dataset** repo.

    dataset_id : 'NAMESPACE/DATASET'  e.g. 'DEREKVERSE/SEED-VII'
    path_in_repo: target path inside the dataset, e.g. 'data/SEED-VII.zip'
    """
    local = Path(local_file)
    if not local.is_file():
        raise FileNotFoundError(local)

    from modelscope.hub.api import HubApi
    api = HubApi()
    tk = token or os.environ.get("MODELSCOPE_API_TOKEN")
    if tk:
        api.login(tk)

    # Try the modern HubApi.upload_file (>= 1.10), repo_type='dataset'
    if hasattr(api, "upload_file"):
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        print(f"[OK] uploaded via HubApi.upload_file -> {dataset_id}:{path_in_repo}")
        return

    # Fallback: MsDataset.upload (object-based)
    from modelscope.msdatasets import MsDataset
    namespace, dataset_name = dataset_id.split("/", 1)
    MsDataset.upload(
        object_name=path_in_repo,
        local_file_path=str(local),
        dataset_name=dataset_name,
        namespace=namespace,
    )
    print(f"[OK] uploaded via MsDataset.upload -> {dataset_id}:{path_in_repo}")


def upload_folder_to_dataset(
    local_dir: str,
    dataset_id: str,
    path_in_repo: str = "",
    token: Optional[str] = None,
    commit_message: str = "upload folder via SDK",
    allow_patterns: Optional[list] = None,
    ignore_patterns: Optional[list] = None,
) -> None:
    """Upload a whole folder to a ModelScope dataset repo (preferred for many small files)."""
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
            ignore_patterns=ignore_patterns,
        )
        print(f"[OK] uploaded folder via HubApi.upload_folder -> {dataset_id}:{path_in_repo or '/'}")
        return

    # Fallback: iterate files
    from modelscope.msdatasets import MsDataset
    namespace, dataset_name = dataset_id.split("/", 1)
    for f in local.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(local).as_posix()
        target = f"{path_in_repo.rstrip('/')}/{rel}" if path_in_repo else rel
        if allow_patterns and not any(_glob_match(rel, p) for p in allow_patterns):
            continue
        if ignore_patterns and any(_glob_match(rel, p) for p in ignore_patterns):
            continue
        MsDataset.upload(
            object_name=target,
            local_file_path=str(f),
            dataset_name=dataset_name,
            namespace=namespace,
        )
        print(f"[OK] uploaded {rel}")


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(name, pattern)
