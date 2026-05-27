"""ModelScope dataset utilities."""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional


def login_if_token(token: Optional[str] = None):
    if not token:
        token = os.environ.get("MODELSCOPE_API_TOKEN", "")
    if token:
        try:
            from modelscope import MsDataset
            print("[INFO] ModelScope token found")
        except ImportError:
            print("[WARN] modelscope not installed, skipping login")


def download_one_file(dataset_id: str, file_path: str, local_dir: str,
                      revision: str = "master", token: Optional[str] = None) -> str:
    """Download a single file from a ModelScope dataset using get_dataset_file_url + requests.

    替代已废弃的 download_dataset_file() API。
    """
    import requests
    from modelscope.hub.api import HubApi

    token = token or os.environ.get("MODELSCOPE_API_TOKEN", "")
    api = HubApi(token=token)

    # 解析 namespace / dataset_name
    parts = dataset_id.split("/")
    namespace = parts[0]
    ds_name = parts[1] if len(parts) > 1 else dataset_id

    url = api.get_dataset_file_url(
        file_name=file_path,
        dataset_name=ds_name,
        namespace=namespace,
        revision=revision,
    )

    local = Path(local_dir) / file_path.replace("/", "_").replace("\\", "_")
    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists() and local.stat().st_size > 0:
        print(f"[SKIP] {local.name} already exists")
        return str(local)

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(local, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[OK] Downloaded {dataset_id}/{file_path} -> {local}")
    return str(local)


def download_save_info(dataset_id: str, local_dir: str,
                       revision: str = "master", token: Optional[str] = None,
                       include: Optional[List[str]] = None):
    """Download save_info CSV files from a ModelScope dataset."""
    print(f"[INFO] save_info download: use the full pipeline in Cell 4 of the Notebook")
    return str(local_dir)