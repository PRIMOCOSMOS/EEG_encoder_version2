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
    """Download a single file from a ModelScope dataset.

    Returns the local path to the downloaded file.
    """
    try:
        from modelscope.hub.api import HubApi
        api = HubApi(token=token if token else os.environ.get("MODELSCOPE_API_TOKEN", ""))
        local = Path(local_dir) / file_path.replace("/", "_")
        local.parent.mkdir(parents=True, exist_ok=True)
        api.download_dataset_file(
            dataset_name=dataset_id,
            file_path=file_path,
            revision=revision,
            target_path=str(local),
        )
        print(f"[OK] Downloaded {dataset_id}/{file_path} -> {local}")
        return str(local)
    except Exception as e:
        raise RuntimeError(f"ModelScope download failed: {e}")


def download_save_info(dataset_id: str, local_dir: str,
                       revision: str = "master", token: Optional[str] = None,
                       include: Optional[List[str]] = None):
    """Download save_info CSV files from a ModelScope dataset."""
    print(f"[INFO] save_info download not implemented — set --save-info-dir manually if needed")
    return str(local_dir)