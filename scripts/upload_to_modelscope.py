#!/usr/bin/env python3
"""Upload an artifact (typically the merged 160G SEED-VII.zip) to a ModelScope dataset repo.

Token via:
  export MODELSCOPE_API_TOKEN=xxxxxxxx
or pass --token.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ms_upload import upload_file_to_dataset, upload_folder_to_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Upload to a ModelScope dataset")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-file", type=str, help="single file to upload")
    src.add_argument("--local-dir", type=str, help="directory to upload")
    p.add_argument("--dataset", default="DEREKVERSE/SEED-VII",
                   help="dataset id 'NAMESPACE/NAME'")
    p.add_argument("--path-in-repo", required=True,
                   help="target path inside the dataset repo")
    p.add_argument("--token", default="", help="API token (overrides env)")
    p.add_argument("--commit-message", default="upload via SDK")
    p.add_argument("--allow", nargs="*", default=None, help="(folder mode) allow_patterns")
    p.add_argument("--ignore", nargs="*", default=None, help="(folder mode) ignore_patterns")
    args = p.parse_args()

    token = args.token or None
    if args.local_file:
        upload_file_to_dataset(
            local_file=args.local_file,
            dataset_id=args.dataset,
            path_in_repo=args.path_in_repo,
            token=token,
            commit_message=args.commit_message,
        )
    else:
        upload_folder_to_dataset(
            local_dir=args.local_dir,
            dataset_id=args.dataset,
            path_in_repo=args.path_in_repo,
            token=token,
            commit_message=args.commit_message,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
        )


if __name__ == "__main__":
    main()
