#!/usr/bin/env python3
"""ModelScope-side helper utilities.

Subcommands:
  list           List remote files (optionally globbed) in a ModelScope dataset.
  fetch-info     Download all save_info CSVs (small files) for continuous labels.
  fetch-one      Download a single file by remote path (for testing).
  fetch-volumes  Stream-download every split volume, one at a time, optionally
                 concatenating into one file (use `merge_volumes.py --ms-dataset` instead
                 for the canonical concat workflow).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ms_download import (  # noqa: E402
    StreamingVolumeFetcher, discover_remote_volumes, download_one_file,
    download_save_info, filter_files, list_dataset_files, login_if_token,
)


def cmd_list(args):
    login_if_token(args.token or None)
    files = list_dataset_files(args.dataset, revision=args.revision, token=(args.token or None))
    if args.pattern:
        files = filter_files(files, include=[args.pattern])
    total = 0
    for f in files:
        sz = int(f.get("Size", 0) or 0)
        total += sz
        print(f"{sz:>14d}  {f.get('Path')}")
    print(f"[OK] {len(files)} files, total {total/1024/1024/1024:.2f} GB")


def cmd_fetch_info(args):
    login_if_token(args.token or None)
    include = [p.strip() for p in args.include.split(",")] if args.include else None
    out = download_save_info(
        dataset_id=args.dataset, local_dir=args.local_dir,
        revision=args.revision, token=(args.token or None),
        include=include,
    )
    print(f"[OK] save_info CSVs downloaded -> {out}")


def cmd_fetch_one(args):
    login_if_token(args.token or None)
    p = download_one_file(
        args.dataset, args.path, args.local_dir,
        revision=args.revision, token=(args.token or None),
    )
    print(f"[OK] {p}")


def cmd_fetch_volumes(args):
    login_if_token(args.token or None)
    paths = discover_remote_volumes(
        args.dataset, pattern=args.pattern,
        revision=args.revision, token=(args.token or None),
    )
    print(f"[INFO] {len(paths)} volumes to fetch")
    with StreamingVolumeFetcher(
        dataset_id=args.dataset, file_paths=paths,
        scratch_dir=args.scratch_dir, revision=args.revision,
        token=(args.token or None), keep_n=args.keep,
    ) as fetcher:
        for i, local in enumerate(fetcher.iter_paths(), 1):
            print(f"[{i:3d}/{len(paths)}] fetched {local} ({local.stat().st_size/1024/1024:.1f} MB)")
            if args.delete_after:
                fetcher.release()


def main():
    ap = argparse.ArgumentParser(description="ModelScope dataset helpers")
    ap.add_argument("--dataset", default="DEREKVERSE/SEED-VII", help="dataset id 'NAMESPACE/NAME'")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--token", default="", help="API token (overrides env MODELSCOPE_API_TOKEN)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    a = sp.add_parser("list", help="list files in the dataset")
    a.add_argument("--pattern", default="")
    a.set_defaults(func=cmd_list)

    a = sp.add_parser("fetch-info", help="download save_info CSVs")
    a.add_argument("--local-dir", required=True)
    a.add_argument("--include", default="",
                   help="optional comma-list of globs (default: */save_info/*_save_info.csv heuristic)")
    a.set_defaults(func=cmd_fetch_info)

    a = sp.add_parser("fetch-one", help="download a single file by remote path")
    a.add_argument("--path", required=True, help="remote file path inside the dataset")
    a.add_argument("--local-dir", required=True)
    a.set_defaults(func=cmd_fetch_one)

    a = sp.add_parser("fetch-volumes", help="stream-download split volumes (testing)")
    a.add_argument("--pattern", default="*.zip.*")
    a.add_argument("--scratch-dir", required=True)
    a.add_argument("--keep", type=int, default=1)
    a.add_argument("--delete-after", action="store_true",
                   help="delete each volume immediately after fetch (smoke test)")
    a.set_defaults(func=cmd_fetch_volumes)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
