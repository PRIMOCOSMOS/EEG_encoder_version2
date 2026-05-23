#!/usr/bin/env python3
"""Inspect / (optionally) physically concatenate split volumes of the SEED-VII zip.

支持两种来源（互斥）：
  A) --volumes-dir  本地分卷目录
  B) --ms-dataset   ModelScope 数据集 ID（实例不能挂载时用这个）
                    远端按 --pattern 列出分卷，仅列表/校验/可选「一卷一下载一删」抽样。

设计原则：
- **默认不落盘** —— 无论本地还是远端，都只生成 manifest，训练管线通过
  `ConcatStream` / `LazyConcatStream` 直接「虚拟拼接」按需读取，避免再多占 160GB 空间。
- **可选** --concat-to PATH：磁盘充裕（> 200GB）时物理拼成单个 zip（远端模式会
  一卷一下载，下完拼到末尾后立刻删该分卷）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zip_stream import discover_volumes  # noqa: E402

CHUNK = 64 * 1024 * 1024  # 64 MB


def main():
    ap = argparse.ArgumentParser(description="Inspect / concat SEED-VII split volumes (local or ModelScope)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--volumes-dir", help="Directory containing local split volumes")
    src.add_argument("--ms-dataset", help="ModelScope dataset id (e.g. DEREKVERSE/SEED-VII)")
    ap.add_argument("--pattern", default="*.zip.*", help="Glob to find volumes (default: *.zip.*)")
    ap.add_argument("--manifest", default="", help="Path to write manifest JSON")
    ap.add_argument("--sha256", action="store_true", help="(local only) compute per-volume sha256")
    ap.add_argument("--concat-to", default="", help="If set, physically concat all volumes into this single file")
    ap.add_argument("--scratch-dir", default="./_ms_scratch", help="(ms-dataset) where to cache downloads")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--token", default="", help="(ms-dataset) MODELSCOPE token (overrides env)")
    args = ap.parse_args()

    if args.volumes_dir:
        _main_local(args)
    else:
        _main_modelscope(args)


def _main_local(args):
    vol_dir = Path(args.volumes_dir)
    parts = discover_volumes(vol_dir, pattern=args.pattern)
    total = 0
    entries = []
    for p in parts:
        sz = p.stat().st_size
        entry = {"name": p.name, "size": sz}
        if args.sha256:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                while True:
                    buf = fh.read(CHUNK)
                    if not buf:
                        break
                    h.update(buf)
            entry["sha256"] = h.hexdigest()
        entries.append(entry)
        total += sz
        print(f"[VOL] {p.name}: {sz/1024/1024:.1f} MB")

    manifest_path = Path(args.manifest) if args.manifest else (vol_dir / "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"source": "local", "volumes_dir": str(vol_dir),
                   "total_bytes": total, "volumes": entries, "pattern": args.pattern}, fh, indent=2)
    print(f"[OK] manifest -> {manifest_path}   total={total/1024/1024/1024:.2f} GB")

    if args.concat_to:
        out = Path(args.concat_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[CONCAT] writing -> {out}")
        with open(out, "wb") as wfh:
            for p in parts:
                with open(p, "rb") as rfh:
                    while True:
                        buf = rfh.read(CHUNK)
                        if not buf:
                            break
                        wfh.write(buf)
                print(f"[..] wrote {p.name}")
        print(f"[OK] concat complete: {out} ({out.stat().st_size/1024/1024/1024:.2f} GB)")
    else:
        print("[INFO] No --concat-to -> virtual concat only. Downstream uses ConcatStream.")


def _main_modelscope(args):
    from src.ms_download import (
        download_one_file, list_dataset_files, login_if_token,
    )
    import fnmatch as _fn

    token = args.token or None
    login_if_token(token)

    listing = list_dataset_files(args.ms_dataset, revision=args.revision, token=token)
    listing = [f for f in listing if _fn.fnmatch(f.get("Path", ""), args.pattern)]
    if not listing:
        raise SystemExit(f"No remote volumes matched {args.pattern} in {args.ms_dataset}")

    def _digit_key(name: str):
        digits = ""
        for ch in reversed(name):
            if ch.isdigit(): digits = ch + digits
            else:
                if digits: break
        return (int(digits) if digits else 0, name)
    listing.sort(key=lambda f: _digit_key(f.get("Path", "")))

    entries = []
    total = 0
    for f in listing:
        name = f.get("Path")
        sz = int(f.get("Size", 0) or 0)
        entries.append({"name": name, "size": sz})
        total += sz
        print(f"[REMOTE] {name}: {sz/1024/1024:.1f} MB")

    manifest_path = Path(args.manifest) if args.manifest else Path("./_ms_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"source": "modelscope", "dataset": args.ms_dataset,
                   "revision": args.revision, "pattern": args.pattern,
                   "total_bytes": total, "volumes": entries}, fh, indent=2)
    print(f"[OK] manifest -> {manifest_path}   total={total/1024/1024/1024:.2f} GB")

    if args.concat_to:
        out = Path(args.concat_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(args.scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
        print(f"[CONCAT] download-once-and-delete → {out}")
        with open(out, "wb") as wfh:
            for f in listing:
                name = f["Path"]
                local = download_one_file(args.ms_dataset, name, str(scratch),
                                          revision=args.revision, token=token)
                with open(local, "rb") as rfh:
                    while True:
                        buf = rfh.read(CHUNK)
                        if not buf: break
                        wfh.write(buf)
                try:
                    Path(local).unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"[..] appended {name}")
        print(f"[OK] concat complete: {out} ({out.stat().st_size/1024/1024/1024:.2f} GB)")
    else:
        print("[INFO] No --concat-to -> virtual concat only. "
              "Downstream uses LazyConcatStream (≤2 volumes resident on disk).")


if __name__ == "__main__":
    main()
