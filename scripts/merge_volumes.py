#!/usr/bin/env python3
"""Verify / (optionally) physically concatenate split volumes of the SEED-VII zip.

设计原则：
- **默认不落盘** —— 只创建一个 manifest（顺序+大小+SHA256（可选）），训练管线通过
  `src.zip_stream.ConcatStream` 直接以「虚拟拼接」方式读取，避免再多占 160GB 空间。
- **可选** --concat-to PATH：如果磁盘充裕（> 200GB），可以物理拼成单个 zip。
- 实现使用「读 X MB 写 X MB」的流式 IO，避免内存爆炸。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Allow running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zip_stream import discover_volumes  # noqa: E402

CHUNK = 64 * 1024 * 1024  # 64 MB


def main():
    ap = argparse.ArgumentParser(description="Inspect / concat SEED-VII split volumes")
    ap.add_argument("--volumes-dir", required=True, help="Directory containing the split volumes")
    ap.add_argument("--pattern", default="*.zip.*", help="Glob to find volumes (default: *.zip.*)")
    ap.add_argument("--manifest", default="", help="Path to write manifest JSON (default: <dir>/_manifest.json)")
    ap.add_argument("--sha256", action="store_true", help="Compute per-volume sha256 (slow)")
    ap.add_argument("--concat-to", default="", help="If set, physically concat all volumes into this single file")
    args = ap.parse_args()

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
        json.dump({"total_bytes": total, "volumes": entries, "pattern": args.pattern}, fh, indent=2)
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
        print("[INFO] No --concat-to specified -> virtual concat only (zero extra disk). "
              "Downstream scripts will read the volumes through src.zip_stream.ConcatStream.")


if __name__ == "__main__":
    main()
