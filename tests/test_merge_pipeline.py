"""End-to-end mock test of the stream-merge-upload pipeline.

We simulate ModelScope by:
  - serving "remote" volumes from a local directory
  - serving the merged zip via a local HTTP server (real HTTP Range requests!)
    to validate the RemoteRangeStream path

This is a STRONG validation: real HTTP, real zipfile, real SHA-256.
"""
import http.server
import io
import os
import shutil
import socketserver
import tempfile
import threading
import zipfile
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.io


def main():
    # ---- 0. Build a synthetic SEED-VII-shaped zip and byte-split it ----
    tmp = Path(tempfile.mkdtemp(prefix="merge_pipeline_test_"))
    print(f"tmp={tmp}")

    zip_path = tmp / "seed_vii_full.zip"
    truth_mats = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for sid in range(1, 4):  # 3 subjects
            bio = io.BytesIO()
            scipy.io.savemat(bio, {str(fid): np.random.randn(62, 1000).astype(np.float64)
                                    for fid in range(1, 5)})  # fields 1..4
            bio.seek(0); payload = bio.read()
            name = f"EEG_preprocessed/{sid}.mat"
            zf.writestr(name, payload)
            truth_mats[name] = hashlib.sha256(payload).hexdigest()
    zsize = zip_path.stat().st_size
    truth_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"original zip: {zsize/1024/1024:.2f} MB, sha256={truth_sha[:16]}...")

    # Byte-split into 5 volumes (mimic user's splitter)
    parts_dir = tmp / "remote_volumes"; parts_dir.mkdir()
    N = 5; part_size = (zsize + N - 1) // N
    sizes = []
    with open(zip_path, "rb") as src:
        for i in range(1, N + 1):
            chunk = src.read(part_size)
            if not chunk: break
            name = f"seed_vii.zip.{i:03d}"
            (parts_dir / name).write_bytes(chunk)
            sizes.append((name, len(chunk)))
    assert sum(s for _, s in sizes) == zsize

    # ---- 1. Validate Stage 1: streaming SHA-256 over remote splits ----
    print("\n=== Test 1: streaming SHA-256 over byte-spliced volumes ===")
    from src.zip_stream import LazyConcatStream

    scratch = tmp / "scratch1"; scratch.mkdir()
    def fetch(name):
        src_, dst_ = parts_dir / name, scratch / name
        if not dst_.exists():
            shutil.copyfile(src_, dst_)
        return str(dst_)
    def evict(p):
        try: p.unlink()
        except FileNotFoundError: pass

    stream = LazyConcatStream(sizes, fetch, evict, max_resident=1,
                              pin_last=False, warmup_last=False)
    h = hashlib.sha256()
    while True:
        buf = stream.read(1024 * 1024)
        if not buf: break
        h.update(buf)
    stream.close()
    got_sha = h.hexdigest()
    assert got_sha == truth_sha, f"hash mismatch: {got_sha} vs {truth_sha}"
    assert sorted(os.listdir(scratch)) == [], f"scratch not clean: {os.listdir(scratch)}"
    print(f"  ✓ stream SHA-256 == truth ({got_sha[:16]}...)")
    print(f"  ✓ scratch cleaned after close()")

    # ---- 2. Two-pass usage (hash + upload) — measure no extra disk usage ----
    print("\n=== Test 2: two-pass usage (hash + upload) ===")
    scratch2 = tmp / "scratch2"; scratch2.mkdir()
    download_log = []
    def fetch2(name):
        src_, dst_ = parts_dir / name, scratch2 / name
        if not dst_.exists():
            shutil.copyfile(src_, dst_)
            download_log.append(name)
        return str(dst_)

    # pass 1
    s1 = LazyConcatStream(sizes, fetch2, evict, max_resident=1, pin_last=False, warmup_last=False)
    h = hashlib.sha256()
    while True:
        b = s1.read(1024*1024)
        if not b: break
        h.update(b)
    s1.close()
    n_pass1 = len(download_log)

    # pass 2 — simulate upload (just consume)
    s2 = LazyConcatStream(sizes, fetch2, evict, max_resident=1, pin_last=False, warmup_last=False)
    total = 0
    max_resident_observed = 0
    while True:
        b = s2.read(1024*1024)
        if not b: break
        total += len(b)
        # observe disk
        live = [f for f in os.listdir(scratch2)]
        if len(live) > max_resident_observed:
            max_resident_observed = len(live)
    s2.close()
    assert total == zsize
    assert n_pass1 == N
    assert len(download_log) == 2 * N
    assert max_resident_observed <= 2, \
        f"too many volumes resident: {max_resident_observed}"
    print(f"  ✓ pass1 downloads={n_pass1}, pass2 downloads={len(download_log) - n_pass1}")
    print(f"  ✓ max resident volumes during pass2 = {max_resident_observed} (≤ 2)")
    print(f"  ✓ total bytes through stream == truth zip size ({total} bytes)")

    # ---- 3. RemoteRangeStream on a real HTTP server ----
    print("\n=== Test 3: HTTP Range streaming over merged zip ===")
    class _RangeHandler(http.server.BaseHTTPRequestHandler):
        """Tiny HTTP server that honors `Range: bytes=A-B` for one file in cwd."""
        def log_message(self, *a, **kw): pass
        def _file(self):
            # serve whatever file is requested, relative to cwd
            return os.path.join(os.getcwd(), self.path.lstrip("/"))
        def do_HEAD(self):
            path = self._file()
            if not os.path.isfile(path):
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        def do_GET(self):
            path = self._file()
            if not os.path.isfile(path):
                self.send_error(404); return
            total = os.path.getsize(path)
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                spec = rng[6:]
                try:
                    a_s, b_s = spec.split("-", 1)
                    a = int(a_s) if a_s else 0
                    b = int(b_s) if b_s else total - 1
                except ValueError:
                    self.send_error(416); return
                if a > b or a >= total:
                    self.send_error(416); return
                b = min(b, total - 1)
                length = b - a + 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{total}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(a)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(64 * 1024, remaining))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(total))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk: break
                        self.wfile.write(chunk)
    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

    http_root = tmp / "http_root"; http_root.mkdir()
    shutil.copyfile(zip_path, http_root / "merged.zip")
    os.chdir(http_root)
    server = _Server(("127.0.0.1", 0), _RangeHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    print(f"  HTTP server at 127.0.0.1:{port}")

    try:
        from src.remote_range import RemoteRangeStream
        url = f"http://127.0.0.1:{port}/merged.zip"
        rs = RemoteRangeStream(
            url_provider=lambda: url, total_size=zsize,
            chunk_size=512 * 1024, cache_bytes=8 * 1024 * 1024,
        )
        zf = zipfile.ZipFile(rs, mode="r")
        names = sorted(i.filename for i in zf.infolist() if not i.is_dir())
        assert sorted(names) == sorted(truth_mats.keys()), names
        print(f"  ✓ zipfile.ZipFile(RemoteRangeStream) lists {len(names)} mats")
        for info in zf.infolist():
            if info.is_dir(): continue
            data = zf.read(info.filename)
            assert hashlib.sha256(data).hexdigest() == truth_mats[info.filename], info.filename
        print(f"  ✓ all {len(names)} mats extracted with matching SHA-256")
        print(f"  ✓ HTTP stats: {rs.n_http_requests} GET reqs, {rs.n_cache_hits} cache hits, "
              f"{rs.bytes_fetched/1024:.1f} KB fetched (vs total {zsize/1024:.1f} KB)")
        zf.close(); rs.close()
    finally:
        server.shutdown(); server.server_close()
        os.chdir(tmp)

    # ---- 4. iter_trials_from_modelscope_single_file via local mock ----
    print("\n=== Test 4: iter_trials_from_modelscope_single_file (full pipeline) ===")
    import src.dataset as _ds
    import src.remote_range as _rr

    os.chdir(http_root)
    server2 = _Server(("127.0.0.1", 0), _RangeHandler)
    port2 = server2.server_address[1]
    t2 = threading.Thread(target=server2.serve_forever, daemon=True); t2.start()

    def _fake_open(dataset_id, path_in_repo, **kw):
        url = f"http://127.0.0.1:{port2}/merged.zip"
        return _rr.RemoteRangeStream(
            url_provider=lambda: url, total_size=zsize,
            chunk_size=512*1024, cache_bytes=8*1024*1024,
        )
    _rr.open_dataset_file_as_range_stream = _fake_open

    try:
        trials = list(_ds.iter_trials_from_modelscope_single_file(
            dataset_id="fake/fake", path_in_repo="merged.zip",
        ))
        print(f"  ✓ yielded {len(trials)} trials (expected 3 subjects × 4 fields = 12)")
        assert len(trials) == 12
        subjects = sorted(set(t.subject for t in trials))
        print(f"  subjects = {subjects}")
        print(f"  EEG shape sample = {trials[0].eeg.shape}")
        assert all(t.eeg.shape == (62, 1000) for t in trials)
    finally:
        server2.shutdown(); server2.server_close()
        os.chdir(tmp.parent)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n✅ ALL TESTS PASS")


if __name__ == "__main__":
    main()
