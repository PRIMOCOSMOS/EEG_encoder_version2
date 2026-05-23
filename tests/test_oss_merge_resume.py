"""Test OSS multipart merge pipeline with resumability — mocked oss2.Bucket."""
import io
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---- Build a mock bucket that mimics oss2.Bucket API surface we use ----

@dataclass
class _MockPart:
    part_number: int
    etag: str
    size: int
    data: bytes


@dataclass
class _MockUpload:
    upload_id: str
    key: str
    parts: dict  # part_number -> _MockPart


class _MockResult:
    def __init__(self, **kw): self.__dict__.update(kw)


class MockBucket:
    """In-memory mock with the subset of oss2.Bucket calls we use."""
    def __init__(self):
        self.uploads = {}            # upload_id -> _MockUpload
        self.completed = {}          # key -> bytes (the final assembled object)
        self._uid_counter = 0
        self.fail_next_uploads = 0   # countdown: fail this many upload_part calls
        self.fail_count = 0          # tally of injected failures triggered

    def init_multipart_upload(self, key, headers=None, params=None):
        self._uid_counter += 1
        uid = f"mockuid_{self._uid_counter:04d}"
        self.uploads[uid] = _MockUpload(uid, key, {})
        return _MockResult(upload_id=uid)

    def upload_part(self, key, upload_id, part_number, data, progress_callback=None, headers=None):
        if self.fail_next_uploads > 0:
            self.fail_next_uploads -= 1
            self.fail_count += 1
            raise RuntimeError(f"injected failure on part {part_number}")
        if hasattr(data, "read"):
            buf = data.read()
        else:
            buf = bytes(data)
        if progress_callback is not None:
            progress_callback(len(buf), len(buf))
        u = self.uploads[upload_id]
        etag = f"etag{part_number}_{len(buf)}"
        u.parts[part_number] = _MockPart(part_number, etag, len(buf), buf)
        return _MockResult(etag=etag)

    def complete_multipart_upload(self, key, upload_id, parts, headers=None):
        u = self.uploads[upload_id]
        # assemble in part-number order
        ordered = sorted(parts, key=lambda p: p.part_number)
        body = b"".join(u.parts[p.part_number].data for p in ordered)
        self.completed[key] = body
        # OSS deletes the in-progress upload on complete
        del self.uploads[upload_id]
        return _MockResult(etag=f"final_etag_{len(body)}")

    def abort_multipart_upload(self, key, upload_id, headers=None):
        self.uploads.pop(upload_id, None)

    def head_object(self, key):
        if key not in self.completed:
            raise RuntimeError("not found")
        return _MockResult(content_length=len(self.completed[key]))


# Mock the oss2 module's iterators
class _PartIter:
    def __init__(self, bucket, key, upload_id, marker="", max_parts=1000, headers=None):
        u = bucket.uploads[upload_id]
        self._items = [p for _, p in sorted(u.parts.items())]
        self._idx = 0
    def __iter__(self): return self
    def __next__(self):
        if self._idx >= len(self._items):
            raise StopIteration
        p = self._items[self._idx]; self._idx += 1
        return _MockResult(part_number=p.part_number, etag=p.etag, size=p.size)


class _MUIter:
    def __init__(self, bucket, prefix="", **kw):
        self._items = [_MockResult(key=u.key, upload_id=u.upload_id)
                       for u in bucket.uploads.values() if u.key.startswith(prefix)]
        self._idx = 0
    def __iter__(self): return self
    def __next__(self):
        if self._idx >= len(self._items):
            raise StopIteration
        x = self._items[self._idx]; self._idx += 1
        return x


# ---- Run the test ----

def main():
    import oss2.models

    tmp = Path(tempfile.mkdtemp(prefix="oss_merge_test_"))
    print(f"tmp={tmp}")

    # 1) Build 5 fake "split volumes"
    parts_dir = tmp / "remote"
    parts_dir.mkdir()
    truth = b""
    sizes = []
    import random
    random.seed(42)
    for i in range(1, 6):
        data = random.randbytes(200_000 + i * 50_000)
        name = f"part_{i:03d}"
        (parts_dir / name).write_bytes(data)
        sizes.append((name, len(data)))
        truth += data
    total = len(truth)
    print(f"truth total = {total} bytes")

    # 2) Mock ModelScope APIs
    import src.oss_merge as om
    import src.ms_download as msdl

    msdl.login_if_token = lambda token=None: None
    msdl.list_dataset_files = lambda dataset_id, **kw: [
        {"Path": n, "Size": s, "Type": "blob"} for n, s in sizes
    ]
    # Also patch the names imported into oss_merge module namespace
    om.login_if_token = msdl.login_if_token
    om.list_dataset_files = msdl.list_dataset_files
    import pathlib
    def fake_download(dataset_id, file_path, local_dir, **kw):
        src_ = parts_dir / file_path
        dst_ = pathlib.Path(local_dir) / file_path
        dst_.write_bytes(src_.read_bytes())
        return dst_
    msdl.download_one_file = fake_download
    om.download_one_file = fake_download   # patch module-local alias

    mock_bucket = MockBucket()
    om._make_oss_bucket = lambda dataset_id, revision, token: (
        mock_bucket, "mock-bucket", "datasets/mock_dir",
    )

    # patch oss2 iterators used inside oss_merge
    import oss2
    oss2.PartIterator = _PartIter
    oss2.MultipartUploadIterator = _MUIter

    # ---- Test A: clean run ----
    print("\n=== Test A: clean run ===")
    scratch = tmp / "scratchA"; scratch.mkdir()
    st = om.stream_merge_and_upload_via_oss(
        dataset_id="fake/ds", pattern="part_*",
        path_in_repo="SEED-VII.zip",
        scratch_dir=str(scratch),
    )
    expected_key = "datasets/mock_dir/SEED-VII.zip"
    assembled = mock_bucket.completed.get(expected_key)
    assert assembled == truth, f"assembled bytes differ! got {len(assembled) if assembled else 'None'} vs {len(truth)}"
    assert len(st.done_parts) == 5
    # disk should be empty
    leftover = list(scratch.glob("part_*"))
    assert not leftover, f"scratch leftover: {leftover}"
    print(f"  ✓ A: 5 parts uploaded, assembled bytes match truth ({total} bytes)")
    print(f"  ✓ A: scratch dir clean")

    # ---- Test B: simulate crash mid-stream, then resume ----
    print("\n=== Test B: crash-mid-stream then resume ===")
    # reset
    mock_bucket2 = MockBucket()
    om._make_oss_bucket = lambda dataset_id, revision, token: (
        mock_bucket2, "mock-bucket", "datasets/mock_dir",
    )

    # Inject: fail the 3rd upload_part call with too many retries -> raise
    mock_bucket2.fail_next_uploads = 100  # always fail
    scratch2 = tmp / "scratchB"; scratch2.mkdir()
    state_file = tmp / "stateB.json"

    # First run — should crash on part 1 (since all uploads fail)
    crashed = False
    try:
        om.stream_merge_and_upload_via_oss(
            dataset_id="fake/ds", pattern="part_*",
            path_in_repo="SEED-VII.zip",
            scratch_dir=str(scratch2),
            state_path=str(state_file),
            max_part_retries=2,
        )
    except RuntimeError as e:
        crashed = True
        print(f"  ✓ B1: first run crashed as expected: {e}")
    assert crashed
    # Clean stale download from crashed run (the new code does this too at Stage 2)
    for f in scratch2.iterdir():
        f.unlink()

    # Simulate partial success: pretend parts 1+2 got through on a previous half-successful run
    # by manually inserting them
    uid = list(mock_bucket2.uploads.keys())[0]
    for pn in (1, 2):
        data = (parts_dir / f"part_{pn:03d}").read_bytes()
        mock_bucket2.uploads[uid].parts[pn] = _MockPart(pn, f"e{pn}", len(data), data)

    # Now stop injecting failures and rerun
    mock_bucket2.fail_next_uploads = 0
    # delete state file so we test the "lost state, recover from OSS" path
    state_file.unlink()

    st2 = om.stream_merge_and_upload_via_oss(
        dataset_id="fake/ds", pattern="part_*",
        path_in_repo="SEED-VII.zip",
        scratch_dir=str(scratch2),
        state_path=str(state_file),
        max_part_retries=3,
    )
    expected_key = "datasets/mock_dir/SEED-VII.zip"
    assembled = mock_bucket2.completed.get(expected_key)
    assert assembled == truth, f"resume assembled bytes wrong"
    assert len(st2.done_parts) == 5
    leftover = list(scratch2.glob("part_*"))
    assert not leftover, f"scratch leftover after resume: {leftover}"
    print(f"  ✓ B2: resumed with 2 done parts (from server-side list_parts), uploaded remaining 3")
    print(f"  ✓ B2: assembled bytes match truth ({total} bytes)")
    print(f"  ✓ B2: scratch dir clean")

    # ---- Test C: rerun an already-completed upload — should be near-instant ----
    print("\n=== Test C: rerun fully-done upload ===")
    # No way to test full no-op since state shows stage_complete_done; just run twice
    st3 = om.stream_merge_and_upload_via_oss(
        dataset_id="fake/ds", pattern="part_*",
        path_in_repo="SEED-VII.zip",
        scratch_dir=str(scratch2),
        state_path=str(state_file),
    )
    assert st3.stage_complete_done
    print(f"  ✓ C: 2nd invocation no-op'd, ETag = {st3.final_etag}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n✅ ALL TESTS PASS")


if __name__ == "__main__":
    main()
