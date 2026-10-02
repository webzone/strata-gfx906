"""Tests for tools/mtp_fetch.py (#327): a range read needs a 206 with the asked Content-Range, the pinned revision's
tensors are checked against their sha256. This fork fails closed and retains corrupt files and conflicting
inventories rather than deleting or replacing them; `verify` reports corruption offline.  A fake checkpoint behind a mocked
urlopen - nothing is downloaded.

    python -m unittest tools.test_mtp_fetch
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import mtp_fetch as M  # noqa: E402


def shard(tensors):
    """A safetensors file: [u64 header length][JSON header][data]."""
    header, data = {}, b""
    for name, blob in tensors:
        header[name] = {"dtype": "BF16", "shape": [len(blob) // 2], "data_offsets": [len(data), len(data) + len(blob)]}
        data += blob
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data


A, B, X = bytes(range(64)), bytes(range(100, 164)) * 2, b"\x07" * 48
FILES = {"model-1.safetensors": shard([("model.x", X), ("mtp.a", A)]), "model-2.safetensors": shard([("mtp.b", B)])}
INDEX = json.dumps({"weight_map": {"model.x": "model-1.safetensors", "mtp.a": "model-1.safetensors",
                                   "mtp.b": "model-2.safetensors"}}).encode()
HASHES = {"mtp.a": hashlib.sha256(A).hexdigest(), "mtp.b": hashlib.sha256(B).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status, self.headers = status, headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Mirror:
    """urlopen for the fake checkpoint.  `ignore_range`: 200 and the whole file (`cut`: cut to the asked length, as
    the proxy in #327 did); `wrong`: the bytes of these tensors' ranges are flipped (a source that sends bad data)."""

    def __init__(self, ignore_range=False, cut=False, wrong=()):
        self.ignore_range, self.cut, self.wrong, self.ranges = ignore_range, cut, set(wrong), []

    def __call__(self, req, timeout=None):
        name = req.full_url.rsplit("/", 1)[1]
        body = INDEX if name == "model.safetensors.index.json" else FILES[name]
        rng = req.get_header("Range")
        if rng is None:
            return Response(body, 200, {"Content-Length": str(len(body))})
        a, b = map(int, rng.split("=")[1].split("-"))
        self.ranges.append((name, a, b))
        if self.ignore_range:
            return Response(body[:b - a + 1] if self.cut else body, 200, {})
        part = body[a:b + 1]
        if any(t in self.wrong and self.span(name, t) == (a, b) for t in ("mtp.a", "mtp.b")):
            part = bytes(x ^ 0xFF for x in part)
        return Response(part, 206, {"Content-Range": "bytes %d-%d/%d" % (a, b, len(body))})

    @staticmethod
    def span(name, tensor):
        raw = FILES[name]
        n = struct.unpack("<Q", raw[:8])[0]
        meta = json.loads(raw[8:8 + n]).get(tensor)
        return (8 + n + meta["data_offsets"][0], 8 + n + meta["data_offsets"][1] - 1) if meta else None


class FetchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.patches = [mock.patch.object(M, "SHA256", HASHES), mock.patch.object(M.time, "sleep", lambda s: None),
                        mock.patch.object(M, "REPO", M.PINNED)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def fetch(self, mirror):
        with mock.patch.object(M.urllib.request, "urlopen", mirror), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            M.fetch(str(self.out), None)
        return err.getvalue()

    def tensor(self, name):
        return (self.out / "tensors" / (name + ".bin")).read_bytes()


class Ranges(FetchCase):
    def test_check_range(self):
        M.check_range(206, "bytes 10-19/100", 10, 19)
        for status, cr in ((200, None), (200, "bytes 10-19/100"), (206, "bytes 0-9/100"), (206, None)):
            with self.subTest(status=status, cr=cr), self.assertRaises(IOError):
                M.check_range(status, cr, 10, 19)

    def test_a_mirror_that_ignores_range_is_refused(self):
        for cut in (False, True):                 # the whole file, or cut to the asked length (#327's proxy)
            with self.subTest(cut=cut), mock.patch.object(M.urllib.request, "urlopen", Mirror(True, cut)), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(IOError, "not honoured"):
                M.get(M.REPO + "model-1.safetensors", 8, 15)

    def test_fetch_from_an_honest_mirror(self):
        self.fetch(Mirror())
        self.assertEqual((self.tensor("mtp.a"), self.tensor("mtp.b")), (A, B))
        manifest = json.loads((self.out / "mtp-manifest.json").read_text())
        self.assertEqual({r["name"]: r["sha256"] for r in manifest}, HASHES)
        self.assertEqual(M.verify(str(self.out)), [])

    def test_corrupt_existing_tensor_is_refused_and_retained(self):
        self.fetch(Mirror())
        bad = bytes(len(A))                       # right size, wrong bytes
        (self.out / "tensors" / "mtp.a.bin").write_bytes(bad)
        self.assertEqual(M.verify(str(self.out)), ["mtp.a"])
        mirror = Mirror()
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            self.fetch(mirror)
        self.assertEqual(self.tensor("mtp.a"), bad)
        self.assertEqual(mirror.ranges, [])
        self.assertEqual(M.verify(str(self.out)), ["mtp.a"])

    def test_wrong_downloaded_bytes_are_refused_and_retained(self):
        with self.assertRaisesRegex(RuntimeError, "is not the checkpoint's"):
            self.fetch(Mirror(wrong={"mtp.a"}))
        self.assertEqual(self.tensor("mtp.a"), bytes(x ^ 0xFF for x in A))
        self.assertFalse((self.out / "mtp-manifest.json").exists())

    def test_an_inventory_from_another_revision_is_refused_and_retained(self):
        self.fetch(Mirror())
        inv = json.loads((self.out / "mtp-inventory.json").read_text())
        inv["repo"] = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
        for r in inv["tensors"]:
            r["start"] += 1
            r["end"] += 1
        saved = json.dumps(inv)
        (self.out / "mtp-inventory.json").write_text(saved)
        (self.out / "tensors" / "mtp.b.bin").write_bytes(B[:10])
        mirror = Mirror()
        with self.assertRaisesRegex(ValueError, "another checkpoint revision"):
            self.fetch(mirror)
        self.assertEqual((self.out / "mtp-inventory.json").read_text(), saved)
        self.assertEqual((self.tensor("mtp.a"), self.tensor("mtp.b")), (A, B[:10]))
        self.assertEqual(mirror.ranges, [])

    def test_partial_tensor_resumes_only_the_pinned_range(self):
        self.fetch(Mirror())
        (self.out / "tensors" / "mtp.b.bin").write_bytes(B[:10])
        mirror = Mirror()
        self.fetch(mirror)
        start, end = Mirror.span("model-2.safetensors", "mtp.b")
        self.assertEqual(mirror.ranges, [("model-2.safetensors", start + 10, end)])
        self.assertEqual(self.tensor("mtp.b"), B)

    def test_oversized_existing_tensor_is_refused_and_retained(self):
        self.fetch(Mirror())
        bad = A + b"extra"
        (self.out / "tensors" / "mtp.a.bin").write_bytes(bad)
        with self.assertRaisesRegex(RuntimeError, "oversized MTP tensor"):
            self.fetch(Mirror())
        self.assertEqual(self.tensor("mtp.a"), bad)

    def test_verify_keeps_its_verdict_and_skips_other_revisions(self):
        self.fetch(Mirror())
        self.assertEqual(M.verify(str(self.out)), [])
        stamps = json.loads((self.out / "tensors" / "verified.json").read_text())
        self.assertEqual(sorted(stamps), ["mtp.a", "mtp.b"])
        with mock.patch.object(M, "sha256_of", side_effect=AssertionError("hashed an unchanged file")):
            self.assertEqual(M.verify(str(self.out)), [])
        with mock.patch.object(M, "REPO", "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"):
            (self.out / "tensors" / "mtp.a.bin").write_bytes(b"x")
            self.assertEqual(M.verify(str(self.out)), [])          # STRATA_MTP_REVISION: other hashes
        self.assertEqual(M.verify(str(self.out)), ["mtp.a"])
        self.assertEqual(M.verify(str(self.out / "nothing")), [])


class Pinned(unittest.TestCase):
    def test_every_tensor_has_a_hash(self):
        self.assertEqual(len(M.SHA256), 31)
        for name, h in M.SHA256.items():
            self.assertTrue(name.startswith("mtp."))
            self.assertRegex(h, "^[0-9a-f]{64}$")
        self.assertIn(M.PINNED_REVISION, M.PINNED)


class Setup(unittest.TestCase):
    def test_setup_rebuilds_a_corrupt_install(self):
        import setup
        with tempfile.TemporaryDirectory() as d:
            mtp = Path(d)
            self.assertFalse(setup.mtp_corrupt(mtp))              # no tensors kept: nothing to check
            (mtp / "tensors").mkdir()
            for code, want in ((0, False), (3, True), (1, False)):
                with self.subTest(code=code), mock.patch.object(
                        setup.subprocess, "run", return_value=mock.Mock(returncode=code)) as run:
                    self.assertEqual(setup.mtp_corrupt(mtp), want)
                    self.assertIn("verify", run.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
