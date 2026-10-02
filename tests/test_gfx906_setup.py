"""CPU-only entrypoint regressions; no GPU runtime, downloads or processes run."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


setup = module("strata_setup_for_test", ROOT / "setup.py")
model = module("gfx906_model_for_test", ROOT / "tools/gfx906_model.py")
mtp = module("gfx906_mtp_for_test", ROOT / "tools/mtp_fetch.py")
CARDS = [dict(index=i, arch="gfx906", name=f"MI50 {i}", vram_gb=31.984, vendor="amd") for i in (0, 1)]


class Selection(unittest.TestCase):
    def test_opt_in_default_off(self):
        self.assertIsNotNone(setup.amd_problem(CARDS[0]))
        self.assertIsNone(setup.amd_problem(CARDS[0], True))
        self.assertIsNotNone(setup.amd_problem(dict(arch="gfx908"), True))
        self.assertIsNone(setup.amd_problem(dict(arch="gfx1201")))

    def test_order_and_all(self):
        self.assertEqual(setup.select_amd_gpus("1,0", CARDS, True), [1, 0])
        self.assertEqual(setup.select_amd_gpus("all", CARDS, True), [0, 1])
        self.assertEqual(setup.select_amd_gpus(None, CARDS, True), [0])
        self.assertEqual(setup.select_amd_gpus(1, CARDS, True), [1])

    def test_rejections(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for value in ("all", [0, 1]):
                with self.assertRaises(SystemExit):
                    setup.select_amd_gpus(value, CARDS)
            for value in ("0,0", "0,3", "a,1", "0,", [], [1]):
                with self.assertRaises(SystemExit):
                    setup.select_amd_gpus(value, CARDS, True)

    def test_check_is_read_only_and_amd_multigpu(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(setup, "ROOT", Path(directory)))
            stack.enter_context(patch.object(sys, "argv", ["setup.py", "--check", "--backend", "hip",
                                                        "--experimental-gfx906", "--gpus", "0,1"]))
            stack.enter_context(patch.object(setup, "gpus", return_value=[]))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=CARDS))
            stack.enter_context(patch.object(setup, "ram_gb", return_value=121))
            stack.enter_context(patch.object(setup, "cpu_info", return_value=("Xeon", True, False)))
            stack.enter_context(patch.object(setup, "page_file_gb", return_value=None))
            migration = stack.enter_context(patch.object(setup, "data_folder"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
            migration.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_builder_default_rejects_gfx906(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, "ROOT", Path(directory)):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                setup.build_engine_hip(CARDS[0], Path("unused"))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_builder_compiles_all_architectures_and_saves_opt_in(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "rocm_root", return_value=(root / "sdk", [str(root / "sdk/lib")])))
            stack.enter_context(patch.object(setup, "source_hash", return_value="changed"))
            stack.enter_context(patch.object(setup, "source_version", return_value="0.1.30"))
            stack.enter_context(patch.object(setup.shutil, "which", return_value="/bin/tool"))
            stack.enter_context(patch.dict(os.environ, {}))
            def fake_build(*args):
                (root / "build-hip").mkdir(exist_ok=True)
                (root / "build-hip" / setup.EXE).write_text("synthetic build output")
            build = stack.enter_context(patch.object(setup, "cmake_build", side_effect=fake_build))
            gpu = {**CARDS[0], "archs": ["gfx906", "gfx1201"]}
            with contextlib.redirect_stdout(io.StringIO()):
                eng = setup.build_engine_hip(gpu, Path("ggml"), experimental_gfx906=True)
            options = build.call_args.args[3]
            self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1201;gfx906", options)
            self.assertIn("-DSTRATA_EXPERIMENTAL_GFX906=ON", options)
            meta = json.loads((eng / "BUILD.json").read_text())
            self.assertEqual(meta["archs"], ["gfx1201", "gfx906"])
            self.assertIs(meta["experimental_gfx906"], True)
            with contextlib.redirect_stdout(io.StringIO()):
                setup.build_engine_hip(gpu, Path("ggml"), experimental_gfx906=True)
            self.assertEqual(build.call_count, 1)

    def test_start_keeps_amd_list_and_uses_hip_builder(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            exe = root / setup.EXE
            exe.touch()
            (root / "BUILD.json").write_text(json.dumps({"lib_dirs": []}))
            cfg = root / "test.json"
            cfg.write_text(json.dumps({"exe": str(exe), "args": [], "backend": "hip", "gpu": 0,
                                       "experimental_gfx906": True}))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=CARDS))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            stack.enter_context(patch.object(setup, "build_engine_hip", return_value=root))
            process = stack.enter_context(patch.object(setup.subprocess, "call", return_value=0))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.start(cfg, 8084, "1,0", open_browser=False, layer_split="24"), 0)
            saved = json.loads(cfg.read_text())
            self.assertEqual(saved["gpu"], [1, 0])
            self.assertEqual(saved["layer_split"], "24")
            self.assertNotIn("--gpu", process.call_args.args[0])  # Saved list, not an override that truncates it.


class Frontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serve.server import child_env, engine_args, gpu_list
        cls.child_env = staticmethod(child_env)
        cls.engine_args = staticmethod(engine_args)
        cls.gpu_list = staticmethod(gpu_list)

    def test_hip_mask_clears_conflicting_masks(self):
        cfg = {"backend": "hip", "gpu": [1, 0], "args": [], "env": {"STRATA_TEST": "1"}}
        with patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}):
            env = self.child_env(cfg)
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "1,0")
            self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertEqual(os.environ["ROCR_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["STRATA_TEST"], "1")

    def test_layer_split_and_selection(self):
        cfg = {"backend": "hip", "gpu": "0,1", "args": ["--pack", "test"]}
        self.assertEqual(self.engine_args(cfg)[-2:], ["--layer-split", "auto"])
        cfg["args"] += ["--layer-split", "24"]
        self.assertEqual(self.engine_args(cfg).count("--layer-split"), 1)
        for value in ([0, 0], "-1,0"):
            with self.assertRaises(ValueError):
                self.gpu_list({"gpu": value})

    def test_cuda_mask_unchanged(self):
        env = self.child_env({"gpu": [0, 2], "args": []})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,2")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")


class Downloads(unittest.TestCase):
    DATA = b"synthetic bytes for SHA256 test"

    def response(self, status=200, start=0):
        class Response(io.BytesIO):
            pass
        response = Response(self.DATA[start:])
        response.status = status
        response.headers = {"Content-Length": str(len(self.DATA) - start),
                            "Content-Range": f"bytes {start}-{len(self.DATA)-1}/{len(self.DATA)}"}
        return response

    def test_hash_check_and_pinned_url(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entry = ("test.gguf", len(self.DATA), hashlib.sha256(self.DATA).hexdigest())
            with patch.object(model.urllib.request, "urlopen", return_value=self.response()) as fetch:
                with contextlib.redirect_stdout(io.StringIO()):
                    model.download(folder, entry, 1e20, 0, threading.Event())
            self.assertEqual((folder / entry[0]).read_bytes(), self.DATA)
            self.assertIn(model.REVISION, fetch.call_args.args[0].full_url)
            self.assertFalse((folder / (entry[0] + ".part")).exists())

    def test_resume_rejects_ignored_range(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entry = ("test.gguf", len(self.DATA), hashlib.sha256(self.DATA).hexdigest())
            partial = folder / "test.gguf.part"
            partial.write_bytes(self.DATA[:5])
            (folder / "test.gguf.download.json").write_text(json.dumps(
                {"repo": model.REPO, "revision": model.REVISION, "bytes": entry[1], "sha256": entry[2]}))
            with patch.object(model.urllib.request, "urlopen", return_value=self.response(200)):
                with self.assertRaisesRegex(RuntimeError, "ignored resume range"):
                    model.download(folder, entry, 1e20, 0, threading.Event())
            self.assertEqual(partial.read_bytes(), self.DATA[:5])
            self.assertFalse((folder / entry[0]).exists())

    def test_corrupt_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "test.gguf").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                model.download(folder, ("test.gguf", len(self.DATA), hashlib.sha256(self.DATA).hexdigest()),
                               1e20, 0, threading.Event())
            self.assertEqual((folder / "test.gguf").read_bytes(), b"corrupt")


class StorageAndRanges(unittest.TestCase):
    def test_complete_model_recheck_needs_only_safety_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "test.gguf").write_bytes(b"ready")
            with patch.object(model, "FILES", (("test.gguf", 5, "unused"),)), \
                 patch.object(model.shutil, "disk_usage", return_value=type("Disk", (), {"free": 5*model.GIB})()), \
                 patch.object(sys, "argv", ["model", "--directory", directory]), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                model.main()
            plan = json.loads(out.getvalue())
            self.assertTrue(plan["fits"])
            self.assertEqual(plan["reserve_bytes"], 4*model.GIB)

    def test_ignored_tensor_range_rejected_before_reading_body(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {}
        with patch.object(mtp.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(IOError, "did not honor"):
                mtp.get(mtp.REPO+"shard", 0, 7, retries=1)
        response.read.assert_not_called()

    def test_valid_tensor_range(self):
        response = io.BytesIO(b"data")
        response.status = 206
        response.headers = {"Content-Range": "bytes 8-11/100"}
        with patch.object(mtp.urllib.request, "urlopen", return_value=response):
            self.assertEqual(mtp.get(mtp.REPO+"shard", 8, 11, retries=1), b"data")

    def test_oversized_safetensors_header(self):
        with patch.object(mtp, "get", return_value=struct.pack("<Q", (64 << 20)+1)) as get:
            with self.assertRaisesRegex(ValueError, "oversized"):
                mtp.shard_header("shard")
            self.assertEqual(get.call_count, 1)

    def test_inventory_revision_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"mtp-inventory.json").write_text(json.dumps({"repo": "old/main", "tensors": []}))
            with patch.object(mtp, "get") as get:
                with self.assertRaisesRegex(ValueError, "another checkpoint"):
                    mtp.fetch(directory, None)
                get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
