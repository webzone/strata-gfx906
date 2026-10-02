"""v0.1.34 merge regressions: mocks only, no GPU, model download or deployment."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup
from serve.server import child_env
from tools import mtp_fetch, strata_mcp

CARD = dict(index=0, arch="gfx906", name="MI50", vram_gb=31.984, vendor="amd")


class V034Integration(unittest.TestCase):
    def test_windows_does_not_admit_gfx906_via_rdna_prebuilt(self):
        with patch.object(setup, "WIN", True):
            self.assertIn("requires Linux", setup.amd_problem(CARD, True))

    def test_cpu_vision_build_retains_gfx906_stamp_and_gate(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            engine = root / "engine"
            engine.mkdir()
            (engine / setup.EXE).touch()
            meta = dict(backend="hip", src="same", archs=["gfx906"], experimental_gfx906=True)
            (engine / "BUILD.json").write_text(json.dumps(meta))
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "source_hash", return_value="same"))
            sdk = stack.enter_context(patch.object(setup, "rocm_root"))
            encoder = stack.enter_context(patch.object(setup, "build_vision_cpu", return_value=engine))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.build_engine_hip(CARD, Path("unused"), "cpu", experimental_gfx906=True), engine)
            encoder.assert_called_once()
            self.assertIs(encoder.call_args.args[2]["experimental_gfx906"], True)
            sdk.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                setup.build_engine_hip(CARD, Path("unused"), "cpu")
            self.assertEqual(encoder.call_count, 1)

    def test_update_preserves_cpu_vision_and_opt_in(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            engine = root / "engine"
            engine.mkdir()
            (engine / setup.EXE).touch()
            (engine / "BUILD.json").write_text(json.dumps(dict(
                backend="hip", src="old", archs=["gfx906"], experimental_gfx906=True, vision="cpu")))
            stack.enter_context(patch.object(setup, "WIN", False))
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "source_hash", return_value="new"))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=[CARD]))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            builder = stack.enter_context(patch.object(setup, "build_engine_hip"))
            with contextlib.redirect_stdout(io.StringIO()):
                setup.update_installed_engine(setup.PREBUILT_URL)
            self.assertEqual(builder.call_args.args[2], "cpu")
            self.assertIs(builder.call_args.kwargs["experimental_gfx906"], True)

    def test_saved_start_preserves_cpu_vision_and_opt_in(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            exe = root / setup.EXE
            exe.touch()
            (root / "BUILD.json").write_text(json.dumps(dict(lib_dirs=[])))
            cfg = root / "test.json"
            cfg.write_text(json.dumps(dict(exe=str(exe), args=[], backend="hip", gpu=0,
                                           experimental_gfx906=True, vision=dict(exe="cpu-encoder"))))
            stack.enter_context(patch.object(setup, "WIN", False))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=[CARD]))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            stack.enter_context(patch.object(setup.subprocess, "call", return_value=0))
            builder = stack.enter_context(patch.object(setup, "build_engine_hip", return_value=root))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.start(cfg, 8095, open_browser=False), 0)
            self.assertEqual(builder.call_args.args[2], "cpu")
            self.assertIs(builder.call_args.kwargs["experimental_gfx906"], True)

    def test_saved_windows_rdna_start_keeps_prebuilt_ordinal_and_dll_dirs(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            exe = root / setup.EXE
            exe.touch()
            dlls = root / "rocm/bin"
            dlls.mkdir(parents=True)
            (root / "BUILD.json").write_text(json.dumps(dict(lib_dirs=["rocm/bin"])))
            cfg = root / "test.json"
            cfg.write_text(json.dumps(dict(exe=str(exe), args=[], backend="hip", gpu=0)))
            card = dict(index=0, arch="gfx1201", name="RDNA4", vram_gb=16)
            stack.enter_context(patch.object(setup, "WIN", True))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=[card]))
            stack.enter_context(patch.object(setup, "get_prebuilt_hip", return_value=root))
            stack.enter_context(patch.object(setup, "hip_card", return_value={**card, "index": 1}))
            stack.enter_context(patch.object(setup.subprocess, "call", return_value=0))
            builder = stack.enter_context(patch.object(setup, "build_engine_hip"))
            llama = stack.enter_context(patch.object(setup, "get_llama_cpp"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.start(cfg, 8095, open_browser=False), 0)
            saved = json.loads(cfg.read_text())
            self.assertEqual(saved["hip_ordinal"], 1)
            self.assertEqual(saved["lib_dirs"], [str(dlls)])
            builder.assert_not_called()
            llama.assert_not_called()

    def test_single_hip_ordinal_owns_mask_after_config_env(self):
        cfg = dict(backend="hip", gpu=0, hip_ordinal=1, env=dict(HIP_VISIBLE_DEVICES="9"))
        with patch.dict(os.environ, dict(ROCR_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0")):
            env = child_env(cfg)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "1")
        self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_multi_hip_list_keeps_order_despite_stale_single_ordinal(self):
        env = child_env(dict(backend="hip", gpu=[1, 0], hip_ordinal=7))
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "1,0")

    def test_mtp_source_override_cannot_change_acceptance_identity(self):
        spec = importlib.util.spec_from_file_location("mtp_identity_test", ROOT / "tools/mtp_fetch.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, dict(STRATA_MTP_REVISION="main")):
            spec.loader.exec_module(module)
        self.assertEqual(module.REVISION, mtp_fetch.MTP_REVISION)
        self.assertEqual(module.REPO, mtp_fetch.PINNED)

    def test_mtp_verify_is_offline(self):
        with tempfile.TemporaryDirectory() as d, patch.object(sys, "argv", ["mtp", "verify", "--out", d]), \
                patch.object(mtp_fetch.urllib.request, "urlopen") as request, self.assertRaises(SystemExit) as exit:
            mtp_fetch.main()
        self.assertEqual(exit.exception.code, 0)
        request.assert_not_called()

    def test_posix_process_identity_rejects_zombie_and_missing_process(self):
        with patch.object(strata_mcp, "WIN", False), patch.object(strata_mcp.Path, "exists", return_value=False), \
                patch.object(strata_mcp.subprocess, "run") as run:
            for result in (Mock(returncode=0, stdout="Z Thu Oct  1 12:00:00 2026"),
                           Mock(returncode=1, stdout="")):
                run.return_value = result
                self.assertIsNone(strata_mcp.proc_identity(123))
            run.return_value = Mock(returncode=0, stdout="S Thu Oct  1 12:00:00 2026")
            self.assertEqual(strata_mcp.proc_identity(123), "posix:Thu Oct  1 12:00:00 2026")

    def test_reused_pid_is_not_terminated(self):
        with patch.object(strata_mcp, "proc_identity", return_value="posix:new-start"), \
                patch.object(strata_mcp.os, "kill") as kill:
            self.assertFalse(strata_mcp.proc_terminate(123, "posix:old-start"))
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
