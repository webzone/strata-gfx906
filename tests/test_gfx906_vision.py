"""Experimental HIP vision opt-in, build-mode transitions and saved starts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import setup

CARD = dict(index=0, arch="gfx906", name="MI50", vram_gb=32)


class HipVision(unittest.TestCase):
    def test_choice_requires_linux_opt_in_and_only_gfx906(self):
        with patch.object(setup, "WIN", False), patch.object(setup.sys, "platform", "linux"), contextlib.redirect_stdout(io.StringIO()):
            for choice in ("gpu", "yes"):
                self.assertEqual(setup.hip_vision(choice, experimental_gfx906=True, archs=["gfx906"]), "gpu")
                self.assertEqual(setup.hip_vision(choice, archs=["gfx906"]), "none")
                self.assertEqual(setup.hip_vision(choice, experimental_gfx906=True,
                                                archs=["gfx906", "gfx1201"]), "none")
        with patch.object(setup, "WIN", True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.hip_vision("gpu", experimental_gfx906=True, archs=["gfx906"]), "none")
        with patch.object(setup, "WIN", False), patch.object(setup.sys, "platform", "darwin"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.hip_vision("gpu", experimental_gfx906=True, archs=["gfx906"]), "none")

    def test_mode_changes_rebuild_encoder_but_do_not_rebuild_text_engine(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).touch()
            (eng / setup.VEXE).touch()
            stamp = eng / "BUILD.json"
            meta = dict(backend="hip", src="S", archs=["gfx906"], experimental_gfx906=True,
                        vision="cpu", vision_src="V")
            stamp.write_text(json.dumps(meta))
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "WIN", False))
            stack.enter_context(patch.object(setup.sys, "platform", "linux"))
            stack.enter_context(patch.object(setup, "source_hash", side_effect=lambda p: "V" if p == setup.VISION_SOURCES else "S"))
            stack.enter_context(patch.object(setup, "cpu_info", return_value=("Test", True, True)))
            stack.enter_context(patch.object(setup, "rocm_root", return_value=(Path("/opt/rocm"), ["/opt/rocm/lib"])))
            built = []

            def build(src, bdir, target, defs, *_):
                built.append((bdir.name, target, defs))
                (bdir / "bin").mkdir(parents=True, exist_ok=True)
                (bdir / "bin" / setup.VEXE).touch()

            stack.enter_context(patch.object(setup, "cmake_build", side_effect=build))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaises(SystemExit):
                setup.build_engine_hip(CARD, "unused", "gpu")
            with self.assertRaises(SystemExit):
                setup.build_engine_hip({**CARD, "archs": ["gfx906", "gfx1201"]}, "unused", "gpu",
                                       experimental_gfx906=True)
            self.assertEqual(built, [])
            setup.build_engine_hip(CARD, "unused", "gpu", experimental_gfx906=True)
            self.assertEqual(json.loads(stamp.read_text())["vision"], "gpu")
            self.assertEqual(built[0][:2], ("build-vision-hip", "strata-vision"))
            for flag in ("-DSTRATA_VISION_HIP=ON", "-DCMAKE_HIP_ARCHITECTURES=gfx906",
                         "-DSTRATA_EXPERIMENTAL_GFX906=ON"):
                self.assertIn(flag, built[0][2])
            setup.build_engine_hip(CARD, "unused", "gpu", experimental_gfx906=True)
            self.assertEqual(len(built), 1)
            setup.build_engine_hip(CARD, "unused", "cpu", experimental_gfx906=True)
            self.assertEqual(built[1][:2], ("build-vision", "strata-vision"))
            self.assertIn("-DSTRATA_VISION_HIP=OFF", built[1][2])
            self.assertEqual(json.loads(stamp.read_text())["vision"], "cpu")

    def test_saved_start_preserves_gpu_encoding(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            exe = root / setup.EXE
            exe.touch()
            (root / "BUILD.json").write_text(json.dumps(dict(lib_dirs=[])))
            cfg = root / "run.json"
            cfg.write_text(json.dumps(dict(exe=str(exe), args=[], backend="hip", gpu=0,
                                           experimental_gfx906=True, vision=dict(exe="hip-encoder", gpu=True))))
            stack.enter_context(patch.object(setup, "WIN", False))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=[CARD]))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            stack.enter_context(patch.object(setup.subprocess, "call", return_value=0))
            builder = stack.enter_context(patch.object(setup, "build_engine_hip", return_value=root))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(setup.start(cfg, 8095, open_browser=False), 0)
            self.assertEqual(builder.call_args.args[2], "gpu")
            self.assertIs(builder.call_args.kwargs["experimental_gfx906"], True)

    def test_update_rebuilds_changed_encoder_with_unchanged_text_engine(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).touch()
            (eng / "BUILD.json").write_text(json.dumps(dict(backend="hip", src="S", archs=["gfx906"],
                experimental_gfx906=True, vision="gpu", vision_src="old")))
            stack.enter_context(patch.object(setup, "WIN", False))
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "source_hash", side_effect=lambda p: "V" if p == setup.VISION_SOURCES else "S"))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=[CARD]))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            builder = stack.enter_context(patch.object(setup, "build_engine_hip"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            setup.update_installed_engine(setup.PREBUILT_URL)
            self.assertEqual(builder.call_args.args[2], "gpu")
            self.assertIs(builder.call_args.kwargs["experimental_gfx906"], True)


if __name__ == "__main__":
    unittest.main()
