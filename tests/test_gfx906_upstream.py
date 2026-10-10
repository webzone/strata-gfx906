"""CPU-only upstream-sync regressions. No real HIP build, GPU launch or model download.

Run: .venv/bin/python -m unittest discover -s tests -p 'test_gfx906*.py'
"""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup
from tools import gfx906_model, mtp_fetch

CARDS = [dict(index=i, arch="gfx906", name=f"MI50 {i}", vram_gb=31.984, vendor="amd") for i in (0, 1)]
RDNA = dict(index=2, arch="gfx1201", name="RDNA4", vram_gb=48.0, vendor="amd")


class UpstreamIntegration(unittest.TestCase):
    def test_version_and_acceptance_pins(self):
        self.assertEqual(setup.source_version(), "0.1.42")
        self.assertEqual(setup.MIN_ENGINE, (0, 1, 42))
        self.assertEqual(setup.LLAMA_CPP_COMMIT, "3cf03257f219afbe7334045ff7c6a06ac68c627d")
        self.assertEqual(setup.HF_REVISIONS[gfx906_model.REPO], gfx906_model.REVISION)
        self.assertEqual(mtp_fetch.REVISION, "de4b8e4d43b917e7706784d8bb445c9af86a3540")
        self.assertNotIn("gfx906", setup.AMD_ARCHS)
        self.assertNotIn("gfx906", setup.ROCM_INDEXES)

    def test_upstream_parser_preserves_gfx906_opt_in_and_order(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("1,0", CARDS, True)], [1, 0])
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus([1, 0], CARDS, True)], [1, 0])
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("all", CARDS + [RDNA], True)], [2, 0, 1])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            setup.amd_parse_gpus("1,0", CARDS)
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("0,2", CARDS + [RDNA], True)], [0, 2])

    def test_missing_system_sdk_never_uses_rdna_wheels(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"ROCM_PATH": directory, "STRATA_ROCM_INDEX": "https://rdna.invalid/"}), \
                patch.object(setup, "run") as run, patch.object(setup, "rocm_index") as index:
            for archs in ("gfx906", ["gfx906"], ["gfx1201", "gfx906"]):
                with self.subTest(archs=archs), contextlib.redirect_stdout(io.StringIO()) as out, \
                        self.assertRaises(SystemExit):
                    setup.rocm_root(archs)
                self.assertIn("system ROCm 7", out.getvalue())
            run.assert_not_called()
            index.assert_not_called()

    def test_update_preserves_gfx906_and_other_compiled_architectures(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            (engine / setup.EXE).touch()
            (engine / "BUILD.json").write_text(json.dumps({
                "backend": "hip", "src": "old", "experimental_gfx906": True,
                "archs": ["gfx906", "gfx1201"],
            }))
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "source_hash", return_value="new"))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=CARDS))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            nv = stack.enter_context(patch.object(setup, "gpus"))
            build = stack.enter_context(patch.object(setup, "build_engine_hip"))
            with contextlib.redirect_stdout(io.StringIO()):
                setup.update_installed_engine(setup.PREBUILT_URL)
            build.assert_called_once()
            self.assertEqual(build.call_args.args[0]["archs"], ["gfx906", "gfx1201"])
            self.assertIs(build.call_args.kwargs["experimental_gfx906"], True)
            nv.assert_not_called()

    def test_saved_amd_start_all_uses_saved_opt_in_not_nvidia_detection(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            exe = root / setup.EXE
            exe.touch()
            (root / "BUILD.json").write_text(json.dumps({"lib_dirs": []}))
            config = root / "test.json"
            config.write_text(json.dumps({"exe": str(exe), "args": [], "backend": "hip", "gpu": 0,
                                          "experimental_gfx906": True}))
            stack.enter_context(patch.object(setup, "amd_gpus", return_value=CARDS))
            stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=Path("unused")))
            build = stack.enter_context(patch.object(setup, "build_engine_hip", return_value=root))
            nv = stack.enter_context(patch.object(setup, "gpus"))
            process = stack.enter_context(patch.object(setup.subprocess, "call", return_value=0))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.start(config, 8095, "all", open_browser=False), 0)
            self.assertEqual(json.loads(config.read_text())["gpu"], [0, 1])
            self.assertIs(build.call_args.kwargs["experimental_gfx906"], True)
            self.assertNotIn("--gpu", process.call_args.args[0])
            nv.assert_not_called()

    def test_builder_does_not_reuse_a_stamp_missing_the_experimental_flag(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            (engine / setup.EXE).write_text("old")
            (engine / "BUILD.json").write_text(json.dumps({"backend": "hip", "src": "same", "archs": ["gfx906"]}))
            stack.enter_context(patch.object(setup, "ROOT", root))
            stack.enter_context(patch.object(setup, "source_hash", return_value="same"))
            stack.enter_context(patch.object(setup, "source_version", return_value="0.1.31"))
            stack.enter_context(patch.object(setup, "rocm_root", return_value=(root / "sdk", [])))
            stack.enter_context(patch.object(setup.shutil, "which", return_value="/bin/tool"))
            stack.enter_context(patch.dict(os.environ, {}))
            def fake_build(*args):
                (root / "build-hip").mkdir()
                (root / "build-hip" / setup.EXE).write_text("new")
            build = stack.enter_context(patch.object(setup, "cmake_build", side_effect=fake_build))
            with contextlib.redirect_stdout(io.StringIO()):
                setup.build_engine_hip(CARDS[0], Path("unused"), experimental_gfx906=True)
            build.assert_called_once()
            self.assertIn("-DSTRATA_EXPERIMENTAL_GFX906=ON", build.call_args.args[3])
            self.assertIs(json.loads((engine / "BUILD.json").read_text())["experimental_gfx906"], True)


@unittest.skipUnless(shutil.which("cmake"), "CMake unavailable; architecture-gate check not run")
class CmakeArchitectureGate(unittest.TestCase):
    """Execute only the architecture gate, stopping BEFORE enable_language(HIP)/SDK discovery."""
    def check_archs(self, archs, experimental=False):
        prefix = (ROOT / "cmake/hip_backend.cmake").read_text().split("enable_language(HIP)", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "gate.cmake"
            script.write_text('cmake_minimum_required(VERSION 3.24)\n'
                              f'set(CMAKE_HIP_ARCHITECTURES "{archs}")\n'
                              + ('set(STRATA_EXPERIMENTAL_GFX906 ON)\n' if experimental else '')
                              + prefix + '\nmessage(STATUS "ARCHS=${STRATA_HIP_ARCHS}")\n')
            return subprocess.run([shutil.which("cmake"), "-P", str(script)],
                                  capture_output=True, text=True, timeout=15)

    def test_gfx906_default_off(self):
        result = self.check_archs("gfx906")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires -DSTRATA_EXPERIMENTAL_GFX906=ON", " ".join(result.stderr.split()))

    def test_gfx906_real_arch_and_feature_suffix(self):
        result = self.check_archs("gfx906:sramecc+:xnack-;gfx1201", True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ARCHS=gfx906,gfx1201", result.stdout)

    def test_rdna_community_architectures_remain_available(self):
        result = self.check_archs("gfx1100;gfx1101;gfx1200;gfx1201")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ARCHS=gfx1100,gfx1101,gfx1200,gfx1201", result.stdout)

    def test_space_separated_rdna2_and_gfx906_architectures(self):
        result = self.check_archs("gfx1030 gfx906:sramecc+:xnack- gfx1201", True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ARCHS=gfx1030,gfx906,gfx1201", result.stdout)

    def test_opt_in_does_not_admit_other_wave64_architectures(self):
        result = self.check_archs("gfx908", True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
