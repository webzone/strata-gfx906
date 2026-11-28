"""A local shape engine (a pinned build for a card the update feed does not carry, e.g. AMD MI50 / gfx906) runs when the feed has none for this PC.

ensure() gets the feed's answer "none-for-this-pc". With an engine of the shape installed, it returns that engine and does not show the feed's note.
With none installed, it shows the note and returns None, as before.

CPU-only: the feed, the update lock, the hardware and the installed-version lookup are mocked. No network, no GPU, no downloads.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata_control import engine_component as ec, shape_engine, signing, updater  # noqa: E402

SHAPE = "q4_lite"
NOTE = "The update feed has no engine for Q4 Lite Shape for this computer."
HW = {"platform": "linux-x86_64", "backend": "cuda", "runtime": None, "nvidia_driver": "610.47", "cuda_archs": ["sm_86"], "gfx": []}
FEED_URL = "https://feed.invalid/strata-feed.json"


class NoneForThisPc(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="strata-none-for-pc-")
        self.addCleanup(tmp.cleanup)
        self.comp = Path(tmp.name)
        self.lines = []
        patches = [
            mock.patch.object(updater, "detect_hardware", return_value=HW),
            mock.patch.object(updater, "open_feed", return_value=object()),
            mock.patch.object(updater, "update_lock", side_effect=lambda root: contextlib.nullcontext()),
            mock.patch.object(updater, "State"),
            mock.patch.object(signing, "load_trusted_keys", return_value={}),
            mock.patch.object(shape_engine, "update_one", return_value={"status": "none-for-this-pc", "note": NOTE}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def ensure(self):
        return shape_engine.ensure(SHAPE, FEED_URL, components=self.comp, emit=self.lines.append)

    def test_a_installed_engine_runs_and_the_feed_note_is_not_shown(self):
        exe = self.comp / "shapes" / SHAPE / "engine" / "0.1.70-linux-x86_64-cuda" / "strata-q4-lite"
        with mock.patch.object(shape_engine, "newest_version", return_value="0.1.70"), \
                mock.patch.object(shape_engine, "find", return_value=SimpleNamespace(exe=exe)) as find:
            got = self.ensure()
        self.assertEqual(got, exe)
        self.assertNotIn(NOTE, self.lines)
        find.assert_called_once_with(SHAPE, self.comp, full=False, title=None)

    def test_b_no_installed_engine_shows_the_feed_note_and_returns_none(self):
        with mock.patch.object(shape_engine, "newest_version", return_value=None), \
                mock.patch.object(shape_engine, "find") as find:
            got = self.ensure()
        self.assertIsNone(got)
        self.assertIn(NOTE, self.lines)
        find.assert_not_called()

    def test_c_installed_engine_that_find_refuses_shows_the_feed_note_and_returns_none(self):
        with mock.patch.object(shape_engine, "newest_version", return_value="0.1.70"), \
                mock.patch.object(shape_engine, "find", side_effect=ec.EngineMissing("no engine for this PC")):
            got = self.ensure()
        self.assertIsNone(got)
        self.assertIn(NOTE, self.lines)


if __name__ == "__main__":
    unittest.main()
