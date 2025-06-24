import unittest
import os
import subprocess
import json
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_main_script() -> str:
    """Return absolute path to main.py, trying a few sensible locations."""
    candidates = [
        Path(__file__).parent / "main.py",
        Path(__file__).parent.parent / "main.py",
        Path.cwd() / "main.py",
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    raise FileNotFoundError(
        f"main.py not found; searched: {[str(c) for c in candidates]}"
    )


def run_main(args: List[str], expect_success: bool = True) -> Tuple[int, Dict[str, Any], str]:
    """
    Execute main.py with *args*.
    Returns (exit_code, parsed_stdout_json | {}, stderr_text).
    """
    cmd = [sys.executable, find_main_script(), *args]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    try:
        payload = json.loads(proc.stdout) if proc.stdout else {}
    except json.JSONDecodeError:
        payload = {}

    if expect_success:
        assert proc.returncode == 0, f"exit {proc.returncode} – stderr:\n{proc.stderr}"
    return proc.returncode, payload, proc.stderr


def touch_input_file(root: Path, *, name="in.json",
                     src="Hello world", trg="Bonjour le monde",
                     src_lang="en", trg_lang="fr") -> str:
    """Create a minimal JSON input file and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source_text": src,
                "translation": trg,
                "source_lang": src_lang,
                "target_lang": trg_lang,
            },
            fh,
        )
    return str(path)


def assert_threshold_file(path: str) -> Dict[str, Any]:
    """Assert the export exists & has required keys, return parsed JSON."""
    assert os.path.exists(path), f"threshold file missing: {path}"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for k in ("global_thresholds", "language_specific_thresholds", "weights"):
        assert k in data, f"key {k} missing in threshold export"
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMainScript(unittest.TestCase):
    """Integration tests for CLI behaviour."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.in_dir = self.tmp / "in"
        self.out_dir = self.tmp / "out"
        self.exp_dir = self.tmp / "tmp"
        self.exp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- shared -----------------------------------------------------------------

    def _export_path(self, name="exp.json") -> str:
        return str(self.exp_dir / name)

    # -- cases ------------------------------------------------------------------

    def test_single_file_tier_export(self):
        """Happy-path: enable tier + export thresholds."""
        inp = touch_input_file(self.in_dir)
        exp = self._export_path()

        _, payload, _ = run_main(
            ["--input", inp, "--enable-tier", "--export-thresholds", exp]
        )

        self.assertIn("analysis", payload)
        self.assertIn("quality_tier", payload["analysis"])

        assert_threshold_file(exp)

    def test_unsupported_language_pair(self):
        """
        Main should still succeed with an unknown language pair
        and fall back to global thresholds.
        """
        inp = touch_input_file(
            self.in_dir,
            name="unsupported.json",
            src="Foo bar",
            trg="Baz qux",
            src_lang="xx",
            trg_lang="yy",
        )
        exp = self._export_path("exp_unsupported.json")

        code, payload, stderr = run_main(
            ["--input", inp, "--enable-tier", "--export-thresholds", exp]
        )

        self.assertEqual(code, 0, stderr)
        self.assertIn("analysis", payload)
        self.assertIn("quality_tier", payload["analysis"])

        data = assert_threshold_file(exp)
        # Unsupported pair unlikely to appear; we just ensure file integrity
        self.assertTrue(isinstance(data["global_thresholds"], dict))


if __name__ == "__main__":
    unittest.main()
