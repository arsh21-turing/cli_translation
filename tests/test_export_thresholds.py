import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Import target helper directly from main
import importlib
main = importlib.import_module("main")

from quality_learning_engine import QualityLearningEngine  # for type hints only


class TestExportThresholds(unittest.TestCase):
    """Light-weight tests for main.export_thresholds helper."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.out_path = Path(self.tmp_dir) / "thresholds.json"

        # Build a fake engine with the required attributes
        self.engine = MagicMock(spec=QualityLearningEngine)
        self.engine.thresholds = {"similarity": {"excellent": 0.85}}
        self.engine.language_specific_thresholds = {"en-fr": {"similarity": {"excellent": 0.9}}}
        self.engine.get_learned_scoring_weights.return_value = {"similarity_weight": 0.6}

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_success(self):
        ok = main.export_thresholds(self.engine, str(self.out_path))
        self.assertTrue(ok)
        self.assertTrue(self.out_path.exists())
        data = json.loads(self.out_path.read_text())
        self.assertIn("global_thresholds", data)
        self.assertIn("weights", data)

    def test_no_engine(self):
        bad = main.export_thresholds(None, str(self.out_path))
        self.assertFalse(bad)

    def test_bad_path(self):
        # Simulate failure by patching open()
        from pathlib import Path as _P
        original_open = _P.open

        def raise_io(self, *a, **kw):  # type: ignore[override]
            raise IOError("fail")

        _P.open = raise_io  # type: ignore
        try:
            bad = main.export_thresholds(self.engine, str(self.out_path))
            self.assertFalse(bad)
        finally:
            _P.open = original_open


if __name__ == "__main__":
    unittest.main() 