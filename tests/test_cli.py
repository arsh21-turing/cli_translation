import unittest
import sys
import os
import tempfile
import json
import yaml
from io import StringIO
from unittest.mock import patch

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main  # noqa: E402
from main import rank_translations_cli  # noqa: E402


class TestCLI(unittest.TestCase):
    """End-to-end tests for the candidate-ranking CLI handler.

    Heavy neural models are mocked so the tests run quickly and offline.
    """

    def setUp(self):
        self.source_text = "The quick brown fox jumps over the lazy dog."
        self.candidates = [
            "Le renard brun rapide saute par-dessus le chien paresseux.",
            "Un rapide renard marron saute au-dessus du chien paresseux.",
            "The dog jumps over the quick brown fox.",
            "Le renard marron rapide saute par-dessus le chien fainéant.",
        ]

        # Patch heavy calculation with lightweight stub
        patcher = patch("main.calculate_translation_confidence", self._mock_confidence)
        self.addCleanup(patcher.stop)
        patcher.start()

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------
    def _mock_confidence(
        self, source_text, candidates, model_name=None, confidence_method=None, include_diagnostics=False
    ):
        # Deterministic pseudo-scores for quick testing
        ranked = [
            {
                "translation": cand,
                "similarity": round(1.0 - idx * 0.1, 4),
                "confidence": round(0.9 - idx * 0.1, 4),
            }
            for idx, cand in enumerate(candidates)
        ]
        result = {"ranked_translations": ranked}
        if include_diagnostics:
            result["diagnostics"] = {
                "optimal_clusters": 1,
                "cluster_sizes": {0: len(candidates)},
                "cluster_cohesion": 1.0,
                "cluster_separation": 0.0,
                "variance_explained": 0.0,
            }
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    class _Args:  # Simple container mimicking argparse.Namespace
        pass

    def _build_args(self, **overrides):
        args = self._Args()
        # Defaults
        args.source_text = self.source_text
        args.source_file = None
        args.candidates = ",".join(self.candidates)
        args.candidates_file = None
        args.model = "all-MiniLM-L6-v2"
        args.confidence_method = "distribution"
        args.output_format = "json"
        args.include_diagnostics = False
        args.output_file = None
        # Apply overrides
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_json_output(self):
        args = self._build_args(output_format="json")
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            exit_code = rank_translations_cli(args)
        self.assertEqual(exit_code, 0)
        payload = json.loads(captured.getvalue())
        self.assertEqual(payload["source_text"], self.source_text)
        self.assertEqual(len(payload["ranked_translations"]), len(self.candidates))
        for item in payload["ranked_translations"]:
            self.assertIn("similarity", item)
            self.assertIn("confidence", item)

    def test_yaml_output(self):
        args = self._build_args(output_format="yaml")
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            exit_code = rank_translations_cli(args)
        self.assertEqual(exit_code, 0)
        payload = yaml.safe_load(captured.getvalue())
        self.assertEqual(payload["source_text"], self.source_text)
        self.assertEqual(len(payload["ranked_translations"]), len(self.candidates))

    def test_include_diagnostics(self):
        args = self._build_args(output_format="json", include_diagnostics=True)
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            exit_code = rank_translations_cli(args)
        self.assertEqual(exit_code, 0)
        payload = json.loads(captured.getvalue())
        self.assertIn("diagnostics", payload)
        diag = payload["diagnostics"]
        for key in [
            "optimal_clusters",
            "cluster_sizes",
            "cluster_cohesion",
            "cluster_separation",
            "variance_explained",
        ]:
            self.assertIn(key, diag)

    def test_output_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            args = self._build_args(output_format="json", output_file=tmp_path)
            captured = StringIO()
            with patch("sys.stdout", new=captured):
                exit_code = rank_translations_cli(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(len(payload["ranked_translations"]), len(self.candidates))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main() 