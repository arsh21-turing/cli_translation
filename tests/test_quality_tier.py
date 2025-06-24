import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from quality_learning_engine import QualityLearningEngine
from config_manager import ConfigManager


class TestQualityTierLogic(unittest.TestCase):
    """Unit tests for the quality-tier voting and reporting logic."""

    def setUp(self):
        # Temporary workspace
        self.tmp = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp) / "data"
        self.out_dir = Path(self.tmp) / "out"
        self.data_dir.mkdir(exist_ok=True)
        self.out_dir.mkdir(exist_ok=True)

        # Lightweight config manager stub
        cfg = MagicMock(spec=ConfigManager)
        cfg.get.return_value = {}

        self.engine = QualityLearningEngine(
            data_dir=str(self.data_dir),
            output_dir=str(self.out_dir),
            config=cfg,
        )

        # Consistent thresholds for tests
        base = {
            "excellent": 0.85,
            "good": 0.75,
            "acceptable": 0.65,
            "poor": 0.55,
            "very_poor": 0.0,
        }
        thr = {k: base.copy() for k in ["similarity", "groq_rating", "combined_score"]}
        self.engine.thresholds = thr
        self.engine.default_thresholds = thr

        # Explicit weights
        self.engine.weights = {"similarity_weight": 0.6, "groq_weight": 0.4}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    # --------------------------------------------------------------
    # Tier determination tests
    # --------------------------------------------------------------
    def test_clear_majority(self):
        tier = self.engine.determine_quality_tier(
            "en",
            "fr",
            similarity_score=0.9,
            groq_rating=0.8,
            combined_score=0.88,
        )
        self.assertEqual(tier, "excellent")

    def test_weighted_tie_break(self):
        # Creates a three-way tie: each metric maps to a different tier
        tier = self.engine.determine_quality_tier(
            "en",
            "fr",
            similarity_score=0.78,  # good
            groq_rating=0.9,        # excellent
            combined_score=0.7,     # acceptable
        )
        # Expect 'acceptable' because combined_score holds highest tie-break weight
        self.assertEqual(tier, "acceptable")

    # --------------------------------------------------------------
    # Report generation test
    # --------------------------------------------------------------
    def test_report_structure(self):
        report = self.engine.get_quality_report(
            "en",
            "fr",
            similarity_score=0.86,
            groq_rating=0.74,
            combined_score=0.8,
        )
        # Basic keys
        for key in [
            "language_pair",
            "quality_tier",
            "metrics",
            "weights",
            "confidence",
            "timestamp",
            "reasoning",
        ]:
            self.assertIn(key, report)
        # Ensure metrics include expected sub-fields
        for m in ["similarity", "groq_rating", "combined_score"]:
            self.assertIn(m, report["metrics"])
            self.assertIn("tier", report["metrics"][m])
            self.assertIn("score", report["metrics"][m])

        # Confidence level should match vote agreement
        conf = report["confidence"]
        self.assertIn(conf["level"], ["high", "medium", "low"])
        self.assertTrue(0.0 <= conf["agreement_ratio"] <= 1.0) 