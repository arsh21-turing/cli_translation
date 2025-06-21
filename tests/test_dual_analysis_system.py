import unittest
from unittest.mock import MagicMock

from dual_analysis_system import DualAnalysisSystem


class DualAnalysisSystemTest(unittest.TestCase):
    """Minimal tests ensuring DualAnalysisSystem ranks and correlates correctly."""

    def setUp(self) -> None:
        # Prepare a fake analyzer whose analyse_pair returns deterministic metrics
        self.mock_analyzer = MagicMock()
        # create predictable results
        def _fake_analyze_pair(source_text, translation, **kwargs):
            # synthetic embedding similarity based on length difference
            sim = max(0.0, 1.0 - abs(len(source_text) - len(translation)) / 100)
            # assign groq score inversely proportional to idx embedded in translation string
            idx = 1 if "#1" in translation else (2 if "#2" in translation else 3)
            return {
                "embedding_similarity": sim,
                "groq_score": 10 - idx,  # 9, 8, 7
                "composite_score": sim * 0.5 + (10 - idx) / 10 * 0.5,
            }
        self.mock_analyzer.analyze_pair.side_effect = _fake_analyze_pair

        self.system = DualAnalysisSystem(analyzer=self.mock_analyzer)

        self.source = "Hello world"
        self.translations = ["Hola mundo #1", "Hola mundo #2", "Hola mundo #3"]

    # ------------------------------------------------------------------
    def test_analyze_multiple_ranking(self) -> None:
        result = self.system.analyze_multiple(self.source, self.translations)
        # Ensure three candidates processed
        self.assertEqual(len(result["candidates"]), 3)
        # Combined scores should be sorted descending
        scores = [c["combined"] for c in result["candidates"]]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
        # Verify best and weakest indexes recorded correctly
        self.assertIn(result["best_index"], {0, 1, 2})
        self.assertIn(result["weakest_index"], {0, 1, 2})
        self.assertNotEqual(result["best_index"], result["weakest_index"])

    def test_correlation_present(self) -> None:
        res = self.system.analyze_multiple(self.source, self.translations)
        # Pearson may be None if variance zero; here should have value
        self.assertIsNotNone(res["correlation"])


if __name__ == "__main__":
    unittest.main() 