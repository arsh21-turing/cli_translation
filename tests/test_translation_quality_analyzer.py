import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation_quality_analyzer import (
    TranslationQualityAnalyzer,
    analyze_translation,
    rank_translations_with_quality,
)


class MockEmbeddingGenerator:
    """Fast fake embedding generator that returns deterministic numeric vectors."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def generate_embedding(self, text: str, language=None):  # noqa: D401
        # Return a small deterministic numeric vector based on char count
        val = float(len(text))
        return np.array([val, val / 2.0, val / 3.0], dtype=float)


class FastAnalyzer(TranslationQualityAnalyzer):
    """Subclass that avoids heavy embedding calculation."""

    def __init__(self):
        super().__init__(embedding_generator=MockEmbeddingGenerator())

    # Override similarity to deterministic value
    def analyze_pair(self, source_text, translation):  # type: ignore[override]
        res = super().analyze_pair(source_text, translation)
        # Force similarity to simple heuristic for repeatability
        common = len(set(source_text.split()) & set(translation.split()))
        total = len(set(source_text.split()) | set(translation.split())) or 1
        res["semantic_similarity"] = common / total
        res["quality_score"] = self.calculate_composite_quality_score(res)
        return res


class TestTranslationQualityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = FastAnalyzer()
        self.source = "The quick brown fox jumps over the lazy dog."
        self.good_translation = "Le renard brun rapide saute par-dessus le chien paresseux."
        self.bad_translation = "Completely unrelated sentence."
        self.candidates = [
            self.good_translation,
            "Un rapide renard marron saute au-dessus du chien paresseux.",
            self.bad_translation,
        ]

    def test_analyze_pair_scores(self):
        good = self.analyzer.analyze_pair(self.source, self.good_translation)
        bad = self.analyzer.analyze_pair(self.source, self.bad_translation)
        self.assertGreater(good["quality_score"], bad["quality_score"])

    def test_rank_candidates(self):
        res = self.analyzer.rank_candidates(self.source, self.candidates)
        self.assertEqual(len(res["ranked_translations"]), len(self.candidates))
        # Ensure sorted by quality
        qs = [item["quality_score"] for item in res["ranked_translations"]]
        self.assertEqual(qs, sorted(qs, reverse=True))

    # ------------------------------------------------------------------
    # Language mismatch penalty
    # ------------------------------------------------------------------

    def test_language_mismatch_penalty(self):
        same_lang = "The agile brown fox leaps above the lazy canine."
        diff_lang = self.good_translation  # French

        res_same = self.analyzer.analyze_pair(self.source, same_lang)
        res_diff = self.analyzer.analyze_pair(self.source, diff_lang)

        self.assertEqual(res_same["language_mismatch_penalty"], 0.0)
        self.assertEqual(res_diff["language_mismatch_penalty"], 1.0)
        self.assertGreater(res_diff["quality_score"], res_same["quality_score"])


if __name__ == "__main__":
    unittest.main() 