import unittest
from unittest.mock import MagicMock

from translation_ranker import TranslationRanker


class MockSimilarityCalculator:
    """A lightweight mock that returns deterministic similarity values."""

    def calculate_similarity(self, text1, text2, metric="cosine", **kwargs):
        # Very simple deterministic similarity: based on shared character count ratio
        common = len(set(text1.lower().split()) & set(text2.lower().split()))
        total = len(set(text1.lower().split()) | set(text2.lower().split()))
        return common / total if total else 0.0


class TestTranslationRanker(unittest.TestCase):
    def setUp(self):
        # Inject mock similarity calculator to avoid heavy model loading
        self.mock_sim_calc = MockSimilarityCalculator()
        self.ranker = TranslationRanker(similarity_calculator=self.mock_sim_calc)

        self.source = "The quick brown fox jumps over the lazy dog."
        self.candidates = [
            "Le renard brun rapide saute par-dessus le chien paresseux.",
            "The quick brown fox jumps over a lazy dog.",  # very close
            "A completely unrelated sentence.",
            "The dog jumps over the fox."  # partial overlap
        ]

    def test_rank_order(self):
        ranked = self.ranker.rank_translations(self.source, self.candidates)
        # Ensure same number of results
        self.assertEqual(len(ranked), len(self.candidates))
        # Similarities should be in non-increasing order
        sims = [item["similarity"] for item in ranked]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_empty_input(self):
        ranked = self.ranker.rank_translations(self.source, [])
        self.assertEqual(ranked, [])

        ranked = self.ranker.rank_translations("", self.candidates)
        self.assertEqual(ranked, [])

    def test_confidence_scores(self):
        from translation_ranker import ConfidenceScorer
        scorer = ConfidenceScorer()
        sims = [0.9, 0.75, 0.4, 0.1]
        for method in ["distribution", "gap", "range"]:
            confs = scorer.calculate_confidence(sims, method=method)
            # Length match
            self.assertEqual(len(confs), len(sims))
            # All conf in 0-1
            self.assertTrue(all(0.0 <= c <= 1.0 for c in confs))


if __name__ == "__main__":
    unittest.main() 