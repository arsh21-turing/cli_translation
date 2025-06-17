"""translation_quality_analyzer.py
A self-contained module that provides a high-level TranslationQualityAnalyzer able
to compute composite quality scores for single source-translation pairs and rank
multiple candidate translations.  It re-uses existing infrastructure (vector
generation, similarity calculation, ranking, confidence scoring) while exposing
an easy interface and configurable component weights.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any, Union
from collections import defaultdict
import logging
import numpy as np

# Local imports – rely on existing pipeline
from config_manager import ConfigManager
from model_loader import ModelLoader, MultilingualModelManager
from text_processor import TextProcessor
from embedding_generator import MultilingualVectorGenerator
from similarity_calculator import SimilarityCalculator
from translation_ranker import (
    TranslationRanker,
    ConfidenceScorer,
    calculate_translation_confidence,
)
from language_utils import LanguageDetector
from embedding_cache import embedding_cache

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper: lightweight embedding generator wrapper
# -----------------------------------------------------------------------------

class EmbeddingGenerator:  # pragma: no cover (wrapper)
    """Light wrapper around MultilingualVectorGenerator with optional in-memory caching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *, use_cache: bool = True) -> None:
        self.model_name = model_name
        self.use_cache = use_cache
        self._init_pipeline(model_name)

    # ------------------------------------------------------------------
    def _init_pipeline(self, model_name: str) -> None:
        config = ConfigManager()
        # Allow overriding the default embedding model via config
        config.set("models.embedding.default", model_name)
        model_loader = ModelLoader(config)
        multilingual_manager = MultilingualModelManager(config, model_loader)
        text_processor = TextProcessor()
        self.vector_generator = MultilingualVectorGenerator(  # type: ignore[arg-type]
            multilingual_manager, text_processor, config
        )

    # ------------------------------------------------------------------
    def generate_embedding(self, text: str, language: Optional[str] = None) -> np.ndarray:
        """Return a single embedding vector for *text* (1-D np.ndarray).

        Caching uses *(text, model_name)* as key.  The returned object is the
        *original* vector, **not** a copy – callers should treat it as
        read-only.
        """
        if self.use_cache:
            cached = embedding_cache.get(text, self.model_name)
            if cached is not None:
                return cached

        vectors = self.vector_generator.generate_vectors(texts=[text], language=language)
        vec = vectors[0]

        if self.use_cache:
            embedding_cache.set(text, self.model_name, vec)
        return vec

    # Convenience pass-throughs ------------------------------------------------
    @staticmethod
    def cache_stats() -> Dict[str, float | int]:
        return embedding_cache.stats()

    @staticmethod
    def clear_cache() -> None:
        embedding_cache.clear()

# -----------------------------------------------------------------------------
# Utility – cosine similarity (independent of SimilarityCalculator for speed)
# -----------------------------------------------------------------------------

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-9
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

# -----------------------------------------------------------------------------
# Main analyzer
# -----------------------------------------------------------------------------

class TranslationQualityAnalyzer:
    """Comprehensive analysis of translation quality with composite quality scores."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "semantic_similarity": 0.45,
        "confidence": 0.15,
        "length_ratio": 0.1,
        "structure_preservation": 0.1,
        "language_detection": 0.1,
        "language_mismatch_penalty": 0.1,  # reward when languages differ
    }

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        ranker: Optional[TranslationRanker] = None,
        weights: Optional[Dict[str, float]] = None,
        *,
        use_cache: bool = True,
    ) -> None:
        self.embedding_generator = embedding_generator or EmbeddingGenerator(use_cache=use_cache)
        # Build a default similarity-calculator-backed ranker if not supplied
        self.ranker = ranker or TranslationRanker()
        self.confidence_scorer = ConfidenceScorer()
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._lang_detector = LanguageDetector()

    # ------------------------------------------------------------------
    # Single pair analysis
    # ------------------------------------------------------------------
    def analyze_pair(self, source_text: str, translation: str) -> Dict[str, Any]:
        """Return detailed quality metrics for *translation* of *source_text*."""
        if not source_text or not translation:
            return {
                "quality_score": 0.0,
                "semantic_similarity": 0.0,
                "confidence": 0.0,
            }

        # Embeddings & similarity
        src_vec = self.embedding_generator.generate_embedding(source_text)
        tgt_vec = self.embedding_generator.generate_embedding(translation)
        similarity = _cosine_similarity(src_vec, tgt_vec)

        # Confidence for this single value
        confidence = self.confidence_scorer.calculate_confidence([similarity], method="distribution")[0]

        # Length / structure metrics
        src_len = len(source_text.split()) or 1
        tgt_len = len(translation.split()) or 1
        length_ratio = min(src_len / tgt_len, tgt_len / src_len)

        src_sent = len([s for s in source_text.split(".") if s.strip()]) or 1
        tgt_sent = len([s for s in translation.split(".") if s.strip()]) or 1
        structure_pres = min(src_sent / tgt_sent, tgt_sent / src_sent)

        # Language detection (simple)
        src_lang = self._lang_detector.detect(source_text).get("code", "und")
        tgt_lang = self._lang_detector.detect(translation).get("code", "und")
        lang_detect_score = 1.0 if src_lang != tgt_lang else 0.0
        mismatch_metric = 1.0 if src_lang != tgt_lang else 0.0

        metrics: Dict[str, Any] = {
            "source_text": source_text,
            "translation_text": translation,
            "semantic_similarity": similarity,
            "confidence": confidence,
            "length_ratio": length_ratio,
            "structure_preservation": structure_pres,
            "language_detection": lang_detect_score,
            "language_mismatch_penalty": mismatch_metric,
            "source_language": src_lang,
            "translation_language": tgt_lang,
        }
        metrics["quality_score"] = self.calculate_composite_quality_score(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------
    def rank_candidates(
        self,
        source_text: str,
        candidates: List[str],
        confidence_method: str = "distribution",
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        if not candidates:
            return {"ranked_translations": [], "diagnostics": {} if include_diagnostics else None}

        # Base ranking w/ similarity + confidence (+ diagnostics)
        ranking_res = calculate_translation_confidence(
            source_text,
            candidates,
            model_name=self.embedding_generator.model_name,
            confidence_method=confidence_method,
            include_diagnostics=include_diagnostics,
        )

        for item in ranking_res["ranked_translations"]:
            translation = item.get("translation") or item.get("text")
            # Compute extra metrics
            metrics = {
                "semantic_similarity": item["similarity"],
                "confidence": item["confidence"],
            }
            src_len = len(source_text.split()) or 1
            tgt_len = len(translation.split()) or 1
            metrics["length_ratio"] = min(src_len / tgt_len, tgt_len / src_len)
            src_sent = len([s for s in source_text.split(".") if s.strip()]) or 1
            tgt_sent = len([s for s in translation.split(".") if s.strip()]) or 1
            metrics["structure_preservation"] = min(src_sent / tgt_sent, tgt_sent / src_sent)
            src_lang = self._lang_detector.detect(source_text).get("code", "und")
            tgt_lang = self._lang_detector.detect(translation).get("code", "und")
            lang_detect_score = 1.0 if src_lang != tgt_lang else 0.0
            mismatch_metric = 1.0 if src_lang != tgt_lang else 0.0
            item["metrics"] = metrics
            item["quality_score"] = self.calculate_composite_quality_score(metrics)

        ranking_res["ranked_translations"].sort(key=lambda d: d["quality_score"], reverse=True)
        return ranking_res

    # ------------------------------------------------------------------
    def calculate_composite_quality_score(
        self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None
    ) -> float:
        weights = weights or self.weights
        total, total_w = 0.0, 0.0
        for comp, w in weights.items():
            if comp in metrics:
                total += metrics[comp] * w
                total_w += w
        return max(0.0, min(1.0, total / total_w if total_w else 0.0))

# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def analyze_translation(
    source_text: str,
    translation: str,
    model_name: str = "all-MiniLM-L6-v2",
    weights: Optional[Dict[str, float]] = None,
    *,
    use_cache: bool = True,
) -> Dict[str, Any]:
    eg = EmbeddingGenerator(model_name, use_cache=use_cache)
    analyzer = TranslationQualityAnalyzer(embedding_generator=eg, weights=weights, use_cache=use_cache)
    return analyzer.analyze_pair(source_text, translation)


def rank_translations_with_quality(
    source_text: str,
    candidates: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    confidence_method: str = "distribution",
    include_diagnostics: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    eg = EmbeddingGenerator(model_name, use_cache=True)
    analyzer = TranslationQualityAnalyzer(embedding_generator=eg, weights=weights, use_cache=True)
    return analyzer.rank_candidates(
        source_text,
        candidates,
        confidence_method=confidence_method,
        include_diagnostics=include_diagnostics,
    ) 