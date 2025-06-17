"""translation_ranker.py
Module for ranking multiple translation candidates based on their semantic
similarity to a source sentence using the existing embedding / similarity
infrastructure.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

from config_manager import ConfigManager
from model_loader import ModelLoader, MultilingualModelManager
from text_processor import TextProcessor
from embedding_generator import MultilingualVectorGenerator
from similarity_calculator import SimilarityCalculator

# External libs for diagnostics (optional during runtime)
import numpy as np
from collections import Counter
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore
    silhouette_score = None  # type: ignore

logger = logging.getLogger(__name__)


class TranslationRanker:
    """Rank translation candidates by semantic similarity to a source text."""

    def __init__(
        self,
        similarity_calculator: Optional[SimilarityCalculator] = None,
        metric: str = "cosine",
    ) -> None:
        """Create a TranslationRanker.

        Args:
            similarity_calculator: Optional pre-constructed SimilarityCalculator.
                If *None*, a default calculator is built from the standard
                configuration and model-loading pipeline.
            metric: Similarity metric to employ ("cosine", "euclidean", "dot", …).
        """
        self.metric = metric

        if similarity_calculator is None:
            # Build default NLP pipeline components.
            logger.debug("Initialising default SimilarityCalculator for ranker …")
            config = ConfigManager()
            model_loader = ModelLoader(config)
            multilingual_manager = MultilingualModelManager(config, model_loader)
            text_processor = TextProcessor()
            vector_generator = MultilingualVectorGenerator(  # type: ignore[arg-type]
                multilingual_manager, text_processor, config
            )
            similarity_calculator = SimilarityCalculator(vector_generator, config)

        self.sim_calc: SimilarityCalculator = similarity_calculator

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def rank_translations(
        self, source_text: str, candidates: List[str]
    ) -> List[Dict[str, float | str]]:
        """Return *candidates* ordered by faithfulness to *source_text*.

        Each result dictionary contains::

            {
                "translation": <candidate string>,
                "similarity":  <float 0-1>
            }
        """
        if not source_text:
            logger.warning("Empty source text supplied to TranslationRanker")
            return []

        if not candidates:
            return []

        # Generate embeddings for all candidates in one shot (efficient) if the
        # underlying similarity calculator exposes a *vector_generator* with a
        # *generate_vectors* method. This is skipped in unit-test scenarios that
        # rely on lightweight mocks.
        candidate_vectors_np = None
        if hasattr(self.sim_calc, "vector_generator") and hasattr(
            self.sim_calc.vector_generator, "generate_vectors"
        ):
            try:
                candidate_vectors_np = self.sim_calc.vector_generator.generate_vectors(candidates)
            except Exception:
                candidate_vectors_np = None

        scored: List[Dict[str, float | str]] = []
        for idx, cand in enumerate(candidates):
            sim = self.sim_calc.calculate_similarity(
                source_text, cand, metric=self.metric
            )
            scored.append({"translation": cand, "similarity": float(sim)})

        # Store for diagnostics (if available)
        if candidate_vectors_np is not None:
            self._candidate_vectors = candidate_vectors_np  # shape (n, dim)

        # Highest similarity first
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored

    # -----------------------------------------------------------------------------
    # Clustering diagnostics
    # -----------------------------------------------------------------------------
    def get_clustering_diagnostics(self, max_clusters: int | None = None) -> Dict[str, object]:
        """Return clustering statistics for already-ranked candidates.

        Uses K-Means and silhouette score (requires scikit-learn). If sklearn is
        unavailable or candidate count < 2, graceful degradation occurs.
        """

        vectors = getattr(self, "_candidate_vectors", None)
        if vectors is None or len(vectors) < 2 or KMeans is None:
            return {
                "optimal_clusters": 1,
                "silhouette_scores": {},
                "cluster_assignments": [0] * (len(vectors) if vectors is not None else 0),
                "cluster_sizes": {0: len(vectors) if vectors is not None else 0},
                "cluster_cohesion": 1.0,
                "cluster_separation": 0.0,
                "variance_explained": 0.0,
            }

        n_samples = vectors.shape[0]
        if max_clusters is None:
            max_clusters = min(5, n_samples - 1)

        best_score = -1.0
        best_k = 1
        best_labels = np.zeros(n_samples, dtype=int)
        silhouette_scores: Dict[int, float] = {}

        for k in range(2, max_clusters + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(vectors)
                if len(set(labels)) < 2:
                    continue
                score = float(silhouette_score(vectors, labels))
                silhouette_scores[k] = score
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_km = km  # type: ignore
            except Exception:
                continue

        if best_k == 1:
            best_labels = np.zeros(n_samples, dtype=int)

        # Cluster sizes
        counts = Counter(best_labels)
        cluster_sizes = {int(k): int(v) for k, v in counts.items()}

        # Cohesion: average cosine similarity within clusters
        cohesion_vals = []
        for cid in counts:
            idxs = np.where(best_labels == cid)[0]
            if len(idxs) < 2:
                continue
            sub = vectors[idxs]
            # pairwise cosine similarities using dot since vectors normalized not guaranteed
            sims = np.dot(sub, sub.T)
            upper = sims[np.triu_indices_from(sims, k=1)]
            cohesion_vals.extend(upper.tolist())
        cluster_cohesion = float(np.mean(cohesion_vals)) if cohesion_vals else 1.0

        # Separation: distances between centroids
        centroids = {}
        for cid in counts:
            centroids[cid] = np.mean(vectors[best_labels == cid], axis=0)
        centroids_list = list(centroids.values())
        separation_vals = []
        for i in range(len(centroids_list)):
            for j in range(i + 1, len(centroids_list)):
                cos = np.dot(centroids_list[i], centroids_list[j]) / (
                    np.linalg.norm(centroids_list[i]) * np.linalg.norm(centroids_list[j]) + 1e-9
                )
                separation_vals.append(1.0 - cos)
        cluster_separation = float(np.mean(separation_vals)) if separation_vals else 0.0

        # Variance explained via inertia
        variance_explained = 0.0
        if best_k != 1 and "best_km" in locals():
            inertia = best_km.inertia_  # type: ignore
            total_var = np.sum(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1) ** 2)
            if total_var > 0:
                variance_explained = float(1.0 - inertia / total_var)

        return {
            "optimal_clusters": best_k,
            "silhouette_scores": {int(k): float(v) for k, v in silhouette_scores.items()},
            "cluster_assignments": [int(lbl) for lbl in best_labels.tolist()],
            "cluster_sizes": cluster_sizes,
            "cluster_cohesion": cluster_cohesion,
            "cluster_separation": cluster_separation,
            "variance_explained": variance_explained,
        }


# -----------------------------------------------------------------------------
# Confidence scoring utilities
# -----------------------------------------------------------------------------
import math
from statistics import mean, stdev


class ConfidenceScorer:
    """Compute reliability/ confidence values for similarity scores (0-1)."""

    # Weights for combination – can be tuned later
    _BASE_WEIGHT = 0.5
    _Z_WEIGHT = 0.25
    _POS_WEIGHT = 0.25

    def calculate_confidence(
        self, similarities: List[float], method: str = "distribution"
    ) -> List[float]:
        """Return a confidence for each similarity in *similarities*.

        Three strategies are available:
            - "distribution" (default): z-score & distribution based.
            - "gap": use gaps between consecutive sorted similarities.
            - "range": linear mapping relative to min/max.
        """
        if not similarities:
            return []

        if method == "gap":
            return self._gap_based(similarities)
        if method == "range":
            return self._range_based(similarities)
        # Fallback to distribution
        return self._distribution_based(similarities)

    # ------------------------------------------------------------------
    def _distribution_based(self, similarities: List[float]) -> List[float]:
        if len(similarities) == 1:
            return [min(max(similarities[0], 0.0), 1.0)]

        mu = mean(similarities)
        try:
            sigma = stdev(similarities)
        except Exception:
            sigma = 1e-6
        if sigma == 0:
            sigma = 1e-6

        min_sim = min(similarities)
        max_sim = max(similarities)
        sim_range = max(max_sim - min_sim, 1e-6)

        confidences: List[float] = []
        for s in similarities:
            base = self._BASE_WEIGHT * s
            z = (s - mu) / sigma
            z_conf = self._Z_WEIGHT * (1 / (1 + math.exp(-z)))  # sigmoid 0-1
            pos = self._POS_WEIGHT * ((s - min_sim) / sim_range)
            conf = base + z_conf + pos
            confidences.append(max(0.0, min(1.0, conf)))
        return confidences

    # ------------------------------------------------------------------
    def _gap_based(self, similarities: List[float]) -> List[float]:
        if len(similarities) == 1:
            return [min(max(similarities[0], 0.0), 1.0)]

        ordered = sorted(similarities, reverse=True)
        gaps = [ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1)]
        gaps.append(0.0)  # last gap 0
        max_gap = max(max(gaps), 1e-6)

        gap_conf_map = {}
        for sim, gap in zip(ordered, gaps):
            base = 0.7 * sim
            gap_conf = 0.3 * (gap / max_gap)
            gap_conf_map[sim] = max(0.0, min(1.0, base + gap_conf))

        return [gap_conf_map[s] for s in similarities]

    # ------------------------------------------------------------------
    def _range_based(self, similarities: List[float]) -> List[float]:
        if len(similarities) == 1:
            return [min(max(similarities[0], 0.0), 1.0)]

        min_sim = min(similarities)
        max_sim = max(similarities)
        sim_range = max(max_sim - min_sim, 1e-6)
        confidences = [0.3 + 0.7 * ((s - min_sim) / sim_range) for s in similarities]
        return [max(0.0, min(1.0, c)) for c in confidences]


# -----------------------------------------------------------------------------
# Helper function that combines ranking & confidence
# -----------------------------------------------------------------------------

def calculate_translation_confidence(
    source_text: str,
    candidates: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    confidence_method: str = "distribution",
    include_diagnostics: bool = False,
) -> Dict[str, object]:
    """Return ranked translations with confidence and optional diagnostics."""
    # Build minimal pipeline using translation ranker default behaviour
    ranker = TranslationRanker(metric="cosine")
    ranked = ranker.rank_translations(source_text, candidates)

    sims = [item["similarity"] for item in ranked]
    conf_scorer = ConfidenceScorer()
    confs = conf_scorer.calculate_confidence(sims, method=confidence_method)
    for item, conf in zip(ranked, confs):
        item["confidence"] = conf

    result: Dict[str, object] = {"ranked_translations": ranked}
    if include_diagnostics:
        result["diagnostics"] = ranker.get_clustering_diagnostics()
    return result 