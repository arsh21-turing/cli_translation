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

# Optional Groq integration
try:
    from groq_evaluator import GroqEvaluator  # type: ignore
except ImportError:  # pragma: no cover
    GroqEvaluator = None  # type: ignore

from language_utils import LanguageDetector
from embedding_cache import embedding_cache
from segment_alignment import WeakAlignmentDetector

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
    """Comprehensive translation quality analysis."""
    
    def __init__(self, embedding_generator=None, ranker=None, config_manager=None, groq_evaluator=None):
        """
        Initialize the translation quality analyzer.
        
        Args:
            embedding_generator: EmbeddingGenerator instance for generating embeddings
            ranker: TranslationRanker instance for ranking translations
            config_manager: ConfigManager instance with quality score weights
            groq_evaluator: GroqEvaluator instance for LLM-based evaluation
        """
        # Initialize embedding generator if not provided
        if embedding_generator is None:
            from embedding_generator import EmbeddingGenerator
            self.embedding_generator = EmbeddingGenerator()
        else:
            self.embedding_generator = embedding_generator
            
        # Initialize ranker if not provided
        if ranker is None:
            from translation_ranker import TranslationRanker
            self.ranker = TranslationRanker(embedding_generator=self.embedding_generator)
        else:
            self.ranker = ranker
        
        # Initialize config manager if not provided
        if config_manager is None:
            from config_manager import ConfigManager
            self.config_manager = ConfigManager()
        else:
            self.config_manager = config_manager
            
        # Initialize Groq evaluator if not provided
        self.groq_evaluator = groq_evaluator
        
    def analyze_pair(self, source_text, translation, use_groq=False, detailed=False, 
                    detect_weak_alignments=False, segment_type=None, custom_weights=None):
        """
        Analyze a single translation pair with enhanced Groq integration.
        
        Args:
            source_text: The original text to be translated
            translation: The translated text to analyze
            use_groq: Whether to use Groq LLM for enhanced evaluation
            detailed: Whether to return detailed Groq evaluation
            detect_weak_alignments: Whether to analyze segment-level alignment issues
            segment_type: Type of segmentation for alignment analysis (if None, uses config)
            custom_weights: Custom weights for this specific analysis
            
        Returns:
            Dictionary with detailed quality metrics and composite score
        """
        # Base analysis using embeddings
        source_embedding = self.embedding_generator.generate_embedding(source_text)
        translation_embedding = self.embedding_generator.generate_embedding(translation)
        
        from similarity_calculator import cosine_similarity
        similarity = cosine_similarity(source_embedding, translation_embedding)
        
        # Basic metrics
        metrics = {
            "embedding_similarity": similarity,
            "source_length": len(source_text),
            "translation_length": len(translation),
            "length_ratio": len(translation) / max(1, len(source_text))
        }
        
        # Add analysis of text characteristics
        metrics.update(self._analyze_text_characteristics(source_text, translation))
        
        # Enhance with Groq LLM evaluation if requested and available
        if use_groq and self.groq_evaluator:
            # Try to detect languages automatically
            try:
                from language_utils import detect_language
                source_lang = detect_language(source_text)
                target_lang = detect_language(translation)
            except ImportError:
                source_lang = None
                target_lang = None
                
            groq_evaluation = self.groq_evaluator.evaluate_translation(
                source_text=source_text,
                translation=translation,
                source_lang=source_lang,
                target_lang=target_lang,
                detailed=detailed
            )
            
            # Add Groq evaluation metrics
            if "error" not in groq_evaluation:
                metrics["groq_evaluation"] = groq_evaluation
                
                # Add overall score (normalized to 0-1 range later)
                if "overall_score" in groq_evaluation:
                    metrics["groq_score"] = groq_evaluation["overall_score"]
                
                # Add detailed metrics if available
                if detailed and "accuracy" in groq_evaluation:
                    metrics["accuracy"] = groq_evaluation["accuracy"]
                    metrics["fluency"] = groq_evaluation["fluency"]
                    metrics["terminology"] = groq_evaluation["terminology"]
                    metrics["style"] = groq_evaluation["style"]
                    
                    # Add error count if available
                    if "errors" in groq_evaluation:
                        metrics["error_count"] = len(groq_evaluation["errors"])
        
        # Add weak alignment analysis if requested
        if detect_weak_alignments:
            # Use segment_type from config if not provided
            if segment_type is None:
                segment_type = self.config_manager.get_alignment_setting('segment_type', 'sentence')
                
            # Get similarity threshold from config
            similarity_threshold = self.config_manager.get_alignment_setting('similarity_threshold', 0.75)
                
            # Import the WeakAlignmentDetector here to use it
            weak_alignment_detector = WeakAlignmentDetector(
                embedding_generator=self.embedding_generator,
                groq_evaluator=self.groq_evaluator if use_groq else None,
                similarity_threshold=similarity_threshold,
                segment_type=segment_type
            )
            
            alignment_analysis = weak_alignment_detector.detect_weak_alignments(
                source_text=source_text,
                translation=translation,
                use_groq=use_groq,
                detailed=detailed
            )
            
            metrics["alignment_analysis"] = alignment_analysis
            
            # Add alignment score to main metrics
            if use_groq and "enhanced_summary" in alignment_analysis:
                metrics["alignment_score"] = alignment_analysis["enhanced_summary"].get("combined_score", 0)
            else:
                metrics["alignment_score"] = alignment_analysis["weak_alignment_summary"].get("alignment_score", 0)
                
            # Add pattern metrics for penalties
            metrics["recurring_patterns"] = len(alignment_analysis["segment_analysis"]["recurring_patterns"])
            
            # Position-based pattern flags
            position_patterns = [p for p in alignment_analysis["segment_analysis"]["recurring_patterns"] 
                               if p["type"] in ["beginning_weakness", "middle_weakness", "end_weakness"]]
            metrics["position_patterns"] = len(position_patterns)
        
        # Calculate composite quality score with configurable weights
        metrics["composite_score"] = self.calculate_composite_quality_score(metrics, custom_weights)
        
        return metrics
    
    def _analyze_text_characteristics(self, source_text, translation):
        """
        Extract additional text characteristics for more nuanced quality assessment.
        
        Args:
            source_text: Original text
            translation: Translated text
            
        Returns:
            Dictionary of text characteristics metrics
        """
        # This is a stub that could be expanded with more sophisticated analysis
        metrics = {}
        
        # Analyze sentence count ratio (helps identify missing/added sentences)
        import re
        source_sentences = len(re.split(r'[.!?]+', source_text.strip()))
        translation_sentences = len(re.split(r'[.!?]+', translation.strip()))
        
        if source_sentences > 0:
            metrics["sentence_ratio"] = translation_sentences / source_sentences
        else:
            metrics["sentence_ratio"] = 1.0
        
        # Flag significant sentence count differences
        metrics["sentence_mismatch"] = abs(source_sentences - translation_sentences) > 1
        
        # Calculate other characteristics that might be useful for quality assessment
        # Examples: named entity preservation, number preservation, etc.
        
        return metrics
    
    def rank_candidates(self, source_text, candidates, confidence_method='distribution', 
                       include_diagnostics=False, use_groq=False, detect_weak_alignments=False,
                       custom_weights=None):
        """
        Rank multiple translations with enhanced Groq integration.
        
        Args:
            source_text: The original text to be translated
            candidates: List of translation candidates
            confidence_method: Method for confidence calculation
            include_diagnostics: Whether to include clustering diagnostics
            use_groq: Whether to use Groq for enhanced comparison
            detect_weak_alignments: Whether to detect weak alignments in the top candidates
            custom_weights: Custom weights for this specific ranking
            
        Returns:
            Dictionary with ranked translations and quality scores
        """
        # First use embedding-based ranking
        ranked_translations = self.ranker.rank_translations(source_text, candidates)
        
        # Add confidence scores
        from translation_ranker import ConfidenceScorer
        confidence_scorer = ConfidenceScorer()
        similarities = [item["similarity"] for item in ranked_translations]
        confidence_scores = confidence_scorer.calculate_confidence(similarities, method=confidence_method)
        
        for i, score in enumerate(confidence_scores):
            ranked_translations[i]["confidence"] = score
            
        # Add detailed quality metrics by analyzing each pair
        for i, candidate in enumerate(ranked_translations):
            translation = candidate["translation"]
            
            # For efficiency, only use detailed analysis for top candidates
            use_detailed = use_groq and i < 3  # Only for top 3 candidates
            detect_alignment = detect_weak_alignments and i < 3  # Only for top 3 candidates
            
            metrics = self.analyze_pair(
                source_text=source_text, 
                translation=translation, 
                use_groq=use_groq,
                detailed=use_detailed,
                detect_weak_alignments=detect_alignment,
                custom_weights=custom_weights
            )
            
            ranked_translations[i]["metrics"] = metrics
            ranked_translations[i]["composite_score"] = metrics["composite_score"]
            
        # If requested and available, use Groq for enhanced comparison of all candidates
        if use_groq and self.groq_evaluator and len(candidates) > 1:
            # Try to detect languages automatically
            try:
                from language_utils import detect_language
                source_lang = detect_language(source_text)
                target_lang = detect_language(candidates[0])  # Assume all candidates are same language
            except ImportError:
                source_lang = None
                target_lang = None
                
            groq_comparison = self.groq_evaluator.compare_translations(
                source_text=source_text,
                translations=[item["translation"] for item in ranked_translations],
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # If successful, add Groq comparison data
            if "error" not in groq_comparison and "rankings" in groq_comparison:
                # Create mapping from Groq rankings to our rankings
                groq_rankings = {r["translation_index"]: r for r in groq_comparison["rankings"]}
                
                # Add Groq rankings to our results
                for i in range(len(ranked_translations)):
                    groq_index = i + 1  # Groq uses 1-based indexing
                    if groq_index in groq_rankings:
                        ranked_translations[i]["groq_rank"] = groq_rankings[groq_index]["rank"]
                        ranked_translations[i]["groq_score"] = groq_rankings[groq_index]["score"] / 10.0
                        ranked_translations[i]["groq_comments"] = groq_rankings[groq_index]["comments"]
                        ranked_translations[i]["groq_strengths"] = groq_rankings[groq_index].get("strengths", [])
                        ranked_translations[i]["groq_weaknesses"] = groq_rankings[groq_index].get("weaknesses", [])
                        
                        # Adjust composite score to incorporate Groq's ranking
                        # This gives additional weight to Groq's comparative assessment
                        rank_factor = (len(ranked_translations) - groq_rankings[groq_index]["rank"] + 1) / len(ranked_translations)
                        current_score = ranked_translations[i]["composite_score"]
                        comparison_weight = self.config_manager.get_weight("groq_comparison_weight", 0.3)
                        ranked_translations[i]["composite_score"] = current_score * (1 - comparison_weight) + rank_factor * comparison_weight
                        
                # Add overall comparison summary
                if "comparison_summary" in groq_comparison:
                    for i in range(len(ranked_translations)):
                        ranked_translations[i]["groq_comparison_summary"] = groq_comparison["comparison_summary"]
        
        # Re-rank based on composite score if we used Groq
        if use_groq or custom_weights:
            ranked_translations.sort(key=lambda x: x["composite_score"], reverse=True)
                
        return {
            "ranked_translations": ranked_translations,
            "source_text": source_text,
            "confidence_method": confidence_method,
            "includes_groq_evaluation": use_groq and self.groq_evaluator is not None,
            "includes_alignment_analysis": detect_weak_alignments,
            "used_custom_weights": custom_weights is not None
        }
    
    def calculate_composite_quality_score(self, metrics, custom_weights=None):
        """
        Calculate a composite quality score that integrates alignment analysis.
        
        Args:
            metrics: Dictionary of quality metrics including embedding similarity,
                    alignment scores, length ratio, and optional Groq scores
            custom_weights: Optional dictionary with custom weights to override config
            
        Returns:
            Composite quality score (0-1)
        """
        # Get weights from config
        weights = {}
        for weight_name in [
            # Base metrics
            "embedding_similarity", "length_ratio_penalty", "alignment_score", 
            "recurring_pattern_penalty", "position_pattern_penalty", "groq_score",
            
            # Detailed metrics
            "accuracy", "fluency", "terminology", "style",
            
            # Group weights
            "embedding_metrics_weight", "alignment_metrics_weight",
            "groq_simple_metrics_weight", "groq_detailed_metrics_weight"
        ]:
            weights[weight_name] = self.config_manager.get_weight(weight_name, 0.5)
        
        # Override with custom weights if provided
        if custom_weights:
            weights.update(custom_weights)
        
        # Initialize score components and weights
        components = {
            "embedding": {"score": 0, "weight": 0},
            "alignment": {"score": 0, "weight": 0},
            "groq_simple": {"score": 0, "weight": 0},
            "groq_detailed": {"score": 0, "weight": 0}
        }
        
        # Add embedding-based metrics
        if "embedding_similarity" in metrics:
            sim_score = metrics["embedding_similarity"] * weights["embedding_similarity"]
            components["embedding"]["score"] += sim_score
            components["embedding"]["weight"] += weights["embedding_similarity"]
        
        # Add length ratio penalty if available
        if "length_ratio" in metrics:
            length_ratio = metrics["length_ratio"]
            # Penalize translations that are too short or too long
            length_penalty = 1.0
            if length_ratio < 0.7 or length_ratio > 1.4:
                length_penalty = 0.8
            elif length_ratio < 0.5 or length_ratio > 1.7:
                length_penalty = 0.6
                
            components["embedding"]["score"] += length_penalty * weights["length_ratio_penalty"]
            components["embedding"]["weight"] += weights["length_ratio_penalty"]
        
        # Add segment alignment score if available
        if "alignment_score" in metrics:
            align_score = metrics["alignment_score"] * weights["alignment_score"]
            components["alignment"]["score"] += align_score
            components["alignment"]["weight"] += weights["alignment_score"]
            
            # Add penalties for recurring patterns if available
            if "recurring_patterns" in metrics and metrics["recurring_patterns"] > 0:
                # Calculate penalty (more patterns = higher penalty)
                pattern_count = metrics["recurring_patterns"]
                pattern_penalty = max(0, 1.0 - (pattern_count * 0.1))  # Each pattern reduces score by 10%
                
                # Apply penalty with configurable weight
                components["alignment"]["score"] += pattern_penalty * weights["recurring_pattern_penalty"]
                components["alignment"]["weight"] += weights["recurring_pattern_penalty"]
            
            # Add penalties for position-based patterns
            if "position_patterns" in metrics and metrics["position_patterns"] > 0:
                position_count = metrics["position_patterns"]
                position_penalty = max(0, 1.0 - (position_count * 0.15))  # Each pattern reduces score by 15%
                
                components["alignment"]["score"] += position_penalty * weights["position_pattern_penalty"]
                components["alignment"]["weight"] += weights["position_pattern_penalty"]
        
        # Add Groq's simple evaluation if available
        if "groq_score" in metrics:
            # Normalize from 0-10 to 0-1 range
            normalized_score = metrics["groq_score"] / 10.0 if isinstance(metrics["groq_score"], (int, float)) else 0
            components["groq_simple"]["score"] += normalized_score * weights["groq_score"]
            components["groq_simple"]["weight"] += weights["groq_score"]
        
        # Add Groq's detailed evaluation if available
        if all(component in metrics for component in ["accuracy", "fluency", "terminology", "style"]):
            for component in ["accuracy", "fluency", "terminology", "style"]:
                # Normalize from 0-10 to 0-1 range
                normalized_value = metrics[component] / 10.0 if isinstance(metrics[component], (int, float)) else 0
                components["groq_detailed"]["score"] += normalized_value * weights[component]
                components["groq_detailed"]["weight"] += weights[component]
        
        # Calculate final score from components
        final_score = 0
        total_weight = 0
        
        # Process each component group
        for group, data in components.items():
            if data["weight"] > 0:
                # Calculate group score
                group_score = data["score"] / data["weight"]
                
                # Apply group weight
                group_weight = weights[f"{group}_weight"]
                
                final_score += group_score * group_weight
                total_weight += group_weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_score /= total_weight
        
        # Ensure score is in 0-1 range
        return max(0.0, min(1.0, final_score))
    
    def get_default_weights(self):
        """
        Get the default quality score weights from config manager.
        
        Returns:
            Dictionary of weight names and values
        """
        return self.config_manager.get_quality_weights()
    
    def set_weight(self, weight_name, value):
        """
        Set a specific quality weight in the config.
        
        Args:
            weight_name: Name of the weight parameter
            value: Weight value (float)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config_manager.set_weight(weight_name, float(value))
            return True
        except Exception:
            return False
    
    def update_weights(self, weights_dict):
        """
        Update multiple weights at once.
        
        Args:
            weights_dict: Dictionary of weight names and values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config_manager.update_weights(weights_dict)
            return True
        except Exception:
            return False
    
    def save_weights_to_config(self, config_path):
        """
        Save current weights to a config file.
        
        Args:
            config_path: Path to save the config file
            
        Returns:
            True if successful, False otherwise
        """
        return self.config_manager.save(config_path)

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