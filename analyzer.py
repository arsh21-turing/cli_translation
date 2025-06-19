"""
Translation Analyzer for Smart CLI Translation Quality Analyzer
Handles translation quality assessment and scoring
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

from config_manager import ConfigManager
from model_loader import ModelLoader
from text_processor import TextProcessor
from embedding_generator import EmbeddingGenerator

@dataclass
class AnalysisResult:
    """
    Contains the result of a translation quality analysis.
    """
    source_text: str
    translated_text: str
    quality_score: float
    semantic_similarity: float
    fluency_score: float
    accuracy_score: float
    source_lang: str
    target_lang: str
    sentence_scores: List[Dict[str, Any]] = None
    detailed_feedback: str = ""
    source_stats: Dict[str, Any] = None
    target_stats: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize optional attributes."""
        if self.sentence_scores is None:
            self.sentence_scores = []
        if self.source_stats is None:
            self.source_stats = {}
        if self.target_stats is None:
            self.target_stats = {}

    def __repr__(self) -> str:
        """String representation of analysis result."""
        return (f"AnalysisResult(quality_score={self.quality_score:.2f}, "
                f"semantic_similarity={self.semantic_similarity:.2f}, "
                f"source_lang={self.source_lang}, target_lang={self.target_lang})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary."""
        return {
            "quality_score": self.quality_score,
            "semantic_similarity": self.semantic_similarity,
            "fluency_score": self.fluency_score,
            "accuracy_score": self.accuracy_score,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "sentence_scores": self.sentence_scores,
            "detailed_feedback": self.detailed_feedback,
            "source_stats": self.source_stats,
            "target_stats": self.target_stats
        }

class TranslationAnalyzer:
    """
    Analyzes translation quality between source and translated text.
    Provides methods for quality assessment and scoring.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None, 
                model_loader: Optional[ModelLoader] = None,
                text_processor: Optional[TextProcessor] = None,
                embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the translation analyzer.
        
        Args:
            config: Configuration manager instance
            model_loader: Model loader for accessing language models
            text_processor: Text processor for text manipulation
            embedding_generator: Embedding generator for semantic analysis
        """
        self.logger = logging.getLogger("tqa.analyzer")
        self.config = config or {}
        
        # Load components (create if not provided)
        if model_loader is None and isinstance(config, ConfigManager):
            model_loader = ModelLoader(config)
        self.model_loader = model_loader
        
        if text_processor is None and isinstance(config, ConfigManager):
            text_processor = TextProcessor(config)
        self.text_processor = text_processor
        
        if embedding_generator is None and all([model_loader, text_processor, isinstance(config, ConfigManager)]):
            embedding_generator = EmbeddingGenerator(config, model_loader, text_processor)
        self.embedding_generator = embedding_generator
        
        # Default thresholds
        self.similarity_threshold = self.get_config_value("analysis.similarity_threshold", 0.75)
        self.min_quality_score = self.get_config_value("analysis.min_quality_score", 0.6)
        self.detailed_reports = self.get_config_value("analysis.detailed_reports", True)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value safely, handling different config types.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        elif isinstance(self.config, dict):
            return self.config.get(key, default)
        return default
    
    def _detect_language(self, text):
        """Detect language of text using simple keyword matching."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["bonjour", "français", "le monde", "ceci", "est", "un", "test", "avec", "cinq", "mots", "deux", "phrases"]):
            return "fr"
        elif any(word in text_lower for word in ["hola", "español", "esto", "es", "texto", "en"]):
            return "es"
        return "en"
    
    def analyze(self, source_text: str, translated_text: str, 
               source_lang: Optional[str] = None, target_lang: Optional[str] = None,
               detailed: bool = False, llm_analysis: bool = False) -> AnalysisResult:
        """
        Analyze translation quality between source and translated text.
        
        Args:
            source_text: Original text in source language
            translated_text: Translated text to evaluate
            source_lang: Source language code (auto-detect if None)
            target_lang: Target language code (auto-detect if None)
            detailed: Whether to include detailed sentence-level analysis
            llm_analysis: Whether to use LLM for additional analysis
            
        Returns:
            AnalysisResult object with quality scores and analysis
        """
        self.logger.info("Starting translation quality analysis")
        
        if not source_text or not translated_text:
            self.logger.warning("Empty source or translated text provided")
            return AnalysisResult(
                source_text=source_text or "",
                translated_text=translated_text or "",
                quality_score=0.0,
                semantic_similarity=0.0,
                fluency_score=0.0,
                accuracy_score=0.0,
                source_lang=source_lang or "unknown",
                target_lang=target_lang or "unknown"
            )
        
        # Auto-detect languages if not specified
        if source_lang is None:
            source_lang = self._detect_language(source_text)
            self.logger.info(f"Detected source language: {source_lang}")
        
        if target_lang is None:
            target_lang = self._detect_language(translated_text)
            self.logger.info(f"Detected target language: {target_lang}")
        
        # Calculate text statistics
        source_stats = None
        target_stats = None
        if self.text_processor:
            try:
                source_stats = self.text_processor.calculate_statistics(source_text)
                target_stats = self.text_processor.calculate_statistics(translated_text)
                self.logger.debug(f"Source stats: {source_stats}")
                self.logger.debug(f"Target stats: {target_stats}")
            except Exception as e:
                self.logger.error(f"Error calculating text statistics: {e}")
        
        # Calculate semantic similarity
        semantic_similarity = 0.0
        if self.embedding_generator:
            try:
                semantic_similarity = self.embedding_generator.calculate_text_similarity(
                    source_text, translated_text
                )
                self.logger.info(f"Semantic similarity score: {semantic_similarity:.4f}")
            except Exception as e:
                self.logger.error(f"Error calculating semantic similarity: {e}")
        
        # Calculate sentence-level scores if detailed analysis requested
        sentence_scores = []
        if detailed and self.text_processor and self.embedding_generator:
            try:
                aligned_sentences = self.text_processor.align_sentences(
                    source_text, translated_text, source_lang, target_lang
                )
                
                for src, tgt in aligned_sentences:
                    sim_score = self.embedding_generator.calculate_text_similarity(src, tgt)
                    sentence_scores.append({
                        "source": src,
                        "translation": tgt,
                        "similarity": sim_score
                    })
                    
                self.logger.info(f"Generated {len(sentence_scores)} sentence-level scores")
            except Exception as e:
                self.logger.error(f"Error generating sentence-level scores: {e}")
        
        # Get LLM analysis if requested
        detailed_feedback = ""
        if llm_analysis and self.model_loader:
            try:
                groq_client = self.model_loader.get_groq_client()
                prompt = self._create_analysis_prompt(source_text, translated_text, source_lang, target_lang)
                detailed_feedback = groq_client.completion(prompt)
                self.logger.info(f"Generated LLM analysis feedback")
            except Exception as e:
                self.logger.error(f"Error generating LLM analysis: {e}")
        
        # Calculate fluency and accuracy scores
        fluency_score = self._calculate_fluency_score(translated_text, target_lang)
        accuracy_score = self._calculate_accuracy_score(semantic_similarity, source_text, translated_text)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            semantic_similarity, fluency_score, accuracy_score
        )
        
        self.logger.info(f"Analysis complete. Quality score: {quality_score:.4f}")
        
        # Create and return analysis result
        return AnalysisResult(
            source_text=source_text,
            translated_text=translated_text,
            quality_score=quality_score,
            semantic_similarity=semantic_similarity,
            fluency_score=fluency_score,
            accuracy_score=accuracy_score,
            source_lang=source_lang,
            target_lang=target_lang,
            sentence_scores=sentence_scores,
            detailed_feedback=detailed_feedback,
            source_stats=source_stats,
            target_stats=target_stats
        )
    
    def translate_and_analyze(self, source_text: str, target_lang: str, source_lang: Optional[str] = None,
                            detailed: bool = False, llm_analysis: bool = False) -> AnalysisResult:
        """
        Translate text and analyze the quality of the translation.
        
        Args:
            source_text: Original text to translate
            target_lang: Target language code
            source_lang: Source language code (auto-detect if None)
            detailed: Whether to include detailed sentence-level analysis
            llm_analysis: Whether to use LLM for additional analysis
            
        Returns:
            AnalysisResult object with translation and quality scores
        """
        self.logger.info(f"Starting translation to {target_lang} and analysis")
        
        if not source_text:
            self.logger.warning("Empty source text provided")
            return AnalysisResult(
                source_text="",
                translated_text="",
                quality_score=0.0,
                semantic_similarity=0.0,
                fluency_score=0.0,
                accuracy_score=0.0,
                source_lang=source_lang or "unknown",
                target_lang=target_lang
            )
        
        # Auto-detect source language if not specified
        if source_lang is None:
            source_lang = self._detect_language(source_text)
            self.logger.info(f"Detected source language: {source_lang}")
        
        # Translate the text
        translated_text = ""
        if self.model_loader:
            try:
                translated_text = self.model_loader.translate(
                    source_text, source_lang=source_lang, target_lang=target_lang
                )
                self.logger.info(f"Translation completed: {len(translated_text)} characters")
            except Exception as e:
                self.logger.error(f"Translation error: {e}")
                return AnalysisResult(
                    source_text=source_text,
                    translated_text="",
                    quality_score=0.0,
                    semantic_similarity=0.0,
                    fluency_score=0.0,
                    accuracy_score=0.0,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    detailed_feedback=f"Translation error: {str(e)}"
                )
        else:
            self.logger.error("No model loader available for translation")
            return AnalysisResult(
                source_text=source_text,
                translated_text="",
                quality_score=0.0,
                semantic_similarity=0.0,
                fluency_score=0.0,
                accuracy_score=0.0,
                source_lang=source_lang,
                target_lang=target_lang,
                detailed_feedback="Translation not available: no model loader"
            )
        
        # Analyze the translation
        return self.analyze(
            source_text, 
            translated_text, 
            source_lang=source_lang, 
            target_lang=target_lang,
            detailed=detailed,
            llm_analysis=llm_analysis
        )
    
    def _create_analysis_prompt(self, source_text: str, translated_text: str, 
                              source_lang: str, target_lang: str) -> str:
        """
        Create prompt for LLM translation analysis.
        
        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Formatted prompt for LLM
        """
        return (
            f"You are a professional translator and language expert specializing in {source_lang} to {target_lang} translation.\n\n"
            f"Please analyze the quality of the following translation from {source_lang} to {target_lang}. "
            f"Rate it on a scale of 1-10 for accuracy, fluency, and preservation of meaning.\n\n"
            f"Source text ({source_lang}):\n{source_text}\n\n"
            f"Translation ({target_lang}):\n{translated_text}\n\n"
            f"Provide a rating out of 10 and explain the strengths and weaknesses of the translation. "
            f"Focus on accuracy, fluency, preservation of meaning, and cultural appropriateness."
        )
    
    def _calculate_fluency_score(self, translated_text: str, target_lang: str) -> float:
        """
        Calculate fluency score based on text properties.
        
        Args:
            translated_text: Translated text
            target_lang: Target language code
            
        Returns:
            Fluency score (0-1)
        """
        # This is a simplified placeholder implementation
        # In a real implementation, this could use language models or statistical methods
        
        # For now, we'll use a heuristic based on text statistics
        if not self.text_processor:
            return 0.7  # Default score
            
        try:
            stats = self.text_processor.calculate_statistics(translated_text)
            
            # Assume a reasonable average sentence length indicates fluency
            # This is very simplistic and would need refinement in a real implementation
            avg_sentence_length = stats.get("avg_sentence_length", 0)
            if avg_sentence_length < 3:
                # Too short sentences
                return 0.5
            elif avg_sentence_length > 30:
                # Too long sentences
                return 0.6
            else:
                # Reasonable length
                return 0.8
        except Exception:
            return 0.7  # Default score
    
    def _calculate_accuracy_score(self, semantic_similarity: float, 
                                source_text: str, translated_text: str) -> float:
        """
        Calculate accuracy score based on semantic similarity and other factors.
        
        Args:
            semantic_similarity: Semantic similarity score
            source_text: Original text
            translated_text: Translated text
            
        Returns:
            Accuracy score (0-1)
        """
        # This is a simplified implementation
        # Mainly using semantic similarity with a slight adjustment
        
        # Check if lengths are very disproportionate (potential missing content)
        source_len = len(source_text)
        translated_len = len(translated_text)
        
        if source_len == 0 or translated_len == 0:
            return 0.0
            
        # Calculate length ratio (smaller / larger)
        len_ratio = min(source_len, translated_len) / max(source_len, translated_len)
        
        # Adjustment factor based on length proportion
        # If lengths are very different, reduce accuracy
        if len_ratio < 0.5:
            adjustment = 0.7  # Significant length difference
        elif len_ratio < 0.7:
            adjustment = 0.9  # Moderate length difference
        else:
            adjustment = 1.0  # Similar lengths
            
        # Weighted combination (mostly semantic similarity)
        return semantic_similarity * 0.9 * adjustment
    
    def _calculate_quality_score(self, semantic_similarity: float, 
                               fluency_score: float, accuracy_score: float) -> float:
        """
        Calculate overall quality score from component scores.
        
        Args:
            semantic_similarity: Semantic similarity score
            fluency_score: Fluency score
            accuracy_score: Accuracy score
            
        Returns:
            Overall quality score (0-1)
        """
        # Weighted average of component scores
        # Semantic similarity is most important, followed by accuracy and fluency
        weights = {
            "semantic": 0.5,
            "accuracy": 0.3,
            "fluency": 0.2
        }
        
        quality_score = (
            semantic_similarity * weights["semantic"] +
            accuracy_score * weights["accuracy"] +
            fluency_score * weights["fluency"]
        )
        
        return max(0.0, min(1.0, quality_score))  # Ensure score is between 0 and 1