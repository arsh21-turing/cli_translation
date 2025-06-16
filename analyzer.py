"""
Translation Analyzer for Smart CLI Translation Quality Analyzer
Handles translation quality assessment and scoring
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
import time
import pycountry

from config_manager import ConfigManager
from model_loader import ModelLoader, MultilingualModelManager
from text_processor import TextProcessor
from embedding_generator import MultilingualEmbeddingGenerator
from embedding_generator import MultilingualVectorGenerator
from similarity_calculator import SimilarityCalculator, SimilarityMetric
from language_utils import EmbeddingBasedLanguageDetector

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

class TranslationQualityAnalyzer:
    """
    Analyzer for translation quality between source and target texts.
    Integrates multilingual models and embedding-based analysis.
    """
    
    def __init__(self, config=None, model_loader=None, multilingual_model_manager=None, text_processor=None):
        """Initialize the analyzer with necessary components."""
        self.config = config or ConfigManager()
        
        # Initialize or use provided components
        self.model_loader = model_loader or ModelLoader(self.config) 
        self.multilingual_model_manager = multilingual_model_manager or MultilingualModelManager(
            self.config, self.model_loader
        )
        self.text_processor = text_processor or TextProcessor()
        
        # Initialize logger
        self.logger = logging.getLogger("tqa.analyzer")
        
    def translate_text(self, text: str, source_lang: Optional[str] = None, 
                     target_lang: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code
            model_name: Specific model name to use (optional)
            
        Returns:
            Translated text
        """
        # Use the combined translate_and_analyze method but only return the translation
        result = self.translate_and_analyze(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model_name,
            analyze_quality=False  # Skip quality analysis for speed
        )
        return result["translation"]
        
    def translate_and_analyze(self, 
                            text: str, 
                            source_lang: Optional[str] = None, 
                            target_lang: Optional[str] = None,
                            model_name: Optional[str] = None,
                            analyze_quality: bool = True) -> Dict:
        """
        Translate text and analyze the quality of the translation in one operation.
        
        Args:
            text: Source text to translate and analyze
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code (required)
            model_name: Optional specific model name to use
            analyze_quality: Whether to analyze translation quality
            
        Returns:
            Dict with translation and analysis results
        """
        if target_lang is None:
            raise ValueError("Target language must be specified for translation")
        
        # Auto-detect source language if not provided
        if source_lang is None:
            detection_result = self.detect_language_advanced(text)
            if isinstance(detection_result, dict):
                source_lang = detection_result['language']
            else:
                source_lang = detection_result
            
        self.logger.info(f"Translating from {source_lang} to {target_lang}")
        
        # Get the appropriate translation model for this language pair
        model = self.multilingual_model_manager.get_model(source_lang, target_lang)
        
        # Translate the text
        translation = self._perform_translation(text, source_lang, target_lang, model, model_name)
        
        # Prepare result dict
        result = {
            "source_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation": translation
        }
        
        # Analyze the translation quality if requested
        if analyze_quality:
            analysis = self._analyze_translation_quality(
                text, translation, source_lang, target_lang, detailed=True
            )
            
            # Add analysis results to the output
            result.update({
                "quality_score": analysis["quality_score"],
                "fluency_score": analysis["fluency_score"],
                "accuracy_score": analysis["accuracy_score"]
            })
            
            # Include detailed analysis if available
            if "detailed_analysis" in analysis:
                result["detailed_analysis"] = analysis["detailed_analysis"]
            
            # Include segment scores if available
            if "segment_scores" in analysis:
                result["segment_scores"] = analysis["segment_scores"]
        
        return result
    
    def _perform_translation(self, 
                           text: str, 
                           source_lang: str,
                           target_lang: str, 
                           model: Any, 
                           model_name: Optional[str] = None) -> str:
        """
        Perform the actual translation with the selected model.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            model: Translation model to use
            model_name: Optional specific model name
            
        Returns:
            Translated text
        """
        import torch
        
        # Handle different model types
        if hasattr(model, 'translate'):
            # Direct translation method
            return model.translate(text, source_lang=source_lang, target_lang=target_lang)
        elif isinstance(model, tuple) and len(model) == 2:
            # Model and tokenizer tuple (common in Hugging Face)
            model_obj, tokenizer = model
            
            # Prepare tokenizer parameters
            if hasattr(tokenizer, 'src_lang') and hasattr(tokenizer, 'tgt_lang'):
                tokenizer.src_lang = source_lang
                tokenizer.tgt_lang = target_lang
                
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model_obj.generate(**inputs)
                
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        else:
            # Fallback to model_loader for API or other model types
            return self.model_loader.translate(
                text, 
                source_lang=source_lang, 
                target_lang=target_lang, 
                model_name=model_name
            )
        
    def _analyze_translation_quality(self, source_text, translated_text, 
                                   source_lang, target_lang, detailed=False):
        """
        Analyze the quality of a translation.
        
        Args:
            source_text (str): Original text
            translated_text (str): Translated text
            source_lang (str): Source language code
            target_lang (str): Target language code
            detailed (bool): Whether to include detailed analysis
            
        Returns:
            dict: Quality analysis results
        """
        # Get embeddings for source and translated text
        source_emb, target_emb = self._get_embeddings(
            source_text, translated_text, source_lang, target_lang
        )
        
        # Calculate semantic similarity
        similarity = self._calculate_similarity(source_emb, target_emb)
        
        # Perform fluency analysis (how natural the translation reads)
        fluency_score = self._analyze_fluency(translated_text, target_lang)
        
        # Derive overall quality score
        # Typically weighted more toward accuracy for translation evaluation
        quality_score = 0.7 * similarity + 0.3 * fluency_score
        
        # Build result dictionary
        result = {
            "quality_score": float(quality_score),
            "accuracy_score": float(similarity),
            "fluency_score": float(fluency_score)
        }
        
        # Add detailed analysis if requested
        if detailed:
            result["detailed_analysis"] = {
                "semantic_similarity": float(similarity),
                "fluency_metrics": self._get_detailed_fluency_metrics(translated_text, target_lang),
                # Add additional metrics as needed
            }
            
        return result
    
    def _get_embeddings(self, source_text, target_text, source_lang, target_lang):
        """
        Get embeddings for source and target text using appropriate models.
        
        Args:
            source_text (str): Source text
            target_text (str): Target/translated text
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            tuple: (source_embedding, target_embedding)
        """
        # Use the multilingual model manager to get appropriate embedding models
        source_model = self.multilingual_model_manager.get_model(source_lang, source_lang)
        target_model = self.multilingual_model_manager.get_model(target_lang, target_lang)
        
        # Generate embeddings
        source_emb = source_model.encode(source_text, convert_to_numpy=True)
        target_emb = target_model.encode(target_text, convert_to_numpy=True)
        
        return source_emb, target_emb
    
    def _calculate_similarity(self, source_emb, target_emb):
        """
        Calculate semantic similarity between source and target embeddings.
        
        Args:
            source_emb (ndarray): Source text embedding
            target_emb (ndarray): Target text embedding
            
        Returns:
            float: Similarity score (0-1)
        """
        # Calculate cosine similarity
        norm_source = np.linalg.norm(source_emb)
        norm_target = np.linalg.norm(target_emb)
        
        if norm_source == 0 or norm_target == 0:
            return 0.0
            
        similarity = np.dot(source_emb, target_emb) / (norm_source * norm_target)
        return float(similarity)
    
    def _analyze_fluency(self, text, language):
        """
        Analyze the fluency/naturalness of text in a given language.
        
        Args:
            text (str): Text to analyze
            language (str): Language code
            
        Returns:
            float: Fluency score (0-1)
        """
        # This is a placeholder for actual fluency analysis
        # In a real implementation, this would use language models or rule-based analysis
        
        # Simple length-based heuristic as fallback
        words = text.split()
        if not words:
            return 0.0
            
        # For testing, return a simple score based on text length
        # Real implementation would be more sophisticated
        return min(0.95, 0.5 + len(words) / 100)
    
    def _get_detailed_fluency_metrics(self, text, language):
        """
        Get detailed fluency metrics for text.
        
        Args:
            text (str): Text to analyze
            language (str): Language code
            
        Returns:
            dict: Detailed fluency metrics
        """
        # This is a placeholder for detailed fluency analysis
        # Real implementation would include multiple metrics
        
        return {
            "avg_word_length": sum(len(word) for word in text.split()) / max(1, len(text.split())),
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            # Add more metrics as needed
        }
    
    def rate_translation(self, source_text, translation, source_lang=None, target_lang=None):
        """
        Rate the quality of an existing translation.
        
        Args:
            source_text (str): Original text
            translation (str): Translated text
            source_lang (str): Source language code (auto-detected if None)
            target_lang (str): Target language code (auto-detected if None)
            
        Returns:
            dict: Rating results
        """
        # Auto-detect languages if not provided
        if not source_lang:
            detection = self.detect_language_advanced(source_text)
            if isinstance(detection, dict):
                source_lang = detection['language']
            else:
                source_lang = detection
                
        if not target_lang:
            detection = self.detect_language_advanced(translation)
            if isinstance(detection, dict):
                target_lang = detection['language']
            else:
                target_lang = detection
        
        # Analyze the translation quality
        analysis = self._analyze_translation_quality(
            source_text, translation, source_lang, target_lang, detailed=True
        )
        
        # Add translation text and language info
        rating = {
            "score": analysis["quality_score"],
            "accuracy": analysis["accuracy_score"],
            "fluency": analysis["fluency_score"],
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        
        # Add qualitative feedback based on scores
        if rating["score"] >= 0.9:
            rating["feedback"] = "Excellent translation with high accuracy and natural phrasing."
        elif rating["score"] >= 0.8:
            rating["feedback"] = "Very good translation with minor inaccuracies or phrasing issues."
        elif rating["score"] >= 0.7:
            rating["feedback"] = "Good translation that conveys the main meaning, but has some issues."
        elif rating["score"] >= 0.5:
            rating["feedback"] = "Fair translation with significant room for improvement."
        else:
            rating["feedback"] = "Poor translation with major accuracy or fluency problems."
        
        return rating

    def analyze_cross_lingual_similarity(self, 
                                        source_texts: Union[str, List[str]],
                                        target_texts: Union[str, List[str]],
                                        source_lang: Optional[str] = None, 
                                        target_lang: Optional[str] = None,
                                        similarity_metric: str = "cosine",
                                        preprocessing_level: str = "standard",
                                        detailed: bool = False) -> Dict:
        """
        Analyze the similarity of source and target texts across different languages.
        
        Args:
            source_texts: Source language text(s)
            target_texts: Target language text(s) (translations)
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code (auto-detected if None)
            similarity_metric: Similarity calculation method ("cosine", "euclidean", "dot")
            preprocessing_level: Text preprocessing intensity ("minimal", "standard", "aggressive")
            detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary with analysis results
        """
        # Convert single strings to lists
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        if isinstance(target_texts, str):
            target_texts = [target_texts]
        
        # Auto-detect languages if not provided
        if source_lang is None or target_lang is None:
            lang_detector = self.model_loader.language_detector
            
            if source_lang is None and source_texts:
                source_lang_info = lang_detector.detect(source_texts[0])
                source_lang = source_lang_info['code']
                self.logger.info(f"Detected source language: {source_lang_info['name']} ({source_lang})")
                
            if target_lang is None and target_texts:
                target_lang_info = lang_detector.detect(target_texts[0])
                target_lang = target_lang_info['code']
                self.logger.info(f"Detected target language: {target_lang_info['name']} ({target_lang})")
        
        # Generate cross-lingual embeddings
        self.logger.info(f"Generating cross-lingual embeddings for {source_lang}-{target_lang} pair")
        
        # Get our vector generator
        vector_generator = MultilingualVectorGenerator(
            self.model_loader, self.text_processor, self.config
        )
        
        # Generate embeddings
        source_vectors, target_vectors = vector_generator.generate_cross_lingual_vectors(
            source_texts=source_texts,
            target_texts=target_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            preprocessing_level=preprocessing_level
        )
        
        # Calculate similarity matrix
        similarity_matrix = vector_generator.calculate_similarity_matrix(
            source_vectors=source_vectors,
            target_vectors=target_vectors,
            metric=similarity_metric
        )
        
        # Basic similarity scores (diagonal of the matrix if same length, or max otherwise)
        if len(source_texts) == len(target_texts):
            # Direct alignment
            similarity_scores = np.diag(similarity_matrix).tolist()
            avg_similarity = np.mean(similarity_scores)
        else:
            # Find best matches
            max_similarities = np.max(similarity_matrix, axis=1).tolist()
            similarity_scores = max_similarities
            avg_similarity = np.mean(max_similarities)
        
        # Prepare results
        results = {
            "source_language": source_lang,
            "target_language": target_lang,
            "average_similarity": float(avg_similarity),
            "similarity_scores": similarity_scores,
            "metric": similarity_metric,
        }
        
        # Add detailed analysis if requested
        if detailed:
            # Find best matches in both directions
            source_to_target = np.argmax(similarity_matrix, axis=1).tolist()
            target_to_source = np.argmax(similarity_matrix, axis=0).tolist()
            
            # Identify mutual best matches (high confidence alignments)
            mutual_matches = []
            for src_idx, tgt_idx in enumerate(source_to_target):
                if target_to_source[tgt_idx] == src_idx:
                    mutual_matches.append((src_idx, tgt_idx, float(similarity_matrix[src_idx, tgt_idx])))
            
            # Calculate perplexity-like measure of alignment confidence
            if similarity_matrix.shape[0] > 1 and similarity_matrix.shape[1] > 1:
                # Softmax over rows and columns
                row_softmax = np.exp(similarity_matrix) / np.sum(np.exp(similarity_matrix), axis=1, keepdims=True)
                col_softmax = np.exp(similarity_matrix.T) / np.sum(np.exp(similarity_matrix.T), axis=1, keepdims=True)
                
                # Calculate entropy
                row_entropy = -np.sum(row_softmax * np.log(np.maximum(row_softmax, 1e-10)), axis=1)
                col_entropy = -np.sum(col_softmax * np.log(np.maximum(col_softmax, 1e-10)), axis=1)
                
                # Lower entropy means more confident alignment
                alignment_confidence = 1.0 - np.mean(np.concatenate([row_entropy, col_entropy])) / np.log(max(similarity_matrix.shape))
            else:
                alignment_confidence = float(avg_similarity)
            
            results.update({
                "alignment_confidence": float(alignment_confidence),
                "similarity_matrix": similarity_matrix.tolist(),
                "source_to_target_mapping": source_to_target,
                "target_to_source_mapping": target_to_source,
                "mutual_best_matches": mutual_matches,
            })
        
        return results

    def analyze_semantic_similarity(self,
                                  source_text: Union[str, List[str]],
                                  target_text: Union[str, List[str]],
                                  metric: str = "cosine",
                                  source_lang: Optional[str] = None,
                                  target_lang: Optional[str] = None,
                                  segmented: bool = False,
                                  classification: bool = False) -> Dict:
        """
        Analyze semantic similarity between source and target texts.
        
        Args:
            source_text: Source text or list of text segments
            target_text: Target text or list of text segments
            metric: Similarity metric to use
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code (auto-detected if None)
            segmented: Whether to treat inputs as individual segments
                       If False, treats inputs as full documents
            classification: Whether to include semantic match classification
            
        Returns:
            Dictionary with similarity analysis results
        """
        # Start timing
        start_time = time.time()
        
        # Handle single string inputs
        if isinstance(source_text, str) and not segmented:
            source_segments = self.text_processor.split_into_segments(source_text)
        elif isinstance(source_text, str) and segmented:
            source_segments = [source_text]
        else:
            source_segments = source_text
        
        if isinstance(target_text, str) and not segmented:
            target_segments = self.text_processor.split_into_segments(target_text)
        elif isinstance(target_text, str) and segmented:
            target_segments = [target_text]
        else:
            target_segments = target_text
            
        # Detect languages if not provided
        if source_lang is None and source_segments:
            source_lang_info = self.model_loader.language_detector.detect(source_segments[0])
            source_lang = source_lang_info['code']
            self.logger.info(f"Detected source language: {source_lang_info['name']} ({source_lang})")
            
        if target_lang is None and target_segments:
            target_lang_info = self.model_loader.language_detector.detect(target_segments[0])
            target_lang = target_lang_info['name']
            self.logger.info(f"Detected target language: {target_lang_info['name']} ({target_lang})")
            
        # Calculate segment-by-segment similarity if there are multiple segments
        if len(source_segments) > 1 or len(target_segments) > 1:
            self.logger.info(f"Calculating segment-by-segment similarity for {len(source_segments)} " 
                            f"source and {len(target_segments)} target segments")
            
            # Calculate similarity matrix
            similarity_matrix = self.similarity_calculator.calculate_pairwise_similarity(
                source_segments,
                target_segments,
                metric=metric,
                lang1=source_lang,
                lang2=target_lang
            )
            
            # For aligned segments (when counts match), use diagonal
            if len(source_segments) == len(target_segments):
                segment_similarities = np.diag(similarity_matrix).tolist()
                avg_similarity = float(np.mean(segment_similarities))
            else:
                # For unaligned segments, use best match for each source segment
                segment_similarities = np.max(similarity_matrix, axis=1).tolist()
                avg_similarity = float(np.mean(segment_similarities))
                
            # Find best matches for each source segment
            best_matches = []
            for i in range(len(source_segments)):
                best_idx = np.argmax(similarity_matrix[i])
                best_matches.append({
                    "source_segment": source_segments[i],
                    "target_segment": target_segments[best_idx],
                    "similarity": float(similarity_matrix[i, best_idx]),
                    "target_index": int(best_idx)
                })
                
            # Create detailed segment analysis
            segment_analysis = []
            for i, src in enumerate(source_segments):
                if len(source_segments) == len(target_segments):
                    # Aligned segments
                    tgt = target_segments[i]
                    sim = segment_similarities[i]
                    segment_analysis.append({
                        "source": src,
                        "target": tgt,
                        "similarity": float(sim),
                        "aligned": True
                    })
                else:
                    # Best matching segment
                    best_idx = np.argmax(similarity_matrix[i])
                    tgt = target_segments[best_idx]
                    sim = similarity_matrix[i, best_idx]
                    segment_analysis.append({
                        "source": src,
                        "target": tgt,
                        "similarity": float(sim),
                        "aligned": False,
                        "target_index": int(best_idx)
                    })
                    
                # Add classification if requested
                if classification:
                    match_class = self.similarity_calculator.classify_semantic_match(
                        src, tgt, lang1=source_lang, lang2=target_lang, metric=metric
                    )
                    segment_analysis[-1]["match_class"] = match_class
        else:
            # Single segment comparison
            self.logger.info("Calculating similarity for single source-target pair")
            
            # Use the direct calculation
            similarity = self.similarity_calculator.calculate_similarity(
                source_segments[0],
                target_segments[0],
                metric=metric,
                lang1=source_lang,
                lang2=target_lang
            )
            
            segment_similarities = [float(similarity)]
            avg_similarity = float(similarity)
            
            segment_analysis = [{
                "source": source_segments[0],
                "target": target_segments[0],
                "similarity": float(similarity),
                "aligned": True
            }]
            
            if classification:
                match_class = self.similarity_calculator.classify_semantic_match(
                    source_segments[0], target_segments[0], 
                    lang1=source_lang, lang2=target_lang, metric=metric
                )
                segment_analysis[0]["match_class"] = match_class
        
        # Prepare results
        results = {
            "metric": metric,
            "source_language": source_lang,
            "target_language": target_lang,
            "average_similarity": avg_similarity,
            "segment_similarities": segment_similarities,
            "segment_analysis": segment_analysis
        }
        
        # Add overall classification if requested
        if classification:
            overall_match = self.similarity_calculator.classify_semantic_match(
                " ".join(source_segments) if len(source_segments) > 1 else source_segments[0],
                " ".join(target_segments) if len(target_segments) > 1 else target_segments[0],
                lang1=source_lang, 
                lang2=target_lang,
                metric=metric
            )
            results["overall_match_class"] = overall_match
            
        # Add timing information
        results["analysis_time"] = time.time() - start_time
        
        return results

    def detect_language_advanced(self, text: str, fast_mode: bool = True, 
                              detailed: bool = False, min_length: int = 30) -> Union[str, Dict]:
        """
        Detect language using the advanced embedding-based approach.
        
        Args:
            text: Text to analyze
            fast_mode: Use faster detection with script-based filtering
            detailed: Whether to return detailed analysis
            min_length: Minimum text length for reliable detection
            
        Returns:
            Language code or detailed dictionary
        """
        # Check if we have the embedding detector initialized
        if not hasattr(self, '_embedding_detector'):
            self._embedding_detector = EmbeddingBasedLanguageDetector(
                self.model_loader, self.config
            )
            # Generate reference data if it doesn't exist
            if not self._embedding_detector.reference_centroids:
                self.logger.info("Initializing language reference data...")
                # Use a subset of languages for faster initialization
                self._embedding_detector.generate_reference_data(
                    languages=EmbeddingBasedLanguageDetector.COMMON_LANGUAGES[:10],  # Start with 10 most common
                    samples_per_lang=20,  # Smaller set for demonstration
                    use_external_sources=False  # Don't fetch external data by default
                )
        
        # Detect language
        result = self._embedding_detector.detect_language(
            text, min_length=min_length, fast_mode=fast_mode, return_scores=detailed
        )
        
        if detailed:
            # Add language name
            try:
                lang = pycountry.languages.get(alpha_2=result['language'])
                result['language_name'] = lang.name if lang else "Unknown"
            except (AttributeError, KeyError):
                result['language_name'] = "Unknown"
            
            # Add script information
            result['scripts'] = self._embedding_detector.detect_script(text)
            
            # Detect if text is multilingual
            if len(text) > 500:  # Only check for longer texts
                lang_composition = self._embedding_detector.detect_language_composition(text)
                is_multilingual = len([l for l, p in lang_composition.items() if p > 0.2]) > 1
                result['is_multilingual'] = is_multilingual
                
                if is_multilingual:
                    result['language_composition'] = lang_composition
            
            return result
        else:
            return result

    def analyze_text_composition(self, text: str) -> Dict:
        """
        Analyze the linguistic composition of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Ensure we have the embedding detector initialized
        if not hasattr(self, '_embedding_detector'):
            self._embedding_detector = EmbeddingBasedLanguageDetector(
                self.model_loader, self.config
            )
        
        # Detect language mix
        language_mix = self._embedding_detector.detect_language_mix(text)
        
        # Calculate language composition
        composition = self._embedding_detector.detect_language_composition(text)
        
        # Create segment-colored text representation (for display)
        colored_segments = []
        for idx, info in language_mix.items():
            lang = info['language']
            segment = info['text']
            colored_segments.append({
                'text': segment,
                'language': lang,
                'confidence': info.get('confidence', 0.0)
            })
        
        # Get additional metrics
        total_chars = len(text)
        language_count = sum(1 for p in composition.values() if p > 0.05)
        primary_language = max(composition.items(), key=lambda x: x[1])[0] if composition else 'und'
        
        # Prepare results
        results = {
            'primary_language': primary_language,
            'language_count': language_count,
            'is_multilingual': language_count > 1,
            'composition': composition,
            'segments': colored_segments,
            'total_chars': total_chars
        }
        
        return results