import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Callable
import logging
from enum import Enum
from functools import lru_cache
import time

logger = logging.getLogger("tqa.similarity")

class SimilarityMetric(Enum):
    """Supported similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean" 
    DOT = "dot"
    MANHATTAN = "manhattan"
    ANGULAR = "angular"  # Distance in radians between vectors

class SimilarityCalculator:
    """
    Advanced semantic similarity calculator supporting multiple metrics 
    and optimization strategies for multilingual text comparisons.
    """
    
    def __init__(self, vector_generator, config_manager=None):
        """
        Initialize the similarity calculator.
        
        Args:
            vector_generator: MultilingualVectorGenerator instance
            config_manager: ConfigManager instance for retrieving settings
        """
        self.vector_generator = vector_generator
        self.config = config_manager
        self.logger = logging.getLogger("tqa.similarity.calculator")
        
        # Default threshold for considering texts similar
        self.default_threshold = 0.75 if config_manager is None else config_manager.get("similarity.threshold", 0.75)
        
        # Default metric
        self.default_metric = SimilarityMetric.COSINE
        
        # Performance tracking
        self.calculation_times = []
    
    def calculate_similarity(self,
                           text1: str,
                           text2: str,
                           metric: Union[str, SimilarityMetric] = None,
                           lang1: Optional[str] = None,
                           lang2: Optional[str] = None,
                           precomputed_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric to use (defaults to cosine)
            lang1: Language of first text (auto-detected if None)
            lang2: Language of second text (auto-detected if None)
            precomputed_vectors: Optional tuple of (vec1, vec2) if already computed
            
        Returns:
            Similarity score between 0 and 1
        """
        start_time = time.time()
        
        # Handle metric selection
        if metric is None:
            metric = self.default_metric
        elif isinstance(metric, str):
            try:
                metric = SimilarityMetric(metric.lower())
            except ValueError:
                self.logger.warning(f"Unknown similarity metric: {metric}. Using cosine.")
                metric = SimilarityMetric.COSINE
        
        # Generate vectors if not provided
        if precomputed_vectors is None:
            if lang1 == lang2 or (lang1 is None and lang2 is None):
                # Same language or both auto-detect - can use standard embedding
                vectors = self.vector_generator.generate_vectors(
                    texts=[text1, text2], 
                    language=lang1 or lang2
                )
                vec1, vec2 = vectors[0], vectors[1]
            else:
                # Cross-lingual comparison - use specialized method
                vec1, vec2 = self.vector_generator.generate_cross_lingual_vectors(
                    source_texts=[text1],
                    target_texts=[text2],
                    source_lang=lang1,
                    target_lang=lang2
                )
                vec1, vec2 = vec1[0], vec2[0]
        else:
            vec1, vec2 = precomputed_vectors
            
        # Calculate similarity based on chosen metric
        similarity = self._apply_similarity_metric(vec1, vec2, metric)
        
        # Track performance
        elapsed = time.time() - start_time
        self.calculation_times.append(elapsed)
        if len(self.calculation_times) > 100:
            self.calculation_times.pop(0)  # Keep only last 100 times
            
        avg_time = sum(self.calculation_times) / len(self.calculation_times)
        self.logger.debug(f"Similarity calculation took {elapsed:.4f}s (avg: {avg_time:.4f}s)")
            
        return similarity
    
    def calculate_pairwise_similarity(self,
                                    texts1: List[str],
                                    texts2: List[str],
                                    metric: Union[str, SimilarityMetric] = None,
                                    lang1: Optional[str] = None,
                                    lang2: Optional[str] = None) -> np.ndarray:
        """
        Calculate pairwise similarities between two lists of texts.
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            metric: Similarity metric to use
            lang1: Language of first texts
            lang2: Language of second texts
            
        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
        """
        # Handle metric selection
        if metric is None:
            metric = self.default_metric
        elif isinstance(metric, str):
            try:
                metric = SimilarityMetric(metric.lower())
            except ValueError:
                self.logger.warning(f"Unknown similarity metric: {metric}. Using cosine.")
                metric = SimilarityMetric.COSINE
        
        # Generate vectors for all texts
        if lang1 == lang2 or (lang1 is None and lang2 is None):
            # Same language - can batch together
            all_texts = texts1 + texts2
            vectors = self.vector_generator.generate_vectors(
                texts=all_texts, 
                language=lang1 or lang2
            )
            vectors1 = vectors[:len(texts1)]
            vectors2 = vectors[len(texts1):]
        else:
            # Different languages - use cross-lingual method
            vectors1, vectors2 = self.vector_generator.generate_cross_lingual_vectors(
                source_texts=texts1,
                target_texts=texts2,
                source_lang=lang1,
                target_lang=lang2
            )
        
        # Calculate similarity matrix
        return self._calculate_similarity_matrix(vectors1, vectors2, metric)
    
    def find_most_similar(self,
                        query_text: str,
                        candidate_texts: List[str],
                        metric: Union[str, SimilarityMetric] = None,
                        query_lang: Optional[str] = None,
                        candidates_lang: Optional[str] = None,
                        threshold: Optional[float] = None,
                        top_k: Optional[int] = None) -> List[Dict]:
        """
        Find most similar texts to a query text.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts to compare against
            metric: Similarity metric to use
            query_lang: Language of query text
            candidates_lang: Language of candidate texts
            threshold: Minimum similarity threshold (0-1)
            top_k: Return only top k results
            
        Returns:
            List of dicts with {'index', 'text', 'similarity'} sorted by similarity
        """
        # Set default threshold if not specified
        if threshold is None:
            threshold = self.default_threshold
        
        # Handle empty candidates
        if not candidate_texts:
            return []
        
        # Calculate similarities
        if len(candidate_texts) == 1:
            # Optimize for single candidate case
            similarity = self.calculate_similarity(
                query_text, 
                candidate_texts[0], 
                metric=metric, 
                lang1=query_lang, 
                lang2=candidates_lang
            )
            similarities = np.array([similarity])
        else:
            # Calculate all similarities
            similarities = self.calculate_pairwise_similarity(
                [query_text], 
                candidate_texts, 
                metric=metric, 
                lang1=query_lang, 
                lang2=candidates_lang
            )[0]
        
        # Create result objects with index, text, and similarity
        results = [
            {'index': i, 'text': text, 'similarity': float(sim)}
            for i, (text, sim) in enumerate(zip(candidate_texts, similarities))
            if sim >= threshold
        ]
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit to top k if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]
            
        return results
    
    def classify_semantic_match(self,
                              text1: str,
                              text2: str,
                              thresholds: Optional[Dict[str, float]] = None,
                              metric: Union[str, SimilarityMetric] = None,
                              lang1: Optional[str] = None,
                              lang2: Optional[str] = None) -> str:
        """
        Classify the semantic relationship between two texts.
        
        Args:
            text1: First text
            text2: Second text
            thresholds: Custom thresholds for classification categories
            metric: Similarity metric
            lang1: Language of first text
            lang2: Language of second text
            
        Returns:
            Classification as string: "exact", "high", "moderate", "low", or "unrelated"
        """
        # Default thresholds
        default_thresholds = {
            "exact": 0.95,     # Nearly identical meaning
            "high": 0.80,      # Very similar meaning
            "moderate": 0.60,  # Related topics
            "low": 0.40        # Slightly related
            # Below "low" threshold is "unrelated"
        }
        
        # Use provided thresholds or defaults
        thresholds = thresholds or default_thresholds
        
        # Calculate similarity
        similarity = self.calculate_similarity(
            text1, text2, metric=metric, lang1=lang1, lang2=lang2
        )
        
        # Classify based on thresholds
        if similarity >= thresholds["exact"]:
            return "exact"
        elif similarity >= thresholds["high"]:
            return "high"
        elif similarity >= thresholds["moderate"]:
            return "moderate"
        elif similarity >= thresholds["low"]:
            return "low"
        else:
            return "unrelated"
    
    def _apply_similarity_metric(self,
                               vec1: np.ndarray,
                               vec2: np.ndarray,
                               metric: SimilarityMetric) -> float:
        """
        Apply the selected similarity metric to two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric to use
            
        Returns:
            Similarity score (0-1 range)
        """
        # Handle empty or invalid vectors
        if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Ensure vectors are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Apply the appropriate metric
        if metric == SimilarityMetric.COSINE:
            # Cosine similarity: dot(v1, v2) / (norm(v1) * norm(v2))
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.dot(vec1, vec2) / (norm1 * norm2)
            
        elif metric == SimilarityMetric.DOT:
            # Normalized dot product (simple but less accurate)
            return np.dot(vec1, vec2) / len(vec1)
            
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(vec1 - vec2)
            # Convert distance to similarity score (0-1)
            return 1.0 / (1.0 + distance)
            
        elif metric == SimilarityMetric.MANHATTAN:
            # Manhattan distance converted to similarity
            distance = np.sum(np.abs(vec1 - vec2))
            # Convert distance to similarity score (0-1)
            return 1.0 / (1.0 + distance)
            
        elif metric == SimilarityMetric.ANGULAR:
            # Angular similarity (cosine similarity converted to angle in radians)
            cos_sim = self._apply_similarity_metric(vec1, vec2, SimilarityMetric.COSINE)
            # Convert to radians and normalize to 0-1 range
            # 0 radians (identical) -> 1.0, π radians (opposite) -> 0.0
            return 1.0 - np.arccos(max(-1.0, min(1.0, cos_sim))) / np.pi
            
        else:
            # Default to cosine similarity
            self.logger.warning(f"Unsupported similarity metric: {metric}. Using cosine.")
            return self._apply_similarity_metric(vec1, vec2, SimilarityMetric.COSINE)
    
    def _calculate_similarity_matrix(self,
                                   vectors1: np.ndarray,
                                   vectors2: np.ndarray,
                                   metric: SimilarityMetric) -> np.ndarray:
        """
        Calculate similarity matrix between two sets of vectors.
        
        Args:
            vectors1: First set of vectors
            vectors2: Second set of vectors
            metric: Similarity metric to use
            
        Returns:
            Similarity matrix of shape (len(vectors1), len(vectors2))
        """
        if len(vectors1) == 0 or len(vectors2) == 0:
            return np.array([[]])
        
        # For cosine similarity, we can optimize with matrix multiplication
        if metric == SimilarityMetric.COSINE:
            # Normalize vectors for cosine similarity
            norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
            
            # Avoid division by zero
            norms1 = np.maximum(norms1, 1e-10)
            norms2 = np.maximum(norms2, 1e-10)
            
            normalized1 = vectors1 / norms1
            normalized2 = vectors2 / norms2
            
            # Compute dot products for all pairs
            return np.dot(normalized1, normalized2.T)
            
        else:
            # For other metrics, compute pairwise similarities
            result = np.zeros((len(vectors1), len(vectors2)))
            
            for i, v1 in enumerate(vectors1):
                for j, v2 in enumerate(vectors2):
                    result[i, j] = self._apply_similarity_metric(v1, v2, metric)
                    
            return result 