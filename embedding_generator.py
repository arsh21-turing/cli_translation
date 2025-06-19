from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import os
import json
import hashlib
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time

from language_utils import LanguageDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualEmbeddingGenerator:
    """Specialized embedding generator for multilingual text."""
    
    def __init__(self, config_manager, multilingual_model_manager, text_processor):
        """
        Initialize with required dependencies.
        
        Args:
            config_manager: Config manager instance
            multilingual_model_manager: Multilingual model manager instance
            text_processor: Text processor for preprocessing
        """
        self.config = config_manager
        self.model_manager = multilingual_model_manager
        self.text_processor = text_processor
        self.language_detector = LanguageDetector()
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.get('embedding_cache_dir', '~/.tqa/embedding_cache'))
        self.cache_dir = self.cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_embeddings(self, 
                           text: Union[str, List[str]], 
                           lang: Optional[str] = None, 
                           use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for text with language awareness.
        
        Args:
            text (str or List[str]): Text to embed
            lang (str, optional): Language code if known
            use_cache (bool): Whether to use embedding cache
            
        Returns:
            np.ndarray: Embeddings for the text
        """
        # Handle empty input
        if not text:
            return np.array([])
            
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
        
        # Detect language if not provided
        if not lang:
            # Use first paragraph for detection (efficiency)
            sample_text = text[0][:1000] if text else ""
            lang_info = self.language_detector.detect(sample_text)
            lang = lang_info['code']
            logger.info(f"Detected language: {lang_info['name']} ({lang}) with confidence {lang_info['confidence']:.2f}")
        
        # Check cache if enabled
        if use_cache:
            cached_embeddings = self._check_cache(text, lang)
            if cached_embeddings is not None:
                return cached_embeddings
        
        # Get the appropriate model for this language
        model = self.model_manager.get_model(lang, lang)  # Same lang for source and target
        
        # Generate embeddings
        try:
            embeddings = model.encode(text, show_progress_bar=len(text) > 10)
            
            # Update cache
            if use_cache:
                self._update_cache(text, embeddings, lang)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(text), 384))  # Standard embedding size
    
    def generate_cross_lingual_embeddings(self,
                                         source_text: Union[str, List[str]],
                                         target_text: Union[str, List[str]],
                                         source_lang: Optional[str] = None,
                                         target_lang: Optional[str] = None,
                                         use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings optimized for cross-lingual comparison.
        
        Args:
            source_text (str or List[str]): Source language text
            target_text (str or List[str]): Target language text
            source_lang (str, optional): Source language code
            target_lang (str, optional): Target language code
            use_cache (bool): Whether to use embedding cache
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Source and target embeddings
        """
        # Convert single strings to lists
        if isinstance(source_text, str):
            source_text = [source_text]
        if isinstance(target_text, str):
            target_text = [target_text]
            
        # Detect languages if not provided
        if not source_lang:
            source_sample = source_text[0][:1000] if source_text else ""
            source_lang_info = self.language_detector.detect(source_sample)
            source_lang = source_lang_info['code']
            logger.info(f"Detected source language: {source_lang_info['name']} ({source_lang}) with confidence {source_lang_info['confidence']:.2f}")
            
        if not target_lang:
            target_sample = target_text[0][:1000] if target_text else ""
            target_lang_info = self.language_detector.detect(target_sample)
            target_lang = target_lang_info['code']
            logger.info(f"Detected target language: {target_lang_info['name']} ({target_lang}) with confidence {target_lang_info['confidence']:.2f}")
        
        # Get model optimized for this language pair
        model = self.model_manager.get_model(source_lang, target_lang)
        
        # Generate embeddings with the optimal model for this language pair
        try:
            # Process source text
            source_embeddings = None
            if use_cache:
                source_embeddings = self._check_cache(source_text, source_lang, target_lang)
                
            if source_embeddings is None:
                source_embeddings = model.encode(source_text, show_progress_bar=len(source_text) > 10)
                if use_cache:
                    self._update_cache(source_text, source_embeddings, source_lang, target_lang)
            
            # Process target text
            target_embeddings = None
            if use_cache:
                target_embeddings = self._check_cache(target_text, target_lang, source_lang)
                
            if target_embeddings is None:
                target_embeddings = model.encode(target_text, show_progress_bar=len(target_text) > 10)
                if use_cache:
                    self._update_cache(target_text, target_embeddings, target_lang, source_lang)
                    
            return source_embeddings, target_embeddings
            
        except Exception as e:
            logger.error(f"Error generating cross-lingual embeddings: {e}")
            # Return zero embeddings as fallback
            dim = 384  # Standard embedding dimension
            return np.zeros((len(source_text), dim)), np.zeros((len(target_text), dim))
    
    def calculate_similarity(self, 
                           source_embedding: np.ndarray, 
                           target_embedding: np.ndarray,
                           method: str = 'cosine') -> float:
        """
        Calculate similarity between embeddings.
        
        Args:
            source_embedding (np.ndarray): Source embedding
            target_embedding (np.ndarray): Target embedding
            method (str): Similarity method ('cosine', 'euclidean', 'dot')
            
        Returns:
            float: Similarity score
        """
        if method == 'cosine':
            # Normalize vectors
            s_norm = np.linalg.norm(source_embedding)
            t_norm = np.linalg.norm(target_embedding)
            
            # Prevent division by zero
            if s_norm == 0 or t_norm == 0:
                return 0.0
                
            return np.dot(source_embedding, target_embedding) / (s_norm * t_norm)
            
        elif method == 'euclidean':
            # Euclidean distance (convert to similarity)
            dist = np.linalg.norm(source_embedding - target_embedding)
            return 1.0 / (1.0 + dist)
            
        elif method == 'dot':
            # Simple dot product
            return np.dot(source_embedding, target_embedding)
            
        else:
            logger.warning(f"Unknown similarity method: {method}. Using cosine.")
            # Default to cosine
            return self.calculate_similarity(source_embedding, target_embedding, 'cosine')
    
    def _generate_cache_key(self, 
                           text: List[str], 
                           source_lang: str, 
                           target_lang: Optional[str] = None) -> str:
        """Generate a unique cache key for text and language pair."""
        # Create hash from text content
        text_str = "||".join(text[:5])  # Use first 5 elements for efficiency
        text_hash = hashlib.md5(text_str.encode()).hexdigest()
        
        # Create a key using the text hash and language info
        if target_lang:
            return f"{text_hash}_{source_lang}_{target_lang}"
        else:
            return f"{text_hash}_{source_lang}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get filesystem path for a cache key."""
        return self.cache_dir / f"{cache_key}.npy"
    
    def _check_cache(self, 
                    text: List[str], 
                    source_lang: str, 
                    target_lang: Optional[str] = None) -> Optional[np.ndarray]:
        """Check if embeddings are in cache and return if found."""
        cache_key = self._generate_cache_key(text, source_lang, target_lang)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {e}")
                return None
        return None
    
    def _update_cache(self, 
                     text: List[str], 
                     embeddings: np.ndarray, 
                     source_lang: str, 
                     target_lang: Optional[str] = None) -> None:
        """Store embeddings in cache."""
        cache_key = self._generate_cache_key(text, source_lang, target_lang)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            np.save(cache_path, embeddings)
        except Exception as e:
            logger.warning(f"Error caching embeddings: {e}")

class MultilingualVectorGenerator:
    """
    Advanced vector representation generator for multilingual text.
    Provides sophisticated embedding generation with language-specific optimizations.
    """
    
    def __init__(self, multilingual_model_manager, text_processor, config_manager):
        """
        Initialize the multilingual vector generator.
        
        Args:
            multilingual_model_manager: Manager for multilingual models
            text_processor: Text processor for preprocessing
            config_manager: Configuration manager
        """
        self.model_manager = multilingual_model_manager
        self.text_processor = text_processor
        self.config = config_manager
        self.logger = logging.getLogger("tqa.embeddings.vectors")
        
        # Get embedding dimension from config or use default
        self.embedding_dim = self.config.get("models.embedding.dimension", 384)
        
        # Configure caching
        self.use_cache = self.config.get("models.embedding.use_cache", True)
        self.cache_dir = Path(self.config.get("models.embedding.cache_dir", "~/.tqa/embedding_cache"))
        self.cache_dir = self.cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_vectors(self,
                        texts: Union[str, List[str]],
                        language: Optional[str] = None,
                        batch_size: Optional[int] = None,
                        preprocessing_level: str = "standard",
                        pooling_strategy: str = "mean",
                        use_cache: Optional[bool] = None,
                        progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """
        Generate vector representations for texts with advanced options.
        
        Args:
            texts: Single text string or list of texts
            language: ISO language code (auto-detected if None)
            batch_size: Number of texts to process at once (None for auto)
            preprocessing_level: Level of text preprocessing ("minimal", "standard", or "aggressive")
            pooling_strategy: How to pool token embeddings ("mean", "max", "cls", or "weighted")
            use_cache: Whether to use cache (overrides class setting if provided)
            progress_callback: Optional callback function for progress updates (receives current, total)
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Skip empty inputs
        if not texts:
            self.logger.warning("Empty input provided to generate_vectors")
            return np.array([])
        
        # Determine whether to use cache
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Auto-detect language if not provided
        detected_language = None
        if language is None:
            # Use the first 1000 characters of the first text for detection
            sample_text = texts[0][:1000] if texts else ""
            if sample_text:
                lang_info = self.model_manager.detect_language(sample_text)
                detected_language = lang_info['code']
                self.logger.info(f"Auto-detected language: {lang_info['name']} ({detected_language}) "
                                f"with {lang_info['confidence']:.2f} confidence")
                language = detected_language
            else:
                language = 'en'  # Default to English for empty text
                self.logger.warning("Could not detect language, defaulting to English")
        
        # Determine optimal batch size if not specified
        if batch_size is None:
            if len(texts) <= 4:
                batch_size = len(texts)  # Process all at once for very small inputs
            elif len(texts) <= 100:
                batch_size = 16  # Small batch for interactive use
            else:
                batch_size = 32  # Larger batch for bulk processing
        
        # Preprocess texts based on language and preprocessing level
        preprocessed_texts = self._preprocess_texts(texts, language, preprocessing_level)
        
        # Initialize result array
        result = np.zeros((len(texts), self.embedding_dim))
        
        # Process in batches
        batches = [preprocessed_texts[i:i+batch_size] for i in range(0, len(preprocessed_texts), batch_size)]
        
        # Setup progress tracking
        total_batches = len(batches)
        processed_count = 0
        
        # Get appropriate model for the language
        model = self.model_manager.get_model(source_lang=language, target_lang=language)
        
        # Process each batch
        for i, batch in enumerate(batches):
            batch_start_idx = i * batch_size
            
            # Check cache for each text in batch
            if should_use_cache:
                cached_embeddings, missing_indices, missing_texts = self._check_cache_batch(
                    batch, language, preprocessing_level, pooling_strategy)
                
                # Place cached embeddings
                for cached_idx, embedding in cached_embeddings.items():
                    result_idx = batch_start_idx + cached_idx
                    result[result_idx] = embedding
                
                # Only encode texts not found in cache
                batch_to_encode = missing_texts
            else:
                missing_indices = list(range(len(batch)))
                batch_to_encode = batch
            
            # If we have texts to encode
            if batch_to_encode:
                try:
                    # Generate embeddings
                    with torch.no_grad():
                        batch_embeddings = model.encode(
                            batch_to_encode,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                    
                    # Apply pooling strategy if needed
                    if pooling_strategy != "mean":  # Mean is default in SentenceTransformer
                        batch_embeddings = self._apply_pooling(batch_embeddings, pooling_strategy)
                    
                    # Place embeddings in result array
                    for idx, missing_idx in enumerate(missing_indices):
                        result_idx = batch_start_idx + missing_idx
                        result[result_idx] = batch_embeddings[idx]
                    
                    # Update cache
                    if should_use_cache:
                        self._update_cache_batch(batch_to_encode, batch_embeddings, language, 
                                              preprocessing_level, pooling_strategy)
                except Exception as e:
                    self.logger.error(f"Error generating embeddings for batch {i+1}/{total_batches}: {e}")
                    # Leave zeros for failed embeddings
            
            # Update progress
            processed_count += len(batch)
            if progress_callback:
                progress_callback(processed_count, len(texts))
        
        # Log statistics
        elapsed = time.time() - start_time
        self.logger.info(f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
                        f"({len(texts)/elapsed:.2f} texts/s)")
        if should_use_cache:
            self.logger.debug(f"Cache stats: {self.cache_hits} hits, {self.cache_misses} misses "
                            f"({self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}% hit rate)")
        
        return result
    
    def generate_cross_lingual_vectors(self,
                                     source_texts: Union[str, List[str]],
                                     target_texts: Union[str, List[str]],
                                     source_lang: Optional[str] = None,
                                     target_lang: Optional[str] = None,
                                     batch_size: Optional[int] = None,
                                     preprocessing_level: str = "standard",
                                     use_cache: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate aligned vector representations for cross-lingual comparison.
        
        Args:
            source_texts: Source language text(s)
            target_texts: Target language text(s)
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code (auto-detected if None)
            batch_size: Batch size for processing (None for auto)
            preprocessing_level: Preprocessing intensity
            use_cache: Whether to use cache
            
        Returns:
            Tuple of source and target embeddings
        """
        # Start timing
        start_time = time.time()
        
        # Handle single text inputs
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        if isinstance(target_texts, str):
            target_texts = [target_texts]
        
        # Auto-detect languages if not provided
        if source_lang is None:
            sample = source_texts[0][:1000] if source_texts else ""
            if sample:
                lang_info = self.model_manager.detect_language(sample)
                source_lang = lang_info['code']
                self.logger.info(f"Auto-detected source language: {lang_info['name']} ({source_lang})")
            else:
                source_lang = 'en'
                
        if target_lang is None:
            sample = target_texts[0][:1000] if target_texts else ""
            if sample:
                lang_info = self.model_manager.detect_language(sample)
                target_lang = lang_info['code']
                self.logger.info(f"Auto-detected target language: {lang_info['name']} ({target_lang})")
            else:
                target_lang = 'en'
        
        # Get the optimal model for this language pair
        model = self.model_manager.get_model(source_lang=source_lang, target_lang=target_lang)
        
        # Process source texts
        source_vectors = self.generate_vectors(
            source_texts,
            language=source_lang,
            batch_size=batch_size,
            preprocessing_level=preprocessing_level,
            use_cache=use_cache
        )
        
        # Process target texts
        target_vectors = self.generate_vectors(
            target_texts,
            language=target_lang,
            batch_size=batch_size,
            preprocessing_level=preprocessing_level,
            use_cache=use_cache
        )
        
        elapsed = time.time() - start_time
        self.logger.info(f"Generated {len(source_texts)} source and {len(target_texts)} target "
                        f"cross-lingual embeddings in {elapsed:.2f}s")
        
        return source_vectors, target_vectors
    
    def calculate_similarity_matrix(self,
                                  source_vectors: np.ndarray,
                                  target_vectors: np.ndarray,
                                  metric: str = "cosine") -> np.ndarray:
        """
        Calculate similarity matrix between source and target vectors.
        
        Args:
            source_vectors: Source embeddings (n_source, dim)
            target_vectors: Target embeddings (n_target, dim)
            metric: Similarity metric ("cosine", "euclidean", or "dot")
            
        Returns:
            Similarity matrix with shape (n_source, n_target)
        """
        if len(source_vectors) == 0 or len(target_vectors) == 0:
            return np.array([[]])
            
        if metric == "cosine":
            # Normalize vectors for cosine similarity
            source_norm = np.linalg.norm(source_vectors, axis=1, keepdims=True)
            target_norm = np.linalg.norm(target_vectors, axis=1, keepdims=True)
            
            # Avoid division by zero
            source_norm = np.maximum(source_norm, 1e-9)
            target_norm = np.maximum(target_norm, 1e-9)
            
            source_vectors = source_vectors / source_norm
            target_vectors = target_vectors / target_norm
            
            # Calculate cosine similarity
            return np.dot(source_vectors, target_vectors.T)
            
        elif metric == "euclidean":
            # Calculate pairwise distances
            distances = np.zeros((len(source_vectors), len(target_vectors)))
            for i, sv in enumerate(source_vectors):
                distances[i] = np.sqrt(np.sum((sv - target_vectors)**2, axis=1))
            
            # Convert distances to similarities (1 / (1 + distance))
            return 1.0 / (1.0 + distances)
            
        elif metric == "dot":
            # Simple dot product
            return np.dot(source_vectors, target_vectors.T)
            
        else:
            self.logger.warning(f"Unknown similarity metric: {metric}. Using cosine.")
            return self.calculate_similarity_matrix(source_vectors, target_vectors, "cosine")
    
    def _preprocess_texts(self,
                         texts: List[str],
                         language: str,
                         level: str = "standard") -> List[str]:
        """
        Apply language-specific preprocessing to texts.
        
        Args:
            texts: List of texts to preprocess
            language: Language code
            level: Preprocessing intensity level
            
        Returns:
            List of preprocessed texts
        """
        # Get language-specific preprocessing if available
        preprocess_fn = self._get_language_preprocessor(language, level)
        
        # Apply preprocessing in parallel for large inputs
        if len(texts) > 100:
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                return list(executor.map(preprocess_fn, texts))
        else:
            return [preprocess_fn(text) for text in texts]
    
    def _get_language_preprocessor(self, language: str, level: str) -> Callable[[str], str]:
        """
        Get appropriate preprocessing function for language and level.
        
        Args:
            language: ISO language code
            level: Preprocessing intensity level
            
        Returns:
            Preprocessing function
        """
        # This could be expanded with language-specific preprocessing
        if level == "minimal":
            # Just basic cleaning
            return lambda text: self.text_processor.basic_clean(text)
            
        elif level == "aggressive":
            # More intensive normalization
            return lambda text: self.text_processor.aggressive_normalize(text, language)
            
        else:  # "standard" or any other value
            # Standard preprocessing
            return lambda text: self.text_processor.normalize(text, language)
    
    def _apply_pooling(self, embeddings: np.ndarray, strategy: str) -> np.ndarray:
        """
        Apply custom pooling strategy to token embeddings.
        
        Args:
            embeddings: Token embeddings
            strategy: Pooling strategy name
            
        Returns:
            Pooled embeddings
        """
        # Most pooling is handled by the SentenceTransformer model already
        # This is a placeholder for custom pooling strategies
        return embeddings
    
    def _generate_cache_key(self,
                          text: str,
                          language: str,
                          preprocessing: str,
                          pooling: str) -> str:
        """
        Generate a unique cache key for an embedding.
        
        Args:
            text: Input text
            language: Language code
            preprocessing: Preprocessing level
            pooling: Pooling strategy
            
        Returns:
            Cache key string
        """
        # Hash the text to avoid file system issues with long texts
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Create a unique key combining all parameters
        return f"{text_hash}_{language}_{preprocessing}_{pooling}"
    
    def _check_cache_batch(self,
                         texts: List[str],
                         language: str,
                         preprocessing: str,
                         pooling: str) -> Tuple[Dict[int, np.ndarray], List[int], List[str]]:
        """
        Check cache for multiple texts.
        
        Args:
            texts: List of texts to check
            language: Language code
            preprocessing: Preprocessing level
            pooling: Pooling strategy
            
        Returns:
            Tuple of (cache_hits, missing_indices, missing_texts)
        """
        cached_embeddings = {}
        missing_indices = []
        missing_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, language, preprocessing, pooling)
            cache_path = self.cache_dir / f"{cache_key}.npy"
            
            if cache_path.exists():
                try:
                    embedding = np.load(cache_path)
                    cached_embeddings[i] = embedding
                    self.cache_hits += 1
                except Exception as e:
                    self.logger.warning(f"Error loading cached embedding: {e}")
                    missing_indices.append(i)
                    missing_texts.append(text)
                    self.cache_misses += 1
            else:
                missing_indices.append(i)
                missing_texts.append(text)
                self.cache_misses += 1
        
        return cached_embeddings, missing_indices, missing_texts
    
    def _update_cache_batch(self,
                          texts: List[str],
                          embeddings: np.ndarray,
                          language: str,
                          preprocessing: str,
                          pooling: str) -> None:
        """
        Update cache with batch of embeddings.
        
        Args:
            texts: Texts corresponding to embeddings
            embeddings: Embedding array
            language: Language code
            preprocessing: Preprocessing level
            pooling: Pooling strategy
        """
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, language, preprocessing, pooling)
            cache_path = self.cache_dir / f"{cache_key}.npy"
            
            try:
                np.save(cache_path, embeddings[i])
            except Exception as e:
                self.logger.warning(f"Error caching embedding: {e}")
                
    def get_language_specific_example_prompts(self, language: str) -> List[str]:
        """
        Get example prompts for specific languages that highlight unique characteristics.
        
        Args:
            language: ISO language code
            
        Returns:
            List of example prompts specific to the language
        """
        # Language-specific examples demonstrate different writing systems,
        # grammatical structures, and vocabulary
        examples = {
            'en': ["This is an example in English with some technical terms like API and vector embeddings.",
                  "Natural language processing helps computers understand human languages."],
            'es': ["Este es un ejemplo en español con algunas palabras acentuadas.",
                  "Los modelos de procesamiento de lenguaje natural son muy útiles."],
            'fr': ["Voici un exemple en français avec certains caractères spéciaux.",
                  "Les plumes de ma tante sont sur le bureau de mon oncle."],
            'de': ["Dies ist ein Beispiel auf Deutsch mit zusammengesetzten Wörtern.",
                  "Natursprachverarbeitung ist ein faszinierendes Forschungsgebiet."],
            'zh': ["这是一个中文例子，包含汉字。",
                  "自然语言处理是人工智能的一个分支。"],
            'ja': ["これは日本語のサンプルテキストです。",
                  "自然言語処理は人工知能の一分野です。"],
            'ar': ["هذا مثال باللغة العربية مع بعض الكتابة من اليمين إلى اليسار.",
                  "معالجة اللغة الطبيعية هي إحدى مجالات الذكاء الاصطناعي."],
            'hi': ["यह हिंदी में एक उदाहरण है, जिसमें देवनागरी लिपि है।",
                  "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर विज्ञान की एक शाखा है।"]
        }
        
        # Return examples for requested language or English if not available
        return examples.get(language, examples['en']) 