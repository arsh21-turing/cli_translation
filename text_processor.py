"""
Text Processor for Smart CLI Translation Quality Analyzer
Handles language detection, normalization, segmentation, and chunking
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import unicodedata

import nltk
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
try:
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
except ImportError:
    # Fallback sentence tokenization if NLTK is not available
    def sent_tokenize(text, language='english'):
        return re.split(r'(?<=[.!?])\s+', text)

# Setting seed for reproducibility in language detection
DetectorFactory.seed = 0

class TextProcessor:
    """
    Handles text processing operations including language detection,
    normalization, sentence segmentation, and chunking for batch processing.
    """
    
    # Language code mapping (ISO 639-1 to NLTK tokenizer language)
    LANG_MAP = {
        'en': 'english',
        'fr': 'french',
        'de': 'german',
        'es': 'spanish',
        'it': 'italian',
        'nl': 'dutch',
        'pt': 'portuguese',
        'ru': 'russian',
        # Add more mappings as needed
    }
    
    # Common normalization patterns
    NORMALIZATION_PATTERNS = [
        # Remove extra whitespace
        (r'\s+', ' '),
        # Normalize quotes
        (r'[\u2018\u2019]', "'"),
        (r'[\u201C\u201D]', '"'),
        # Normalize dashes
        (r'[\u2013\u2014]', '-'),
        # Normalize ellipses
        (r'\.{3,}', '...'),
        # Fix spacing around punctuation
        (r'\s+([.,;:!?])', r'\1'),
        # Fix spacing with parentheses and brackets
        (r'\(\s+', '('),
        (r'\s+\)', ')'),
        (r'\[\s+', '['),
        (r'\s+\]', ']'),
    ]
    
    def __init__(self, config=None):
        """
        Initialize text processor.
        
        Args:
            config: Optional configuration object
        """
        self.logger = logging.getLogger("tqa.text")
        self.config = config or {}
        
        # Default chunk settings
        self.default_chunk_size = self.config.get("processing.chunk_size", 1000)
        self.default_chunk_overlap = self.config.get("processing.chunk_overlap", 200)
        
        # Compiled regex patterns for efficiency
        self.norm_patterns = [(re.compile(pattern), repl) 
                              for pattern, repl in self.NORMALIZATION_PATTERNS]
                              
    def detect_language(self, text: str, min_length: int = 20) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: The text to analyze
            min_length: Minimum text length for reliable detection
            
        Returns:
            ISO 639-1 language code (en, fr, de, etc.) or 'unknown' if detection fails
        """
        if not text or len(text.strip()) < min_length:
            self.logger.warning(f"Text too short for reliable language detection: '{text[:20]}...'")
            return 'unknown'
            
        try:
            return detect(text)
        except LangDetectException as e:
            self.logger.warning(f"Language detection failed: {e}")
            return 'unknown'
            
    def normalize_text(self, text: str, aggressive: bool = False) -> str:
        """
        Normalize text by applying various cleaning patterns.
        
        Args:
            text: Text to normalize
            aggressive: Whether to apply more aggressive normalization
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Basic normalization
        # First normalize unicode (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # Apply all regex normalization patterns
        for pattern, replacement in self.norm_patterns:
            text = pattern.sub(replacement, text)
            
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # More aggressive normalization if requested
        if aggressive:
            # Convert to lowercase
            text = text.lower()
            
            # Remove all non-alphanumeric except punctuation and whitespace
            text = re.sub(r'[^\w\s.,;:!?"\'-]', '', text)
            
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
        return text
        
    def segment_sentences(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to segment
            language: ISO 639-1 language code (auto-detect if None)
            
        Returns:
            List of sentences
        """
        if not text:
            return []
            
        # Detect language if not provided
        if language is None or language == 'unknown':
            language = self.detect_language(text)
            
        # Map ISO language code to NLTK language
        nltk_lang = self.LANG_MAP.get(language, 'english')
        
        try:
            # Use NLTK's sentence tokenizer
            sentences = sent_tokenize(text, language=nltk_lang)
            return sentences
        except Exception as e:
            self.logger.warning(f"Error in sentence segmentation: {e}. Using fallback.")
            # Fallback: split on basic sentence boundaries
            return re.split(r'(?<=[.!?])\s+', text)
            
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, 
                 overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks for batch processing.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk (default from config)
            overlap: Overlap between chunks (default from config)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
            
        # Get sentences first for better chunking
        sentences = self.segment_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed the chunk size and we already have content,
            # finish the current chunk and start a new one
            if current_length + sentence_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep some sentences for overlap
                if overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_text = ' '.join(current_chunk)
                    overlap_sentences = []
                    
                    # Add sentences from the end until we reach overlap size
                    for s in reversed(current_chunk):
                        if len(' '.join(overlap_sentences + [s])) <= overlap:
                            overlap_sentences.insert(0, s)
                        else:
                            break
                            
                    current_chunk = overlap_sentences
                    current_length = len(' '.join(current_chunk))
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def process_text(self, text: str, normalize: bool = True, 
                   segment: bool = False, chunk: bool = False,
                   language: Optional[str] = None) -> Union[str, List[str]]:
        """
        Apply a complete processing pipeline to text.
        
        Args:
            text: Text to process
            normalize: Whether to normalize text
            segment: Whether to segment into sentences
            chunk: Whether to chunk text for batch processing
            language: Language code (auto-detect if None)
            
        Returns:
            Processed text (string or list depending on segment/chunk flags)
        """
        if not text:
            return "" if not (segment or chunk) else []
            
        # Detect language if needed and not provided
        if language is None:
            language = self.detect_language(text)
            
        # Apply normalization if requested
        if normalize:
            text = self.normalize_text(text)
            
        # Apply segmentation if requested
        if segment:
            sentences = self.segment_sentences(text, language)
            
            # If chunking is also requested
            if chunk:
                return self.chunk_text(' '.join(sentences))
            return sentences
            
        # Apply chunking if requested
        if chunk:
            return self.chunk_text(text)
            
        # Return normalized text
        return text
        
    def align_sentences(self, source_text: str, translated_text: str,
                       source_lang: Optional[str] = None, 
                       target_lang: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Attempt to align sentences between source and translated text.
        Uses a naive approach based on number of sentences.
        
        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language code (auto-detect if None)
            target_lang: Target language code (auto-detect if None)
            
        Returns:
            List of (source_sentence, translated_sentence) pairs
        """
        # Detect languages if not provided
        if source_lang is None:
            source_lang = self.detect_language(source_text)
        if target_lang is None:
            target_lang = self.detect_language(translated_text)
            
        # Segment both texts into sentences
        source_sentences = self.segment_sentences(source_text, source_lang)
        target_sentences = self.segment_sentences(translated_text, target_lang)
        
        # Simple alignment based on sentence count
        source_count = len(source_sentences)
        target_count = len(target_sentences)
        
        # Log warnings if significant mismatch
        if abs(source_count - target_count) > 3 and abs(source_count - target_count) / max(source_count, target_count) > 0.3:
            self.logger.warning(f"Significant sentence count mismatch: {source_count} vs {target_count}")
        
        # Simple alignment strategies based on sentence count ratio
        if source_count == target_count:
            # 1:1 alignment
            return list(zip(source_sentences, target_sentences))
        elif source_count < target_count:
            # Source has fewer sentences - some source sentences might map to multiple target sentences
            aligned_pairs = []
            ratio = target_count / source_count
            
            for i, source in enumerate(source_sentences):
                # Calculate start and end indices in target
                start_idx = int(i * ratio)
                end_idx = int((i + 1) * ratio)
                # Ensure we don't exceed bounds
                end_idx = min(end_idx, target_count)
                
                # Combine target sentences if needed
                if start_idx + 1 >= end_idx:
                    aligned_pairs.append((source, target_sentences[start_idx]))
                else:
                    combined_target = ' '.join(target_sentences[start_idx:end_idx])
                    aligned_pairs.append((source, combined_target))
            
            return aligned_pairs
        else:
            # Target has fewer sentences - some target sentences might map to multiple source sentences
            aligned_pairs = []
            ratio = source_count / target_count
            
            for i, target in enumerate(target_sentences):
                # Calculate start and end indices in source
                start_idx = int(i * ratio)
                end_idx = int((i + 1) * ratio)
                # Ensure we don't exceed bounds
                end_idx = min(end_idx, source_count)
                
                # Combine source sentences if needed
                if start_idx + 1 >= end_idx:
                    aligned_pairs.append((source_sentences[start_idx], target))
                else:
                    combined_source = ' '.join(source_sentences[start_idx:end_idx])
                    aligned_pairs.append((combined_source, target))
            
            return aligned_pairs
            
    def calculate_statistics(self, text: str) -> Dict[str, Any]:
        """
        Calculate text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with statistics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'language': 'unknown'
            }
            
        # Detect language
        language = self.detect_language(text)
        
        # Get sentences
        sentences = self.segment_sentences(text, language)
        sentence_count = len(sentences)
        
        # Count words and characters
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        char_count = len(text)
        
        # Calculate averages
        avg_sentence_length = word_count / max(1, sentence_count)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'language': language
        }