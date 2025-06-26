import langdetect
import pycountry
from fasttext.FastText import load_model
import os
from typing import Dict, List, Tuple, Optional, Union, Set
from pathlib import Path
import logging
import re
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageDetector:
    """Enhanced language detection with support for multiple detection methods."""
    
    # Language families for optimizing embedding models
    LANGUAGE_FAMILIES = {
        'germanic': ['en', 'de', 'nl', 'sv', 'da', 'no'],
        'romance': ['fr', 'es', 'it', 'pt', 'ro'],
        'slavic': ['ru', 'pl', 'cs', 'bg', 'uk', 'sr', 'hr'],
        'indic': ['hi', 'bn', 'pa', 'gu', 'mr'],
        'semitic': ['ar', 'he'],
        'sino-tibetan': ['zh', 'ja', 'ko']
    }
    
    # Mapping language codes to full names
    LANGUAGE_NAMES = {}
    
    def __init__(self, fasttext_model_path: Optional[str] = None):
        """Initialize language detector with optional FastText model."""
        self._initialize_language_names()
        self.fasttext_model = None
        
        # Try to load FastText model if path is provided
        if fasttext_model_path:
            try:
                self.fasttext_model = load_model(fasttext_model_path)
                logger.info(f"Loaded FastText language detection model from {fasttext_model_path}")
            except Exception as e:
                logger.warning(f"Could not load FastText model: {e}")
                logger.info("Falling back to langdetect for language detection")
    
    def _initialize_language_names(self):
        """Initialize mapping of language codes to full names."""
        for lang in pycountry.languages:
            if hasattr(lang, 'alpha_2'):
                self.LANGUAGE_NAMES[lang.alpha_2] = lang.name
        
        # Add some common languages that might be missing
        additional_langs = {
            'zh': 'Chinese',
            'jv': 'Javanese',
            'ko': 'Korean',
            'ja': 'Japanese'
        }
        self.LANGUAGE_NAMES.update(additional_langs)
    
    def detect(self, text: str, min_length: int = 50) -> Dict:
        """
        Detect language of a text using multiple detection methods.
        
        Args:
            text (str): Text to analyze
            min_length (int): Minimum text length for reliable detection
            
        Returns:
            Dict: {'code': 'en', 'name': 'English', 'confidence': 0.92, 'family': 'germanic'}
        """
        # Empty input yields undetermined immediately
        if not text:
            return {
                'code': 'und',
                'name': 'Undetermined',
                'confidence': 0.0,
                'family': None,
            }

        # Warn about very short input but still attempt detection instead of bailing out
        if len(text) < min_length:
            logger.warning(
                "Text too short (%d chars) – attempting language detection anyway, results may be unreliable",
                len(text),
            )
        
        # Try FastText detection if available
        if self.fasttext_model:
            try:
                labels, probabilities = self.fasttext_model.predict(text, k=1)
                code = labels[0].replace('__label__', '')
                confidence = float(probabilities[0])
                
                # Fallback to langdetect for very low confidence results
                if confidence < 0.5:
                    return self._detect_with_langdetect(text)
                
                return {
                    'code': code,
                    'name': self.LANGUAGE_NAMES.get(code, 'Unknown'),
                    'confidence': confidence,
                    'family': self.get_language_family(code)
                }
            except Exception as e:
                logger.warning(f"FastText detection failed: {e}")
                return self._detect_with_langdetect(text)
        else:
            return self._detect_with_langdetect(text)
    
    def _detect_with_langdetect(self, text: str) -> Dict:
        """Use langdetect as a fallback detection method."""
        try:
            # Get language probabilities
            langdetect.DetectorFactory.seed = 0  # For consistent results
            lang_probs = langdetect.detect_langs(text)
            
            if not lang_probs:
                return {'code': 'und', 'name': 'Undetermined', 'confidence': 0.0, 'family': None}
            
            # Get the most probable language
            code = lang_probs[0].lang
            confidence = lang_probs[0].prob
            
            return {
                'code': code,
                'name': self.LANGUAGE_NAMES.get(code, 'Unknown'),
                'confidence': confidence,
                'family': self.get_language_family(code)
            }
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {'code': 'und', 'name': 'Undetermined', 'confidence': 0.0, 'family': None}
    
    def get_language_family(self, code: str) -> Optional[str]:
        """Determine the language family for a given language code."""
        for family, languages in self.LANGUAGE_FAMILIES.items():
            if code in languages:
                return family
        return None
    
    def get_optimal_model(self, source_lang: str, target_lang: str) -> str:
        """
        Get the optimal embedding model for a language pair.
        
        Args:
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            str: Model name recommendation
        """
        # Default multilingual model
        default_model = "paraphrase-multilingual-MiniLM-L12-v2"
        
        # Special case for English pairs - can use specialized models
        if source_lang == 'en' and target_lang != 'en':
            return default_model
        elif target_lang == 'en' and source_lang != 'en':
            return default_model
            
        # For same language or same family, default multilingual works well
        source_family = self.get_language_family(source_lang)
        target_family = self.get_language_family(target_lang)
        
        if source_lang == target_lang or source_family == target_family:
            return default_model
            
        # For distant language pairs, ensure robust multilingual model
        return "paraphrase-multilingual-mpnet-base-v2"  # More powerful model for distant languages

def get_supported_languages() -> List[Dict]:
    """
    Get a list of languages supported by the system.
    
    Returns:
        List[Dict]: List of dictionaries with language info
    """
    detector = LanguageDetector()
    supported = []
    
    for code, name in detector.LANGUAGE_NAMES.items():
        family = detector.get_language_family(code)
        supported.append({
            'code': code,
            'name': name,
            'family': family
        })
    
    return sorted(supported, key=lambda x: x['name'])

class EmbeddingBasedLanguageDetector:
    """
    Advanced language detector using embedding-based clustering for improved accuracy.
    
    This detector uses multilingual embeddings to represent reference text samples
    from various languages, then identifies the language of input text by finding
    the closest matches in embedding space.
    """
    
    # ISO Language codes by frequency of global usage (approximate)
    COMMON_LANGUAGES = [
        'en', 'zh', 'es', 'ar', 'hi', 'fr', 'ru', 'pt', 'id', 'bn', 'ja', 
        'de', 'pa', 'ur', 'tr', 'it', 'ko', 'vi', 'pl', 'uk', 'fa', 'ro', 
        'nl', 'th', 'el', 'cs', 'sv', 'hu', 'da', 'fi', 'no', 'he'
    ]
    
    # Language families for better clustering
    LANGUAGE_FAMILIES = {
        'germanic': ['en', 'de', 'nl', 'sv', 'da', 'no', 'is'],
        'romance': ['fr', 'es', 'it', 'pt', 'ro', 'ca'],
        'slavic': ['ru', 'pl', 'cs', 'bg', 'uk', 'sr', 'hr', 'sk', 'sl'],
        'indic': ['hi', 'bn', 'pa', 'gu', 'mr', 'ne', 'si', 'ur'],
        'semitic': ['ar', 'he', 'am', 'mt'],
        'sino-tibetan': ['zh', 'my', 'bo'],
        'japonic': ['ja'],
        'koreanic': ['ko'],
        'turkic': ['tr', 'az', 'kk', 'ky', 'uz'],
        'austronesian': ['id', 'ms', 'tl', 'jv'],
        'tai-kadai': ['th', 'lo'],
        'dravidian': ['ta', 'te', 'kn', 'ml'],
        'uralic': ['fi', 'hu', 'et']
    }
    
    # Script to language mapping for quick identification
    SCRIPT_TO_LANGUAGES = {
        'Arabic': ['ar', 'fa', 'ur', 'ps'],
        'Cyrillic': ['ru', 'uk', 'bg', 'sr', 'mk', 'be', 'kk'],
        'Devanagari': ['hi', 'mr', 'ne', 'sa'],
        'Greek': ['el'],
        'Han': ['zh', 'ja', 'ko'],  # Chinese characters used in multiple languages
        'Hangul': ['ko'],
        'Hebrew': ['he', 'yi'],
        'Hiragana': ['ja'],
        'Katakana': ['ja'],
        'Latin': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'cs', 'hu', 'ro', 'sv', 'da', 'fi', 'no', 'tr'],
        'Thai': ['th'],
        'Bengali': ['bn'],
        'Gujarati': ['gu'],
        'Gurmukhi': ['pa'],
        'Tamil': ['ta'],
        'Telugu': ['te'],
    }
    
    # Character ranges for script detection
    SCRIPT_RANGES = {
        'Arabic': [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ],
        'Cyrillic': [(0x0400, 0x04FF)],  # Cyrillic
        'Devanagari': [(0x0900, 0x097F)],  # Devanagari
        'Greek': [(0x0370, 0x03FF)],  # Greek and Coptic
        'Han': [(0x4E00, 0x9FFF)],  # CJK Unified Ideographs
        'Hangul': [(0xAC00, 0xD7AF)],  # Hangul Syllables
        'Hebrew': [(0x0590, 0x05FF)],  # Hebrew
        'Hiragana': [(0x3040, 0x309F)],  # Hiragana
        'Katakana': [(0x30A0, 0x30FF)],  # Katakana
        'Latin': [(0x0020, 0x007F), (0x00A0, 0x00FF)],  # Basic Latin + Latin-1 Supplement
        'Thai': [(0x0E00, 0x0E7F)],  # Thai
        'Bengali': [(0x0980, 0x09FF)],  # Bengali
        'Gujarati': [(0x0A80, 0x0AFF)],  # Gujarati
        'Gurmukhi': [(0x0A00, 0x0A7F)],  # Gurmukhi
        'Tamil': [(0x0B80, 0x0BFF)],  # Tamil
        'Telugu': [(0x0C00, 0x0C7F)],  # Telugu
    }
    
    def __init__(self, model_manager, config, reference_data_path=None):
        """
        Initialize the embedding-based language detector.
        
        Args:
            model_manager: MultilingualModelManager instance
            config: Config manager instance
            reference_data_path: Path to reference embeddings data
        """
        self.logger = logging.getLogger("tqa.language.embeddings")
        self.model_manager = model_manager
        self.config = config
        
        # Get embedding dimension from the model
        self.embedding_dim = self.config.get("models.embedding.dimension", 384)
        
        # Initialize reference data containers
        self.reference_embeddings = {}  # {lang_code: np.array of embeddings}
        self.reference_centroids = {}   # {lang_code: centroid vector}
        self.reference_phrases = {}     # {lang_code: [phrases]}
        
        # Setup the cache directory
        self.cache_dir = Path(self.config.get("language.cache_dir", "~/.tqa/language_cache"))
        self.cache_dir = self.cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load reference data if provided
        if reference_data_path:
            self.load_reference_data(reference_data_path)
        else:
            # Try to load from default location
            default_path = self.cache_dir / "language_reference_data.pkl"
            if default_path.exists():
                self.load_reference_data(default_path)
            else:
                self.logger.info("No reference data loaded. Use generate_reference_data() to create it.")
                
        # Initialize nearest neighbors model (lazy initialization)
        self.nn_model = None
        self.languages = None  # List of languages in the same order as reference vectors
    
    def detect_script(self, text: str) -> Dict[str, float]:
        """
        Detect the script used in the text based on Unicode character ranges.
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping script names to their proportions in the text
        """
        if not text:
            return {"Unknown": 1.0}
            
        script_counts = Counter()
        total_chars = 0
        
        for char in text:
            if char.isspace():
                continue
                
            code_point = ord(char)
            matched = False
            
            for script, ranges in self.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[script] += 1
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                script_counts["Other"] += 1
                
            total_chars += 1
            
        # Convert counts to proportions
        if total_chars > 0:
            script_proportions = {script: count/total_chars for script, count in script_counts.items()}
            return script_proportions
        else:
            return {"Unknown": 1.0}
    
    def get_possible_languages_from_script(self, text: str) -> Set[str]:
        """
        Get possible languages based on detected script.
        
        Args:
            text: Input text
            
        Returns:
            Set of possible language codes
        """
        # Detect script proportions
        script_proportions = self.detect_script(text)
        
        # Get dominant script (with at least 30% of characters)
        dominant_scripts = [script for script, prop in script_proportions.items() 
                          if prop >= 0.3 and script in self.SCRIPT_TO_LANGUAGES]
        
        # Combine possible languages from all dominant scripts
        possible_langs = set()
        for script in dominant_scripts:
            possible_langs.update(self.SCRIPT_TO_LANGUAGES.get(script, []))
            
        # If no dominant script found, return all common languages
        if not possible_langs:
            return set(self.COMMON_LANGUAGES)
            
        return possible_langs
    
    def generate_reference_data(self, languages: List[str] = None, 
                              samples_per_lang: int = 100, 
                              save_path: Optional[str] = None,
                              use_external_sources: bool = True) -> None:
        """
        Generate reference embeddings data for language detection.
        
        Args:
            languages: List of language codes to include (defaults to COMMON_LANGUAGES)
            samples_per_lang: Number of reference samples per language
            save_path: Path to save reference data
            use_external_sources: Whether to use external data sources
        """
        languages = languages or self.COMMON_LANGUAGES
        self.logger.info(f"Generating reference data for {len(languages)} languages")
        
        # Get reference phrases for each language
        for lang in languages:
            self.logger.info(f"Collecting reference phrases for {lang}")
            phrases = self._get_reference_phrases(lang, samples_per_lang, use_external_sources)
            self.reference_phrases[lang] = phrases
            
        # Generate embeddings for all phrases
        self.logger.info("Generating embeddings for reference phrases")
        for lang, phrases in self.reference_phrases.items():
            self.logger.info(f"Processing {len(phrases)} phrases for {lang}")
            
            # Get model for this language
            model = self.model_manager.get_model(lang, lang)
            
            # Generate embeddings
            try:
                embeddings = model.encode(phrases, convert_to_numpy=True)
                self.reference_embeddings[lang] = embeddings
                
                # Calculate centroid
                centroid = np.mean(embeddings, axis=0)
                self.reference_centroids[lang] = centroid
                
                self.logger.info(f"Generated {len(embeddings)} embeddings for {lang}")
            except Exception as e:
                self.logger.error(f"Error generating embeddings for {lang}: {e}")
        
        # Build the nearest neighbors model
        self._build_nearest_neighbors_model()
        
        # Save the reference data if requested
        if save_path:
            self._save_reference_data(save_path)
        elif self.cache_dir:
            default_path = self.cache_dir / "language_reference_data.pkl"
            self._save_reference_data(default_path)
    
    def load_reference_data(self, path: Union[str, Path]) -> None:
        """
        Load reference embeddings data from file.
        
        Args:
            path: Path to reference data file
        """
        path = Path(path)
        if not path.exists():
            self.logger.error(f"Reference data file not found: {path}")
            return
            
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            self.reference_embeddings = data.get('embeddings', {})
            self.reference_centroids = data.get('centroids', {})
            self.reference_phrases = data.get('phrases', {})
            
            self.logger.info(f"Loaded reference data for {len(self.reference_embeddings)} languages from {path}")
            
            # Build nearest neighbors model
            self._build_nearest_neighbors_model()
        except Exception as e:
            self.logger.error(f"Error loading reference data: {e}")
    
    def detect_language(self, text: str, 
                       min_length: int = 20, 
                       fast_mode: bool = False,
                       return_scores: bool = False) -> Union[str, Dict]:
        """
        Detect the language of a text using embedding-based clustering.
        
        Args:
            text: Text to analyze
            min_length: Minimum text length for reliable detection
            fast_mode: Use faster but potentially less accurate detection
            return_scores: Whether to return confidence scores for all languages
            
        Returns:
            Language code (str) or dict with detailed information
        """
        if not text or len(text) < min_length:
            self.logger.warning(f"Text too short ({len(text) if text else 0} chars) for reliable detection")
            
            # For very short text, try to detect script as fallback
            if text:
                script_props = self.detect_script(text)
                dominant_script = max(script_props.items(), key=lambda x: x[1])
                
                if dominant_script[0] in self.SCRIPT_TO_LANGUAGES:
                    possible_langs = self.SCRIPT_TO_LANGUAGES[dominant_script[0]]
                    if len(possible_langs) == 1:
                        lang_code = possible_langs[0]
                        return lang_code if not return_scores else {
                            'language': lang_code,
                            'confidence': dominant_script[1],
                            'all_scores': {lang_code: dominant_script[1]},
                            'method': 'script'
                        }
            
            # Use langdetect as backup
            try:
                langdetect.DetectorFactory.seed = 0  # For consistent results
                detected = langdetect.detect(text)
                confidence = 0.5  # Medium confidence for langdetect on short text
                
                return detected if not return_scores else {
                    'language': detected,
                    'confidence': confidence,
                    'all_scores': {detected: confidence},
                    'method': 'langdetect'
                }
            except:
                return 'und' if not return_scores else {
                    'language': 'und',  # Undetermined
                    'confidence': 0.0,
                    'all_scores': {},
                    'method': 'fallback'
                }
        
        # Fast mode - use script detection to narrow down candidates
        possible_languages = None
        if fast_mode:
            possible_languages = self.get_possible_languages_from_script(text)
            
        # For efficiency with long texts, sample a few sentences
        if len(text) > 1000:
            # Extract sentences and take a representative sample
            sentences = re.split(r'[.!?।।\n]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > min_length]
            
            # Take a diverse sample of sentences
            if len(sentences) > 5:
                sample_indices = [0, len(sentences)//4, len(sentences)//2, 
                                 3*len(sentences)//4, len(sentences)-1]
                sample_text = ' '.join([sentences[i] for i in sample_indices])
            else:
                sample_text = ' '.join(sentences[:5])
                
            # Ensure sample isn't too long
            if len(sample_text) > 1000:
                sample_text = sample_text[:1000]
        else:
            sample_text = text
            
        # Generate embedding for the text
        model = self.model_manager.get_model(lang, lang)
        text_embedding = model.encode(sample_text, convert_to_numpy=True)
        
        # Find nearest languages
        if self.nn_model is not None:
            # Get the languages we want to consider
            lang_indices = None
            if possible_languages:
                lang_indices = [i for i, lang in enumerate(self.languages) 
                               if lang in possible_languages]
            
            # Get nearest neighbors
            if lang_indices is not None:
                # Only search within possible languages
                filtered_vectors = np.vstack([self.all_centroids[i] for i in lang_indices])
                distances, indices = self._compute_nearest(filtered_vectors, text_embedding)
                nearest_langs = [self.languages[lang_indices[idx]] for idx in indices[0]]
                scores = 1.0 / (1.0 + distances[0])
            else:
                # Search all languages
                distances, indices = self.nn_model.kneighbors([text_embedding])
                nearest_langs = [self.languages[idx] for idx in indices[0]]
                scores = 1.0 / (1.0 + distances[0])
            
            # Calculate confidence scores
            lang_scores = {lang: float(score) for lang, score in zip(nearest_langs, scores)}
            
            # Normalize scores to sum to 1
            total_score = sum(lang_scores.values())
            if total_score > 0:
                lang_scores = {k: v/total_score for k, v in lang_scores.items()}
            
            # Get the top language and its score
            top_lang = nearest_langs[0]
            confidence = lang_scores[top_lang]
            
            if return_scores:
                return {
                    'language': top_lang,
                    'confidence': confidence,
                    'all_scores': lang_scores,
                    'method': 'embedding'
                }
            else:
                return top_lang
        else:
            # Fallback to direct comparison
            best_lang = None
            best_similarity = -1
            lang_scores = {}
            
            for lang, centroid in self.reference_centroids.items():
                # Skip if not in possible languages (for fast mode)
                if possible_languages and lang not in possible_languages:
                    continue
                    
                # Calculate cosine similarity
                similarity = self._cosine_similarity(text_embedding, centroid)
                lang_scores[lang] = float(similarity)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_lang = lang
            
            # Normalize scores
            total_score = sum(lang_scores.values())
            if total_score > 0:
                lang_scores = {k: v/total_score for k, v in lang_scores.items()}
            
            confidence = lang_scores.get(best_lang, 0.0) if best_lang else 0.0
            
            if return_scores:
                return {
                    'language': best_lang or 'und',
                    'confidence': confidence,
                    'all_scores': lang_scores,
                    'method': 'centroid'
                }
            else:
                return best_lang or 'und'
    
    def detect_language_mix(self, text: str, min_segment_length: int = 30) -> Dict[str, Dict]:
        """
        Detect multiple languages in a single text by segment.
        
        Args:
            text: Text to analyze
            min_segment_length: Minimum segment length to analyze
            
        Returns:
            Dict mapping segment indices to language information
        """
        # Split text into reasonable segments
        segments = self._split_text_for_detection(text, min_segment_length)
        
        result = {}
        prev_lang = None
        
        for i, segment in enumerate(segments):
            # Skip very short segments
            if len(segment) < min_segment_length:
                # Try to merge with previous or next segment
                if prev_lang is not None:
                    result[i] = {'text': segment, 'language': prev_lang, 'inherited': True}
                continue
                
            # Detect language for this segment
            lang_info = self.detect_language(segment, return_scores=True)
            result[i] = {'text': segment, **lang_info, 'inherited': False}
            prev_lang = lang_info['language']
            
        return result
    
    def detect_language_composition(self, text: str) -> Dict[str, float]:
        """
        Detect the composition of languages in text as percentages.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict mapping language codes to their percentage in the text
        """
        # Get language mix by segment
        segments = self.detect_language_mix(text)
        
        # Count character lengths per language
        lang_chars = defaultdict(int)
        for info in segments.values():
            lang = info['language']
            # Only count non-inherited segments
            if not info.get('inherited', False):
                lang_chars[lang] += len(info['text'])
        
        # Calculate percentages
        total_chars = sum(lang_chars.values())
        if total_chars == 0:
            return {'und': 1.0}
            
        return {lang: count/total_chars for lang, count in lang_chars.items()}
    
    def _get_reference_phrases(self, language: str, 
                             count: int = 100, 
                             use_external: bool = True) -> List[str]:
        """
        Get reference phrases for a language.
        
        Args:
            language: Language code
            count: Number of phrases to get
            use_external: Whether to use external sources
            
        Returns:
            List of reference phrases
        """
        phrases = []
        
        # Add common phrases in this language
        phrases.extend(self._get_common_phrases(language))
        
        # Use Wikipedia content if available
        if use_external:
            wiki_phrases = self._get_wikipedia_samples(language, count - len(phrases))
            phrases.extend(wiki_phrases)
        
        # Fill with generated phrases if needed
        if len(phrases) < count:
            phrases.extend(self._get_language_model_samples(language, count - len(phrases)))
        
        # Ensure we have exactly the requested number
        if len(phrases) > count:
            phrases = phrases[:count]
        
        return phrases
    
    def _get_common_phrases(self, language: str) -> List[str]:
        """Get common phrases for a language."""
        # This could be expanded with a more comprehensive database
        common_phrases = {
            'en': [
                "Hello, how are you?", 
                "Thank you very much.",
                "Welcome to our service.",
                "Please let me know if you need assistance.", 
                "I would like to make a reservation."
            ],
            'es': [
                "Hola, ¿cómo estás?",
                "Muchas gracias.",
                "Bienvenido a nuestro servicio.",
                "Por favor, avísame si necesitas ayuda.",
                "Me gustaría hacer una reserva."
            ],
            'fr': [
                "Bonjour, comment allez-vous ?",
                "Merci beaucoup.",
                "Bienvenue à notre service.",
                "S'il vous plaît, faites-moi savoir si vous avez besoin d'aide.",
                "Je voudrais faire une réservation."
            ],
            'de': [
                "Hallo, wie geht es Ihnen?",
                "Vielen Dank.",
                "Willkommen zu unserem Service.",
                "Bitte lassen Sie mich wissen, wenn Sie Hilfe benötigen.",
                "Ich möchte eine Reservierung machen."
            ],
            'zh': [
                "你好，最近好吗？",
                "非常感谢。",
                "欢迎使用我们的服务。",
                "如果需要帮助，请告诉我。",
                "我想做个预订。"
            ],
            'ja': [
                "こんにちは、お元気ですか？",
                "どうもありがとうございます。",
                "私たちのサービスへようこそ。",
                "何かお手伝いが必要でしたら、お知らせください。",
                "予約をしたいと思います。"
            ],
            'ru': [
                "Здравствуйте, как дела?",
                "Большое спасибо.",
                "Добро пожаловать в наш сервис.",
                "Пожалуйста, дайте мне знать, если вам нужна помощь.",
                "Я хотел бы сделать бронирование."
            ]
        }
        
        # Return phrases for the language or empty list if not found
        return common_phrases.get(language, [])
    
    def _get_wikipedia_samples(self, language: str, count: int) -> List[str]:
        """Get text samples from Wikipedia for a language."""
        # In a real implementation, this would fetch from Wikipedia API
        # For now, return placeholders
        return [
            f"Sample text in {language} #{i+1} (would be from Wikipedia)" 
            for i in range(min(count, 3))
        ]
    
    def _get_language_model_samples(self, language: str, count: int) -> List[str]:
        """Generate text samples for a language using LLM."""
        # In a real implementation, this would use a LLM to generate samples
        # For now, return placeholders
        return [
            f"Generated sample in {language} #{i+1}" 
            for i in range(count)
        ]
    
    def _split_text_for_detection(self, text: str, min_length: int) -> List[str]:
        """Split text into segments for language detection."""
        # First try to split by paragraph markers
        paragraphs = re.split(r'\n\s*\n', text)
        
        # If we have very few paragraphs, try sentence splitting
        if len(paragraphs) < 3:
            segments = re.split(r'[.!?।।\n]+', text)
        else:
            segments = paragraphs
        
        # Clean up segments
        segments = [s.strip() for s in segments if s.strip()]
        
        # Merge very short segments
        result = []
        current = ""
        
        for segment in segments:
            if len(current) + len(segment) < min_length:
                current += " " + segment if current else segment
            else:
                if current:
                    result.append(current)
                current = segment
        
        # Add the last segment if it exists
        if current:
            result.append(current)
            
        return result
    
    def _build_nearest_neighbors_model(self) -> None:
        """Build nearest neighbors model for fast language lookup."""
        if not self.reference_centroids:
            self.logger.warning("No reference data available for nearest neighbors model")
            return
            
        # Compile all centroids into a single array
        self.languages = list(self.reference_centroids.keys())
        self.all_centroids = np.vstack([self.reference_centroids[lang] for lang in self.languages])
        
        # Build nearest neighbors model
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
        self.nn_model.fit(self.all_centroids)
        
        self.logger.info(f"Built nearest neighbors model with {len(self.languages)} languages")
    
    def _compute_nearest(self, vectors: np.ndarray, query: np.ndarray, 
                        k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Compute k nearest vectors to query."""
        # Calculate cosine distances
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1e-10
        
        normalized_query = query / query_norm
        
        # Normalize reference vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_vectors = vectors / norms
        
        # Calculate similarities
        similarities = np.dot(normalized_vectors, normalized_query)
        
        # Convert to distances
        distances = 1 - similarities
        
        # Get top k indices
        top_indices = np.argsort(distances)[:k]
        top_distances = distances[top_indices]
        
        return np.array([top_distances]), np.array([top_indices])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _save_reference_data(self, path: Union[str, Path]) -> None:
        """Save reference data to file."""
        path = Path(path)
        
        # Prepare data
        data = {
            'embeddings': self.reference_embeddings,
            'centroids': self.reference_centroids,
            'phrases': self.reference_phrases
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved reference data to {path}")
        except Exception as e:
            self.logger.error(f"Error saving reference data: {e}") 