"""
Model Loader for Smart CLI Translation Quality Analyzer
Handles downloading, caching, and inference with HuggingFace models and Groq API
"""

import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import hashlib

import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from config_manager import ConfigManager
from language_utils import LanguageDetector, get_supported_languages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration of supported model types"""
    EMBEDDING = "embedding"
    TRANSLATOR = "translator"
    LLM = "llm"

class InferenceMode(Enum):
    """Enumeration of inference modes"""
    LOCAL = "local"
    API = "api"
    HYBRID = "hybrid"  # Try local first, fall back to API

class ModelLoader:
    """
    Handles loading, caching, and inference for NLP models.
    Supports both local Hugging Face models and remote APIs.
    """
    
    def __init__(self, config: ConfigManager, 
                inference_mode: Union[str, InferenceMode] = InferenceMode.HYBRID):
        """
        Initialize the model loader.
        
        Args:
            config: Configuration manager instance
            inference_mode: Preferred inference mode (LOCAL, API, or HYBRID)
        """
        self.config = config
        self.logger = logging.getLogger("tqa.models")
        
        # Convert string to enum if needed
        if isinstance(inference_mode, str):
            try:
                self.inference_mode = InferenceMode(inference_mode.lower())
            except ValueError:
                self.logger.warning(f"Invalid inference mode: {inference_mode}. Using HYBRID.")
                self.inference_mode = InferenceMode.HYBRID
        else:
            self.inference_mode = inference_mode
            
        # Setup cache directory
        self.cache_dir = Path(os.path.expanduser(config.get("cache.directory", "~/.tqa/cache")))
        self.models_dir = Path(os.path.expanduser(config.get("models.embedding.cache_dir", "~/.tqa/models")))
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache tracking
        self._model_cache = {}
        self._api_cache = {}
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
    def get_embedding_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Get a sentence transformer model for embeddings.
        
        Args:
            model_name: Name of the model to load, uses default from config if None
            
        Returns:
            Loaded SentenceTransformer model
        """
        if model_name is None:
            model_name = self.config.get_model_path(ModelType.EMBEDDING.value)
            
        # Check if model is already loaded
        cache_key = f"embedding_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        # Try to load model
        try:
            if self.inference_mode in [InferenceMode.LOCAL, InferenceMode.HYBRID]:
                self.logger.info(f"Loading embedding model: {model_name}")
                model = SentenceTransformer(model_name, cache_folder=str(self.models_dir))
                model.to(self.device)
                self._model_cache[cache_key] = model
                return model
                
        except Exception as e:
            self.logger.error(f"Error loading embedding model {model_name}: {e}")
            
            if self.inference_mode == InferenceMode.LOCAL:
                raise
                
        # Fall back to API if local loading failed or API mode is selected
        if self.inference_mode in [InferenceMode.API, InferenceMode.HYBRID]:
            api_model = self._create_api_embedding_model(model_name)
            self._model_cache[cache_key] = api_model
            return api_model
            
        raise ValueError(f"Failed to load embedding model: {model_name}")
        
    def get_translator_model(self, model_name: Optional[str] = None) -> Any:
        """
        Get a translator model.
        
        Args:
            model_name: Name of the model to load, uses default from config if None
            
        Returns:
            Loaded translator model
        """
        if model_name is None:
            model_name = self.config.get_model_path(ModelType.TRANSLATOR.value)
            
        # Check if model is already loaded
        cache_key = f"translator_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        # Try to load model locally
        try:
            if self.inference_mode in [InferenceMode.LOCAL, InferenceMode.HYBRID]:
                self.logger.info(f"Loading translator model: {model_name}")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    cache_dir=str(self.models_dir)
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=str(self.models_dir)
                )
                model.to(self.device)
                self._model_cache[cache_key] = (model, tokenizer)
                return (model, tokenizer)
                
        except Exception as e:
            self.logger.error(f"Error loading translator model {model_name}: {e}")
            
            if self.inference_mode == InferenceMode.LOCAL:
                raise
                
        # Fall back to API if local loading failed or API mode is selected
        if self.inference_mode in [InferenceMode.API, InferenceMode.HYBRID]:
            api_model = self._create_api_translator_model(model_name)
            self._model_cache[cache_key] = api_model
            return api_model
            
        raise ValueError(f"Failed to load translator model: {model_name}")
        
    def get_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """
        Get embedding for the given text.
        
        Args:
            text: Text to embed
            model_name: Optional model name to use (default from config if None)
            
        Returns:
            Embedding vector
        """
        model = self.get_embedding_model(model_name)
        
        # Check if it's a local model or API wrapper
        if isinstance(model, SentenceTransformer):
            emb = model.encode(text, convert_to_tensor=False)
            return emb.tolist()
        else:
            # Assume it's our API wrapper
            return model.encode(text)
            
    def get_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model_name: Optional model name to use (default from config if None)
            
        Returns:
            List of embedding vectors
        """
        model = self.get_embedding_model(model_name)
        
        # Check if it's a local model or API wrapper
        if isinstance(model, SentenceTransformer):
            show_progress = self.config.get("ui.progress_bars", True) and len(texts) > 10
            embs = model.encode(texts, 
                               convert_to_tensor=False,
                               show_progress_bar=show_progress)
            return embs.tolist()
        else:
            # Assume it's our API wrapper
            return model.encode_batch(texts)
            
    def translate(self, text: str, source_lang: Optional[str] = None, 
                 target_lang: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Translate the given text.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default: auto-detect)
            target_lang: Target language code (required for some models)
            model_name: Optional model name to use (default from config if None)
            
        Returns:
            Translated text
        """
        model = self.get_translator_model(model_name)
        
        # Check if it's a local model or API wrapper
        if isinstance(model, tuple) and len(model) == 2:
            model_obj, tokenizer = model
            
            # Prepare input
            input_text = text
            if source_lang and target_lang and model_name:
                if "nllb" in model_name or "m2m100" in model_name:
                    # NLLB and M2M100 use special formatting
                    tokenizer.src_lang = source_lang
                    tokenizer.tgt_lang = target_lang
                elif "opus-mt" in model_name:
                    # These models might have the language pair in the name
                    pass
                
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = model_obj.generate(**inputs)
            
            # Decode translation
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
            
        else:
            # Assume it's our API wrapper
            return model.translate(text, source_lang, target_lang)
    
    def _create_api_embedding_model(self, model_name: str) -> Any:
        """
        Create an API wrapper for embedding models.
        
        Args:
            model_name: Model identifier
            
        Returns:
            API model wrapper
        """
        # This is a simplified placeholder for the API wrapper
        # In a real implementation, this would handle the API client more robustly
        api_key = self.config.get_api_key("huggingface")
        if not api_key:
            raise ValueError("Hugging Face API key not configured")
            
        class HuggingFaceApiEmbedding:
            def __init__(self, model_name, api_key, api_url, logger):
                self.model_name = model_name
                self.api_key = api_key
                self.api_url = api_url
                self.logger = logger
                self.timeout = 30
                
            def encode(self, text):
                url = f"{self.api_url}/{self.model_name}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                payload = {"inputs": text, "options": {"wait_for_model": True}}
                
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    self.logger.error(f"API error: {response.status_code} - {response.text}")
                    raise ValueError(f"API error: {response.status_code}")
                    
                return response.json()
                
            def encode_batch(self, texts):
                results = []
                for text in tqdm(texts, disable=not self.config.get("ui.progress_bars", True)):
                    results.append(self.encode(text))
                return results
                
        return HuggingFaceApiEmbedding(
            model_name,
            api_key, 
            self.config.get("api.huggingface.endpoint"),
            self.logger
        )
    
    def _create_api_translator_model(self, model_name: str) -> Any:
        """
        Create an API wrapper for translator models.
        
        Args:
            model_name: Model identifier
            
        Returns:
            API model wrapper
        """
        # This is a simplified placeholder for the API wrapper
        api_key = self.config.get_api_key("huggingface")
        if not api_key:
            raise ValueError("Hugging Face API key not configured")
            
        class HuggingFaceApiTranslator:
            def __init__(self, model_name, api_key, api_url, logger, config):
                self.model_name = model_name
                self.api_key = api_key
                self.api_url = api_url
                self.logger = logger
                self.config = config
                self.timeout = config.get("api.huggingface.timeout", 30)
                
            def translate(self, text, source_lang=None, target_lang=None):
                url = f"{self.api_url}/{self.model_name}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                payload = {"inputs": text}
                if source_lang and target_lang:
                    payload["parameters"] = {"src_lang": source_lang, "tgt_lang": target_lang}
                
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    self.logger.error(f"API error: {response.status_code} - {response.text}")
                    raise ValueError(f"API error: {response.status_code}")
                    
                return response.json()[0]["translation_text"]
                
        return HuggingFaceApiTranslator(
            model_name,
            api_key, 
            self.config.get("api.huggingface.endpoint"),
            self.logger,
            self.config
        )
        
    def get_groq_client(self) -> Any:
        """
        Get a Groq API client for LLM inference.
        
        Returns:
            Groq client for API calls
        """
        api_key = self.config.get_api_key("groq")
        if not api_key:
            raise ValueError("Groq API key not configured")
            
        # Here we'd normally use the Groq Python client
        # For now, we'll create a simple wrapper using requests
        class GroqClient:
            def __init__(self, api_key, api_url, logger, config):
                self.api_key = api_key
                self.api_url = api_url
                self.logger = logger
                self.config = config
                self.default_model = config.get("api.groq.model", "mixtral-8x7b-32768")
                self.timeout = config.get("api.groq.timeout", 30)
                
            def completion(self, prompt, model=None, temperature=0.7, max_tokens=1024):
                if model is None:
                    model = self.default_model
                    
                url = f"{self.api_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    self.logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    raise ValueError(f"Groq API error: {response.status_code}")
                    
                return response.json()["choices"][0]["message"]["content"]
                
        return GroqClient(
            api_key,
            self.config.get("api.groq.endpoint"),
            self.logger,
            self.config
        )
        
    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        self._model_cache.clear()
        self._api_cache.clear()
        
        # Force garbage collection to release memory
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Model cache cleared")

    def load_sentence_transformer_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a SentenceTransformer model by name.
        
        Args:
            model_name (str): The name of the model to load.
            
        Returns:
            SentenceTransformer: The loaded model.
        """
        # Check if model is already loaded
        cache_key = f"embedding_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Try to load model
        try:
            self.logger.info(f"Loading SentenceTransformer model: {model_name}")
            model = SentenceTransformer(model_name, cache_folder=str(self.models_dir))
            model.to(self.device)
            self._model_cache[cache_key] = model
            return model
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer model {model_name}: {e}")
            raise ValueError(f"Failed to load SentenceTransformer model: {model_name}")

class MultilingualModelManager:
    """Manager for specialized multilingual embedding models."""
    
    # Default multilingual model
    DEFAULT_MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Alternative models for different language needs
    MULTILINGUAL_MODELS = {
        "default": "paraphrase-multilingual-MiniLM-L12-v2",  # Balanced size/performance
        "high_quality": "paraphrase-multilingual-mpnet-base-v2",  # Higher quality, larger
        "efficient": "distiluse-base-multilingual-cased-v2"  # Smaller, faster
    }
    
    def __init__(self, config_manager, model_loader):
        """Initialize with config and model loader references."""
        self.config = config_manager
        self.model_loader = model_loader
        self.language_detector = LanguageDetector()
        self.loaded_models = {}
        # Initialize logger for this class
        self.logger = logging.getLogger("tqa.multilingual")
        
        # Initialize with default multilingual model
        self._load_default_model()
    
    def _load_default_model(self):
        """Load the default multilingual model."""
        model_name = self.MULTILINGUAL_MODELS["default"]
        try:
            model = self.model_loader.load_sentence_transformer_model(model_name)
            self.loaded_models["default"] = model
            self.logger.info(f"Loaded default multilingual model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load default multilingual model: {e}")
    
    def get_model(self, source_lang: str = None, target_lang: str = None) -> SentenceTransformer:
        """
        Get the best model for a language pair.
        
        Args:
            source_lang (str, optional): Source language code
            target_lang (str, optional): Target language code
            
        Returns:
            SentenceTransformer: Model for the language pair
        """
        # If no languages specified, return default model
        if not source_lang or not target_lang:
            return self.loaded_models.get("default")
            
        # Get optimized model recommendation
        model_name = self.language_detector.get_optimal_model(source_lang, target_lang)
        
        # Check if we already have this model loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Load the model if it's not already loaded
        try:
            model = self.model_loader.load_sentence_transformer_model(model_name)
            self.loaded_models[model_name] = model
            self.logger.info(f"Loaded model {model_name} for {source_lang}-{target_lang} pair")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            # Fallback to default
            return self.loaded_models.get("default")
    
    def get_supported_languages(self) -> List[Dict]:
        """Get list of supported languages for the embedding models."""
        return get_supported_languages()
    
    def detect_language(self, text: str) -> Dict:
        """
        Detect the language of a piece of text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Language detection results
        """
        return self.language_detector.detect(text) 