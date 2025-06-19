"""
Unit tests for ModelLoader class
Tests model loading, API integration, and inference functionality
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import numpy as np
import torch

# Import required classes
from model_loader import ModelLoader, ModelType, InferenceMode
from config_manager import ConfigManager

# --- Mock classes for testing ---

class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""
    def __init__(self, model_name, cache_folder=None):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = "cpu"
        
    def to(self, device):
        self.device = device
        return self
        
    def encode(self, texts, convert_to_tensor=False, show_progress_bar=False):
        """Return deterministic mock embeddings based on input."""
        if isinstance(texts, str):
            # Single text
            # Create a deterministic embedding based on text hash
            text_hash = hash(texts) % 100000
            return np.array([float(text_hash) / 100000] * 10)
        else:
            # Batch of texts
            return np.array([
                [float(hash(text) % 100000) / 100000] * 10
                for text in texts
            ])

class MockTranslatorModel:
    """Mock translator model for testing."""
    def __init__(self):
        self.device = "cpu"
        
    def to(self, device):
        self.device = device
        return self
        
    def generate(self, **kwargs):
        """Return mock generated tokens."""
        # Just return some fixed IDs that our mock tokenizer will handle
        return [[1, 2, 3, 4, 5]]

class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.src_lang = None
        self.tgt_lang = None
        
    def __call__(self, text, return_tensors="pt"):
        """Convert text to mock tensors."""
        return MagicMock(to=lambda device: MagicMock())
        
    def decode(self, tokens, skip_special_tokens=True):
        """Decode tokens to text."""
        if isinstance(tokens, list) and len(tokens) > 0:
            # Just return a mock translation that includes the model name
            # for verification in test assertions
            return f"Translated with {self.model_name}"
        return "Empty translation"

# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # Cleanup after tests
    shutil.rmtree(tmp_dir)

@pytest.fixture
def standard_mock_config(temp_dir):
    """Create a standard mock configuration with valid API keys."""
    config = MagicMock(spec=ConfigManager)
    
    # Set up a dictionary for configuration values for cleaner side_effect handling
    config_values = {
        "cache.directory": os.path.join(temp_dir, "cache"),
        "models.embedding.cache_dir": os.path.join(temp_dir, "models"),
        "api.huggingface.endpoint": "https://api-inference.huggingface.co/models",
        "api.groq.endpoint": "https://api.groq.com/v1",
        "api.groq.model": "mixtral-8x7b-32768",
        "ui.progress_bars": False
    }
    
    # Configure get method with dictionary lookup
    config.get.side_effect = lambda key, default=None: config_values.get(key, default)
    
    # Configure get_model_path
    model_paths = {
        "embedding": "sentence-transformers/test-embedding-model",
        "translator": "Helsinki-NLP/test-translator-model",
    }
    config.get_model_path.side_effect = lambda model_type="embedding": model_paths.get(model_type, "default-model")
    
    # Configure get_api_key with valid keys
    api_keys = {
        "huggingface": "test-hf-api-key",
        "groq": "test-groq-api-key"
    }
    config.get_api_key.side_effect = lambda service: api_keys.get(service, None)
    
    # Configure is_api_configured
    config.is_api_configured.side_effect = lambda service: api_keys.get(service) is not None
    
    return config

@pytest.fixture
def no_api_mock_config(temp_dir):
    """Create a mock configuration with no API keys."""
    config = MagicMock(spec=ConfigManager)
    
    # Set up a dictionary for configuration values
    config_values = {
        "cache.directory": os.path.join(temp_dir, "cache"),
        "models.embedding.cache_dir": os.path.join(temp_dir, "models"),
        "api.huggingface.endpoint": "https://api-inference.huggingface.co/models",
        "api.groq.endpoint": "https://api.groq.com/v1",
        "api.groq.model": "mixtral-8x7b-32768",
        "ui.progress_bars": False
    }
    
    # Configure get method with dictionary lookup
    config.get.side_effect = lambda key, default=None: config_values.get(key, default)
    
    # Configure get_model_path
    model_paths = {
        "embedding": "sentence-transformers/test-embedding-model",
        "translator": "Helsinki-NLP/test-translator-model",
    }
    config.get_model_path.side_effect = lambda model_type="embedding": model_paths.get(model_type, "default-model")
    
    # Configure get_api_key to return None (no API keys)
    config.get_api_key.return_value = None
    
    # Configure is_api_configured to return False
    config.is_api_configured.return_value = False
    
    return config

@pytest.fixture
def model_loader(standard_mock_config):
    """Create a ModelLoader instance with mocks for testing."""
    with patch('model_loader.SentenceTransformer', MockSentenceTransformer), \
         patch('model_loader.AutoModel'), \
         patch('model_loader.AutoModelForSeq2SeqLM'), \
         patch('model_loader.AutoTokenizer'):
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.HYBRID)
        yield loader

# --- Test classes ---

class TestModelLoader:
    """Test suite for ModelLoader class."""
    
    # --- Initialization tests ---
    
    def test_initialization(self, model_loader, temp_dir):
        """Test model loader initialization."""
        # Check inference mode
        assert model_loader.inference_mode == InferenceMode.HYBRID
        
        # Check directory creation
        cache_dir = Path(os.path.join(temp_dir, "cache"))
        models_dir = Path(os.path.join(temp_dir, "models"))
        assert cache_dir.exists()
        assert models_dir.exists()
        
        # Check device setup
        assert model_loader.device == "cuda" or model_loader.device == "cpu"
        
    def test_inference_mode_string_conversion(self, standard_mock_config):
        """Test string to enum conversion for inference mode."""
        with patch('model_loader.SentenceTransformer', MockSentenceTransformer):
            # Test valid mode
            loader = ModelLoader(standard_mock_config, inference_mode="local")
            assert loader.inference_mode == InferenceMode.LOCAL
            
            # Test invalid mode (should default to HYBRID)
            loader = ModelLoader(standard_mock_config, inference_mode="invalid_mode")
            assert loader.inference_mode == InferenceMode.HYBRID
            
    # --- Embedding model tests ---
    
    @patch('model_loader.SentenceTransformer', MockSentenceTransformer)
    def test_get_embedding_model_local(self, standard_mock_config):
        """Test getting an embedding model in LOCAL mode."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Get default model
        model = loader.get_embedding_model()
        
        # Check model properties
        assert isinstance(model, MockSentenceTransformer)
        assert model.model_name == standard_mock_config.get_model_path()
        
        # Check model is cached
        cache_key = f"embedding_{standard_mock_config.get_model_path()}"
        assert cache_key in loader._model_cache
        assert loader._model_cache[cache_key] is model
        
    @patch('model_loader.SentenceTransformer', side_effect=Exception("Model not found"))
    def test_get_embedding_model_local_fallback(self, mock_st, standard_mock_config):
        """Test getting an embedding model with fallback to API."""
        # Only test with HYBRID mode to check fallback
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.HYBRID)
        
        # Mock requests for API wrapper
        with patch('model_loader.requests.post') as mock_post:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [0.1] * 10
            mock_post.return_value = mock_response
            
            # Get model (should fall back to API)
            model = loader.get_embedding_model()
            
            # Check model is not a SentenceTransformer (should be API wrapper)
            assert not isinstance(model, MockSentenceTransformer)
            
            # Check API wrapper was used
            assert hasattr(model, 'encode')
            
    @patch('model_loader.SentenceTransformer', side_effect=Exception("Model not found"))
    def test_get_embedding_model_api(self, mock_st, standard_mock_config):
        """Test getting an embedding model in API mode."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.API)
        
        # Mock requests for API wrapper
        with patch('model_loader.requests.post') as mock_post:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [0.1] * 10
            mock_post.return_value = mock_response
            
            # Get model
            model = loader.get_embedding_model()
            
            # Check model is API wrapper
            assert hasattr(model, 'encode')
            
            # Test using the model
            embedding = model.encode("test text")
            assert len(embedding) == 10
            
    @patch('model_loader.SentenceTransformer', MockSentenceTransformer)
    def test_get_specific_embedding_model(self, standard_mock_config):
        """Test getting a specific embedding model."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Get specific model
        specific_model_name = "sentence-transformers/specific-model"
        model = loader.get_embedding_model(specific_model_name)
        
        # Check model properties
        assert isinstance(model, MockSentenceTransformer)
        assert model.model_name == specific_model_name
        
        # Check model is cached
        cache_key = f"embedding_{specific_model_name}"
        assert cache_key in loader._model_cache
        
    @patch('model_loader.SentenceTransformer', MockSentenceTransformer)
    def test_get_embedding(self, standard_mock_config):
        """Test getting a single text embedding."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Get embedding
        text = "This is a test text"
        embedding = loader.get_embedding(text)
        
        # Check embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 10  # MockSentenceTransformer returns 10-dim embeddings
        
    @patch('model_loader.SentenceTransformer', MockSentenceTransformer)
    def test_get_embeddings_batch(self, standard_mock_config):
        """Test getting embeddings for multiple texts."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Get embeddings for multiple texts
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = loader.get_embeddings(texts)
        
        # Check embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 10 for emb in embeddings)
        
    # --- Translator model tests ---
    
    @patch('model_loader.AutoModelForSeq2SeqLM')
    @patch('model_loader.AutoTokenizer')
    def test_get_translator_model_local(self, mock_tokenizer, mock_model_class, standard_mock_config):
        """Test getting a translator model in LOCAL mode."""
        # Set up mocks
        mock_model = MockTranslatorModel()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MockTokenizer("test-model")
        
        # Create model loader
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Get translator model
        model_tuple = loader.get_translator_model()
        
        # Check model and tokenizer
        assert isinstance(model_tuple, tuple)
        assert len(model_tuple) == 2
        model, tokenizer = model_tuple
        
        # Check model is our mock
        assert model is mock_model
        
        # Check model is cached
        model_name = standard_mock_config.get_model_path("translator")
        cache_key = f"translator_{model_name}"
        assert cache_key in loader._model_cache
        
    @patch('model_loader.AutoModelForSeq2SeqLM', side_effect=Exception("Model not found"))
    @patch('model_loader.AutoTokenizer', side_effect=Exception("Tokenizer not found"))
    def test_get_translator_model_api(self, mock_tokenizer, mock_model_class, standard_mock_config):
        """Test getting a translator model in API mode."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.API)
        
        # Mock requests for API wrapper
        with patch('model_loader.requests.post') as mock_post:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{"translation_text": "Translated text"}]
            mock_post.return_value = mock_response
            
            # Get model
            model = loader.get_translator_model()
            
            # Check model is API wrapper
            assert hasattr(model, 'translate')
            
            # Test translation
            translation = model.translate("test text", "en", "fr")
            assert translation == "Translated text"
            
    @patch('model_loader.AutoModelForSeq2SeqLM')
    @patch('model_loader.AutoTokenizer')
    def test_translate(self, mock_tokenizer, mock_model_class, standard_mock_config):
        """Test text translation functionality."""
        # Set up mocks
        mock_model = MockTranslatorModel()
        mock_model_class.from_pretrained.return_value = mock_model
        
        model_name = "Helsinki-NLP/test-translator-model"
        mock_tokenizer.from_pretrained.return_value = MockTokenizer(model_name)
        
        # Create model loader
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Translate text
        text = "This is a test text"
        translation = loader.translate(text, "en", "fr", model_name)
        
        # Check translation contains expected model name
        assert model_name in translation
        
        # Test with model_name=None to ensure no null pointer exception
        translation = loader.translate(text, "en", "fr")
        assert model_name in translation
        
        # Test with only source_lang
        translation = loader.translate(text, source_lang="en")
        assert model_name in translation
        
        # Test with only target_lang
        translation = loader.translate(text, target_lang="fr")
        assert model_name in translation
        
        # Test with no language parameters
        translation = loader.translate(text)
        assert model_name in translation
            
    # --- Groq API tests ---
    
    def test_get_groq_client(self, model_loader):
        """Test getting a Groq API client."""
        with patch('model_loader.requests.post') as mock_post:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Generated text"}}]
            }
            mock_post.return_value = mock_response
            
            # Get Groq client
            groq_client = model_loader.get_groq_client()
            
            # Check client
            assert groq_client is not None
            assert hasattr(groq_client, 'completion')
            
            # Test completion
            response = groq_client.completion("Test prompt")
            assert response == "Generated text"
            
            # Check API endpoint was called correctly
            mock_post.assert_called_once()
            # Check it was called with the correct URL
            assert mock_post.call_args[0][0].endswith("/chat/completions")
            
    def test_groq_client_no_api_key(self, no_api_mock_config):
        """Test error when Groq API key is not configured."""
        # Create model loader with the no-API config
        loader = ModelLoader(no_api_mock_config)
        
        with pytest.raises(ValueError, match="Groq API key not configured"):
            loader.get_groq_client()
            
    # --- Cache and resource management tests ---
    
    @patch('model_loader.SentenceTransformer', MockSentenceTransformer)
    def test_model_cache(self, standard_mock_config):
        """Test model caching functionality."""
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Get the same model twice
        model1 = loader.get_embedding_model()
        model2 = loader.get_embedding_model()
        
        # Check models are the same instance
        assert model1 is model2
        
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_clear_cache(self, mock_gc_collect, mock_cuda_empty_cache, model_loader):
        """Test cache clearing functionality."""
        # Fill cache
        model_loader._model_cache["test_key"] = "test_value"
        model_loader._api_cache["test_api"] = "test_api_value"
        
        # Clear cache
        model_loader.clear_cache()
        
        # Check cache is empty
        assert not model_loader._model_cache
        assert not model_loader._api_cache
        
        # Check GC was called
        mock_gc_collect.assert_called_once()
        
        # Check CUDA cache was cleared if available
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            mock_cuda_empty_cache.assert_called_once()
            
    @patch('model_loader.SentenceTransformer', side_effect=Exception("Model not available"))
    def test_local_failure_api_disabled(self, mock_st, standard_mock_config):
        """Test handling of local model failure when API is disabled."""
        # Create model loader with LOCAL mode
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.LOCAL)
        
        # Attempt to get model should raise an exception
        with pytest.raises(Exception, match="Model not available"):
            loader.get_embedding_model()
            
    # --- Edge case tests ---
    
    @patch('model_loader.requests.post')
    def test_api_error_handling(self, mock_post, standard_mock_config):
        """Test handling of API errors."""
        # Create model loader with API mode
        loader = ModelLoader(standard_mock_config, inference_mode=InferenceMode.API)
        
        # Set up API to return an error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Service unavailable"
        mock_post.return_value = mock_response
        
        # Getting embedding should raise an error
        with pytest.raises(ValueError, match="API error: 500"):
            loader._create_api_embedding_model("test-model").encode("test text")
            
    def test_api_no_api_key(self, no_api_mock_config):
        """Test error when API key is not configured."""
        # Create model loader with API mode using the no-API config
        loader = ModelLoader(no_api_mock_config, inference_mode=InferenceMode.API)
        
        # Getting embedding model should raise an error
        with pytest.raises(ValueError, match="Hugging Face API key not configured"):
            loader.get_embedding_model()