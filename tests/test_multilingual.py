import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import os
import tempfile

# Import components to test
from config_manager import ConfigManager
from language_utils import LanguageDetector, get_supported_languages
from model_loader import MultilingualModelManager
from embedding_generator import MultilingualEmbeddingGenerator

class TestMultilingual:
    """Tests for multilingual embedding functionality."""
    
    @pytest.fixture
    def setup_components(self):
        """Set up necessary components for testing."""
        # Create a temporary config
        config = {
            'model_cache_dir': tempfile.mkdtemp(),
            'embedding_cache_dir': tempfile.mkdtemp(),
            'inference_mode': 'LOCAL'
        }
        
        # Create mock objects
        config_manager = MagicMock()
        config_manager.get.side_effect = lambda key, default=None: config.get(key, default)
        
        model_loader = MagicMock()
        mock_model = MagicMock()
        # Mock encoding method to return fake embeddings
        mock_model.encode.side_effect = lambda texts, **kwargs: np.random.rand(len(texts) if isinstance(texts, list) else 1, 384)
        model_loader.load_sentence_transformer_model.return_value = mock_model
        
        text_processor = MagicMock()
        
        multilingual_manager = MultilingualModelManager(config_manager, model_loader)
        
        # Replace the actual model loading with mock
        multilingual_manager.loaded_models = {"default": mock_model}
        multilingual_manager.get_model = MagicMock(return_value=mock_model)
        
        embedding_generator = MultilingualEmbeddingGenerator(
            config_manager, multilingual_manager, text_processor)
        
        # Return all mocked components
        return {
            'config_manager': config_manager,
            'model_loader': model_loader,
            'text_processor': text_processor,
            'multilingual_manager': multilingual_manager,
            'embedding_generator': embedding_generator,
            'mock_model': mock_model
        }
    
    def test_language_detection(self):
        """Test the language detection functionality."""
        detector = LanguageDetector()
        
        # Test English detection
        english_text = "This is a sample of English text for language detection testing."
        result = detector.detect(english_text)
        assert result['code'] == 'en'
        assert result['name'] == 'English'
        assert result['family'] == 'germanic'
        
        # Test Spanish detection
        spanish_text = "Este es un ejemplo de texto en español para probar la detección de idiomas."
        result = detector.detect(spanish_text)
        assert result['code'] == 'es'
        assert result['name'] == 'Spanish'
        assert result['family'] == 'romance'
        
        # Test with very short text
        short_text = "Hello"
        result = detector.detect(short_text, min_length=10)
        assert 'code' in result and isinstance(result['code'], str)
        assert 'name' in result and isinstance(result['name'], str)
        assert result['code'] != 'und'
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = get_supported_languages()
        
        # Verify some common languages are included
        language_codes = [lang['code'] for lang in languages]
        assert 'en' in language_codes
        assert 'es' in language_codes
        assert 'fr' in language_codes
        
        # Verify structure
        for lang in languages[:5]:  # Check first 5
            assert 'code' in lang
            assert 'name' in lang
            assert isinstance(lang['code'], str)
            assert isinstance(lang['name'], str)
    
    def test_multilingual_model_manager(self, setup_components):
        """Test the multilingual model manager."""
        components = setup_components
        manager = components['multilingual_manager']
        
        # Test getting model for same language
        model = manager.get_model('en', 'en')
        assert model is not None
        
        # Test getting model for different languages
        model = manager.get_model('en', 'fr')
        assert model is not None
        
        # Verify model loading was called
        components['model_loader'].load_sentence_transformer_model.assert_called()
    
    def test_multilingual_embeddings(self, setup_components):
        """Test generating multilingual embeddings."""
        components = setup_components
        generator = components['embedding_generator']
        
        # Test for single language
        english_text = ["This is a test sentence in English."]
        embeddings = generator.generate_embeddings(english_text, lang='en')
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(english_text)
        
        # Test cross-lingual embeddings
        english_text = ["This is an English sentence."]
        spanish_text = ["Esta es una frase en español."]
        
        source_emb, target_emb = generator.generate_cross_lingual_embeddings(
            english_text, spanish_text, source_lang='en', target_lang='es')
            
        assert source_emb.shape[0] == len(english_text)
        assert target_emb.shape[0] == len(spanish_text)
    
    def test_similarity_calculation(self, setup_components):
        """Test similarity calculation between embeddings."""
        components = setup_components
        generator = components['embedding_generator']
        
        # Create some test embeddings
        emb1 = np.array([0.1, 0.2, 0.3, 0.4])
        emb2 = np.array([0.2, 0.3, 0.4, 0.5])
        
        # Test cosine similarity
        cosine_sim = generator.calculate_similarity(emb1, emb2, method='cosine')
        assert 0 <= cosine_sim <= 1
        
        # Test euclidean similarity
        euclidean_sim = generator.calculate_similarity(emb1, emb2, method='euclidean')
        assert 0 <= euclidean_sim <= 1
        
        # Test dot product
        dot_sim = generator.calculate_similarity(emb1, emb2, method='dot')
        assert isinstance(dot_sim, float) 