import pytest
import json
import unittest.mock as mock
from unittest.mock import MagicMock, patch

# Import the modules to test
from groq_evaluator import GroqEvaluator
from groq_client import GroqClient
from config_manager import ConfigManager

@pytest.fixture
def mock_groq_client():
    """Create a mock GroqClient for testing."""
    client = MagicMock(spec=GroqClient)
    
    # Set up basic mock responses
    client.generate_completion.return_value = {
        "text": "Score: 8.5 - The translation is accurate and fluent.",
        "model": "llama3-8b-8192"
    }
    
    client.generate_chat_completion.return_value = {
        "content": json.dumps({
            "accuracy": 9,
            "accuracy_comments": "The translation captures the meaning well",
            "fluency": 8,
            "fluency_comments": "The translation reads naturally",
            "terminology": 9,
            "terminology_comments": "Terms are appropriately translated",
            "style": 8,
            "style_comments": "Style matches the original",
            "overall_score": 8.5,
            "summary": "High quality translation with minor issues",
            "errors": []
        }),
        "model": "llama3-8b-8192"
    }
    
    return client

@pytest.fixture
def mock_config_manager():
    """Create a mock ConfigManager for testing."""
    config = MagicMock(spec=ConfigManager)
    config.get_groq_model.return_value = "llama3-8b-8192"
    config.get_groq_api_key.return_value = "mock-api-key"
    return config

@pytest.fixture
def evaluator(mock_groq_client):
    """Create a GroqEvaluator instance with the mock client."""
    return GroqEvaluator(client=mock_groq_client)

@pytest.fixture
def evaluator_with_config(mock_groq_client, mock_config_manager):
    """Create a GroqEvaluator instance with the mock client and config manager."""
    return GroqEvaluator(client=mock_groq_client, config_manager=mock_config_manager)


class TestGroqEvaluator:
    """Test cases for the GroqEvaluator class."""
    
    def test_init_with_client(self, mock_groq_client):
        """Test initialization with provided client."""
        evaluator = GroqEvaluator(client=mock_groq_client)
        assert evaluator.client == mock_groq_client
    
    def test_init_with_config(self, mock_config_manager):
        """Test initialization with config manager."""
        with patch('groq_evaluator.GroqClient') as mock_client_class:
            evaluator = GroqEvaluator(config_manager=mock_config_manager)
            
            # Check if GroqClient was initialized with correct parameters
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args['config_manager'] == mock_config_manager
    
    def test_init_with_no_parameters(self):
        """Test initialization with no parameters."""
        with patch('groq_evaluator.GroqClient') as mock_client_class:
            evaluator = GroqEvaluator()
            # Should create a client with defaults
            mock_client_class.assert_called_once()
    
    def test_evaluate_translation_simple(self, evaluator, mock_groq_client):
        """Test simple translation evaluation."""
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            detailed=False
        )
        
        # Check the result
        assert result["overall_score"] == 8.5
        assert "The translation is accurate and fluent" in result["summary"]
        assert "detailed" in result and result["detailed"] is False
        
        # Verify client was called with correct parameters
        mock_groq_client.generate_completion.assert_called_once()
        call_args = mock_groq_client.generate_completion.call_args[1]
        assert source_text in call_args["prompt"]
        assert translation in call_args["prompt"]
        assert call_args["temperature"] == 0.3  # Default temperature
    
    def test_evaluate_translation_with_languages(self, evaluator, mock_groq_client):
        """Test translation evaluation with specified languages."""
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        source_lang = "en"
        target_lang = "es"
        
        # Call the method to test
        result = evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang,
            detailed=False
        )
        
        # Verify client was called with correct parameters
        mock_groq_client.generate_completion.assert_called_once()
        call_args = mock_groq_client.generate_completion.call_args[1]
        prompt = call_args["prompt"]
        
        # Check if languages are included in the prompt
        assert source_lang in prompt
        assert target_lang in prompt
    
    def test_evaluate_translation_detailed(self, evaluator, mock_groq_client):
        """Test detailed translation evaluation."""
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            detailed=True
        )
        
        # Check the result
        assert result["overall_score"] == 8.5
        assert result["accuracy"] == 9
        assert result["fluency"] == 8
        assert result["terminology"] == 9
        assert result["style"] == 8
        assert "detailed" in result and result["detailed"] is True
        assert "accuracy_comments" in result
        assert "fluency_comments" in result
        
        # Verify client was called with correct parameters
        mock_groq_client.generate_chat_completion.assert_called_once()
        call_args = mock_groq_client.generate_chat_completion.call_args[1]
        
        # Check messages format
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["role"] == "user"
        assert source_text in call_args["messages"][1]["content"]
        assert translation in call_args["messages"][1]["content"]
        assert "JSON" in call_args["messages"][1]["content"]
    
    def test_evaluate_translation_api_error(self, evaluator, mock_groq_client):
        """Test handling of API errors."""
        # Configure mock to return an error
        mock_groq_client.generate_completion.return_value = {
            "error": "API rate limit exceeded",
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            detailed=False
        )
        
        # Check the result has the error and a default score
        assert "error" in result
        assert result["overall_score"] == 0
    
    def test_evaluate_translation_parse_error(self, evaluator, mock_groq_client):
        """Test handling of response parsing errors."""
        # Configure mock to return an unparseable response
        mock_groq_client.generate_completion.return_value = {
            "text": "This is not in the expected format",
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            detailed=False
        )
        
        # Check the result has raw response
        assert "raw_response" in result
        assert result["overall_score"] == 0
    
    def test_detailed_evaluation_parse_error(self, evaluator, mock_groq_client):
        """Test handling of JSON parsing errors in detailed evaluation."""
        # Configure mock to return an invalid JSON
        mock_groq_client.generate_chat_completion.return_value = {
            "content": "This is not valid JSON",
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            detailed=True
        )
        
        # Check error handling
        assert "raw_response" in result
        assert result["overall_score"] == 0
    
    def test_compare_translations(self, evaluator, mock_groq_client):
        """Test comparing multiple translations."""
        # Configure mock for comparison response
        mock_groq_client.generate_chat_completion.return_value = {
            "content": json.dumps({
                "rankings": [
                    {
                        "rank": 1,
                        "translation_index": 2,
                        "score": 9.0,
                        "strengths": ["Accurate", "Fluent"],
                        "weaknesses": [],
                        "comments": "Best translation overall"
                    },
                    {
                        "rank": 2,
                        "translation_index": 1,
                        "score": 7.5,
                        "strengths": ["Accurate"],
                        "weaknesses": ["Less fluent"],
                        "comments": "Good but less fluent"
                    }
                ],
                "comparison_summary": "Translation 2 is better in fluency while maintaining accuracy."
            }),
            "model": "llama3-8b-8192"
        }
        
        # Test data
        source_text = "Hello world"
        translations = ["Hola mundo", "Hola a todos"]
        
        # Call the method to test
        result = evaluator.compare_translations(
            source_text=source_text,
            translations=translations
        )
        
        # Check the result
        assert "rankings" in result
        assert len(result["rankings"]) == 2
        assert result["rankings"][0]["rank"] == 1
        assert result["rankings"][0]["translation_index"] == 2
        assert result["rankings"][0]["score"] == 9.0
        assert "comparison_summary" in result
        
        # Verify client was called with correct parameters
        mock_groq_client.generate_chat_completion.assert_called_once()
        call_args = mock_groq_client.generate_chat_completion.call_args[1]
        content = call_args["messages"][1]["content"]
        
        # Check content contains all translations
        assert source_text in content
        assert "Translation 1: Hola mundo" in content
        assert "Translation 2: Hola a todos" in content
    
    def test_compare_translations_too_few(self, evaluator):
        """Test handling when too few translations are provided."""
        # Test data
        source_text = "Hello world"
        translations = ["Hola mundo"]  # Only one translation
        
        # Call the method to test
        result = evaluator.compare_translations(
            source_text=source_text,
            translations=translations
        )
        
        # Check error is returned
        assert "error" in result
        assert "Need at least 2 translations" in result["error"]
    
    def test_compare_translations_api_error(self, evaluator, mock_groq_client):
        """Test handling of API errors during comparison."""
        # Configure mock to return an error
        mock_groq_client.generate_chat_completion.return_value = {
            "error": "API rate limit exceeded",
        }
        
        # Test data
        source_text = "Hello world"
        translations = ["Hola mundo", "Hola a todos"]
        
        # Call the method to test
        result = evaluator.compare_translations(
            source_text=source_text,
            translations=translations
        )
        
        # Check the result has the error
        assert "error" in result
    
    def test_compare_translations_parse_error(self, evaluator, mock_groq_client):
        """Test handling of JSON parsing errors in comparison."""
        # Configure mock to return invalid JSON
        mock_groq_client.generate_chat_completion.return_value = {
            "content": "This is not valid JSON",
        }
        
        # Test data
        source_text = "Hello world"
        translations = ["Hola mundo", "Hola a todos"]
        
        # Call the method to test
        result = evaluator.compare_translations(
            source_text=source_text,
            translations=translations
        )
        
        # Check error handling
        assert "error" in result
        assert "raw_response" in result
    
    def test_analyze_translation_errors(self, evaluator, mock_groq_client):
        """Test translation error analysis."""
        # Configure mock for error analysis response
        mock_groq_client.generate_chat_completion.return_value = {
            "content": json.dumps({
                "errors": [
                    {
                        "segment": "mundo",
                        "error_type": "terminology",
                        "description": "Incorrect term for context",
                        "suggestion": "universo",
                        "severity": "minor"
                    }
                ],
                "error_summary": {
                    "meaning_errors": 0,
                    "language_errors": 0,
                    "terminology_errors": 1,
                    "style_errors": 0,
                    "total_errors": 1,
                    "major_errors": 0,
                    "minor_errors": 1
                },
                "overall_assessment": "Good translation with minor terminology issue"
            }),
            "model": "llama3-8b-8192"
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.analyze_translation_errors(
            source_text=source_text,
            translation=translation
        )
        
        # Check the result
        assert "errors" in result
        assert len(result["errors"]) == 1
        assert result["errors"][0]["error_type"] == "terminology"
        assert "error_summary" in result
        assert result["error_summary"]["total_errors"] == 1
        assert "overall_assessment" in result
        
        # Verify client was called with correct parameters
        mock_groq_client.generate_chat_completion.assert_called_once()
        call_args = mock_groq_client.generate_chat_completion.call_args[1]
        content = call_args["messages"][1]["content"]
        
        # Check content contains the texts and required analysis instructions
        assert source_text in content
        assert translation in content
        assert "error analyst" in call_args["messages"][0]["content"].lower()
    
    def test_analyze_translation_errors_with_languages(self, evaluator, mock_groq_client):
        """Test error analysis with language specifications."""
        # Configure mock for response
        mock_groq_client.generate_chat_completion.return_value = {
            "content": json.dumps({
                "errors": [],
                "error_summary": {
                    "meaning_errors": 0,
                    "language_errors": 0,
                    "terminology_errors": 0,
                    "style_errors": 0,
                    "total_errors": 0,
                    "major_errors": 0,
                    "minor_errors": 0
                },
                "overall_assessment": "Perfect translation with no errors"
            }),
            "model": "llama3-8b-8192"
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        source_lang = "en"
        target_lang = "es"
        
        # Call the method to test
        result = evaluator.analyze_translation_errors(
            source_text=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Verify client was called with language information
        mock_groq_client.generate_chat_completion.assert_called_once()
        call_args = mock_groq_client.generate_chat_completion.call_args[1]
        content = call_args["messages"][1]["content"]
        
        # Check if languages are included in the prompt
        assert source_lang in content
        assert target_lang in content
    
    def test_analyze_translation_errors_api_error(self, evaluator, mock_groq_client):
        """Test handling of API errors during error analysis."""
        # Configure mock to return an error
        mock_groq_client.generate_chat_completion.return_value = {
            "error": "API rate limit exceeded",
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.analyze_translation_errors(
            source_text=source_text,
            translation=translation
        )
        
        # Check the result has the error
        assert "error" in result
    
    def test_analyze_translation_errors_parse_error(self, evaluator, mock_groq_client):
        """Test handling of JSON parsing errors in error analysis."""
        # Configure mock to return invalid JSON
        mock_groq_client.generate_chat_completion.return_value = {
            "content": "This is not valid JSON",
        }
        
        # Test data
        source_text = "Hello world"
        translation = "Hola mundo"
        
        # Call the method to test
        result = evaluator.analyze_translation_errors(
            source_text=source_text,
            translation=translation
        )
        
        # Check error handling
        assert "error" in result
        assert "raw_response" in result


class TestGroqEvaluatorIntegration:
    """Integration tests for GroqEvaluator with TranslationQualityAnalyzer."""
    
    @pytest.fixture
    def mock_translation_analyzer(self, mock_groq_client, mock_config_manager):
        """Create a mock TranslationQualityAnalyzer for integration testing."""
        with patch('translation_quality_analyzer.TranslationQualityAnalyzer') as mock_analyzer_class:
            from translation_quality_analyzer import TranslationQualityAnalyzer
            
            # Create evaluator
            evaluator = GroqEvaluator(client=mock_groq_client, config_manager=mock_config_manager)
            
            # Create analyzer with evaluator
            analyzer = TranslationQualityAnalyzer(
                groq_evaluator=evaluator,
                config_manager=mock_config_manager
            )
            
            return analyzer
    
    def test_integration_with_quality_analyzer(self, mock_translation_analyzer, mock_groq_client):
        """Test integration with TranslationQualityAnalyzer."""
        # This is a simple integration test to ensure the evaluator can be used with the analyzer
        # The actual functionality is tested in test_translation_quality_analyzer.py
        
        # Set up mock embedding generator and basic response
        mock_translation_analyzer.embedding_generator.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        with patch('similarity_calculator.cosine_similarity', return_value=0.85):
            # Call analyze_pair
            result = mock_translation_analyzer.analyze_pair(
                source_text="Hello world",
                translation="Hola mundo",
                use_groq=True
            )
            
            # Ensure the GroqEvaluator is present and the mock client is wired
            assert hasattr(mock_translation_analyzer, "groq_evaluator")
            assert mock_translation_analyzer.groq_evaluator is not None 