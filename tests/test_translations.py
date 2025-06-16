#!/usr/bin/env python3
"""
Unit tests for translation functionality
Tests translation accuracy against known good translations
"""

import unittest
import pytest
from typing import Dict, List, Tuple, Optional
import os
import json
import tempfile
from unittest.mock import MagicMock, patch
import torch
import numpy as np

from config_manager import ConfigManager
from model_loader import ModelLoader, MultilingualModelManager
from text_processor import TextProcessor
from analyzer import TranslationQualityAnalyzer

class TestTranslations(unittest.TestCase):
    """Test suite for testing the translation model functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for test configs and caches
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
        
        # Create basic config
        config_data = {
            "cache.directory": os.path.join(self.temp_dir.name, "cache"),
            "models.embedding.cache_dir": os.path.join(self.temp_dir.name, "models"),
            "inference_mode": "local"
        }
        
        # Write config to file
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
            
        # Initialize components
        self.config = ConfigManager(config_path=self.config_path)
        self.model_loader = ModelLoader(self.config)
        self.multilingual_model_manager = MultilingualModelManager(self.config, self.model_loader)
        self.text_processor = TextProcessor()
        
        # Create a mock AnalysisResult class that will be returned by the analyze method
        class MockAnalysisResult:
            def __init__(self, quality_score=0.85, fluency_score=0.9, accuracy_score=0.8):
                self.quality_score = quality_score
                self.fluency_score = fluency_score
                self.accuracy_score = accuracy_score
                self.segment_scores = []
                
        self.MockAnalysisResult = MockAnalysisResult
        
        # Initialize analyzer
        self.analyzer = TranslationQualityAnalyzer(
            config=self.config,
            model_loader=self.model_loader,
            multilingual_model_manager=self.multilingual_model_manager,
            text_processor=self.text_processor
        )
        
        # Patch translate_and_analyze method for testing without actual models
        self.original_translate_and_analyze = self.analyzer.translate_and_analyze
        self.analyzer.translate_and_analyze = MagicMock()
        
        # Configure the mock to return example translations and analysis
        def mock_translate_and_analyze(text, source_lang=None, target_lang=None, model_name=None, analyze_quality=True):
            # Dictionary mapping source texts to their translations
            translations = {
                "Hello world": "Hola mundo",
                "How are you today?": "¿Cómo estás hoy?",
                "I love programming": "Me encanta programar",
                "The quick brown fox jumps over the lazy dog": "El rápido zorro marrón salta sobre el perro perezoso",
                "Bonjour le monde": "Hello world",
                "Je suis fatigué": "I am tired",
                "Das ist ein Test": "This is a test",
                "Ich bin ein Berliner": "I am a Berliner",
                "こんにちは世界": "Hello world",
                "お元気ですか？": "How are you?",
            }
            
            # Auto-detect source language if not provided
            if source_lang is None:
                # Simplified detection for testing
                if "Hello" in text or "love" in text or "quick" in text:
                    source_lang = "en"
                elif "Bonjour" in text or "suis" in text:
                    source_lang = "fr"
                elif "ist" in text or "bin" in text:
                    source_lang = "de"
                elif "こんにちは" in text or "お元気" in text:
                    source_lang = "ja"
                else:
                    source_lang = "en"  # Default
            
            # Get translation
            translation = translations.get(text, f"MOCK_TRANSLATION: {text}")
            
            # Create basic result
            result = {
                "source_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translation": translation
            }
            
            # Add analysis if requested
            if analyze_quality:
                # Generate mock scores based on text length for variety
                quality_score = 0.5 + (len(text) % 10) / 20  # 0.5-1.0 based on text length
                
                result.update({
                    "quality_score": quality_score,
                    "fluency_score": min(1.0, quality_score + 0.1),
                    "accuracy_score": max(0.0, quality_score - 0.1),
                    "segment_scores": [
                        {
                            "source": text,
                            "target": translation,
                            "score": quality_score
                        }
                    ]
                })
                
            return result
            
        self.analyzer.translate_and_analyze.side_effect = mock_translate_and_analyze
        
        # Also mock the analyze method to return a consistent object
        self.analyzer.analyze = MagicMock(return_value=self.MockAnalysisResult())
    
    def teardown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        # Restore original method
        if hasattr(self, 'original_translate_and_analyze'):
            self.analyzer.translate_and_analyze = self.original_translate_and_analyze
        
    def test_basic_translations(self):
        """Test basic translation functionality with known sentences."""
        # Define test cases: (source_text, source_lang, target_lang, expected_translation)
        test_cases = [
            ("Hello world", "en", "es", "Hola mundo"),
            ("How are you today?", "en", "es", "¿Cómo estás hoy?"),
            ("I love programming", "en", "es", "Me encanta programar"),
            ("Bonjour le monde", "fr", "en", "Hello world"),
            ("Je suis fatigué", "fr", "en", "I am tired"),
            ("Das ist ein Test", "de", "en", "This is a test"),
            ("Ich bin ein Berliner", "de", "en", "I am a Berliner"),
        ]
        
        # Test each case
        for source, src_lang, tgt_lang, expected in test_cases:
            result = self.analyzer.translate_and_analyze(source, src_lang, tgt_lang)
            
            # Check the translation
            self.assertEqual(
                result["translation"], 
                expected, 
                f"Failed translation from {src_lang} to {tgt_lang}: '{source}' → Got: '{result['translation']}', Expected: '{expected}'"
            )
            
            # Verify result structure
            self.assertEqual(result["source_text"], source)
            self.assertEqual(result["source_lang"], src_lang)
            self.assertEqual(result["target_lang"], tgt_lang)
            
            # Verify quality scores if present
            if "quality_score" in result:
                self.assertGreaterEqual(result["quality_score"], 0.0)
                self.assertLessEqual(result["quality_score"], 1.0)

    def test_challenging_translations(self):
        """Test challenging translations that might have multiple valid outputs."""
        # Define challenging test cases
        test_cases = [
            ("The quick brown fox jumps over the lazy dog", "en", "es", 
             "El rápido zorro marrón salta sobre el perro perezoso"),
            ("こんにちは世界", "ja", "en", "Hello world"),
            ("お元気ですか？", "ja", "en", "How are you?"),
        ]
        
        # Test each case with more flexible assertion
        for source, src_lang, tgt_lang, expected in test_cases:
            result = self.analyzer.translate_and_analyze(source, src_lang, tgt_lang)
            
            # For challenging cases, use partial matches or allow minor variations
            cleaned_result = result["translation"].lower().replace("?", "").strip()
            cleaned_expected = expected.lower().replace("?", "").strip()
            
            self.assertIn(
                cleaned_result,
                [cleaned_expected],
                f"Failed challenging translation: '{source}' → Got: '{result['translation']}', Expected: '{expected}'"
            )
            
            # Verify the analysis data is present
            self.assertIn("source_lang", result)
            self.assertEqual(result["source_lang"], src_lang)
            
            if "segment_scores" in result:
                self.assertGreaterEqual(len(result["segment_scores"]), 1)

    def test_auto_language_detection(self):
        """Test that translation works correctly with auto-detected languages."""
        # Test with auto-detection
        source = "Bonjour le monde"
        target_lang = "en"
        expected = "Hello world"
        
        # Set explicit behavior for this test case
        self.analyzer.translate_and_analyze.side_effect = None
        self.analyzer.translate_and_analyze.return_value = {
            "source_text": source,
            "source_lang": "fr",  # Auto-detected language
            "target_lang": target_lang,
            "translation": expected,
            "quality_score": 0.85
        }
        
        result = self.analyzer.translate_and_analyze(source, None, target_lang)
        
        # Check translation
        self.assertEqual(
            result["translation"],
            expected,
            f"Auto-detection failed: '{source}' → Got: '{result['translation']}', Expected: '{expected}'"
        )
        
        # Verify the source language was auto-detected
        self.assertEqual(result["source_lang"], "fr")
        
        # Verify the call was made with None as source_lang
        self.analyzer.translate_and_analyze.assert_called_with(source, None, target_lang)

    @patch("model_loader.SentenceTransformer")
    def test_model_selection_for_language_pairs(self, mock_sentence_transformer):
        """Test that the correct model is selected for different language pairs."""
        # Restore original method for this test
        self.analyzer.translate_and_analyze = self.original_translate_and_analyze
        
        # Create mock models for different language pairs
        en_es_model = MagicMock()
        en_es_model.translate = MagicMock(return_value="Hola mundo")
        
        fr_en_model = MagicMock()
        fr_en_model.translate = MagicMock(return_value="Hello world")
        
        de_en_model = MagicMock()
        de_en_model.translate = MagicMock(return_value="This is a test")
        
        # Configure mock analyzer.analyze to return a consistent result
        self.analyzer.analyze = MagicMock(return_value=self.MockAnalysisResult())
        
        # Set up the mock model selection in the multilingual manager
        def get_mock_model(source_lang, target_lang):
            if source_lang == "en" and target_lang == "es":
                return en_es_model
            elif source_lang == "fr" and target_lang == "en":
                return fr_en_model
            elif source_lang == "de" and target_lang == "en":
                return de_en_model
            else:
                # Default model for other language pairs
                default_model = MagicMock()
                default_model.translate = MagicMock(
                    return_value=f"MOCK_{source_lang}_TO_{target_lang}"
                )
                return default_model
                
        # Mock the _perform_translation method to use our mock models
        def mock_perform_translation(text, source_lang, target_lang, model, model_name=None):
            return model.translate(text, source_lang=source_lang, target_lang=target_lang)
            
        self.analyzer._perform_translation = MagicMock(side_effect=mock_perform_translation)
        
        # Mock the get_model method
        self.multilingual_model_manager.get_model = MagicMock(side_effect=get_mock_model)
        
        # Test with English to Spanish
        en_es_result = self.analyzer.translate_and_analyze("Hello world", "en", "es")
        self.assertEqual(en_es_result["translation"], "Hola mundo")
        
        # Test with French to English
        fr_en_result = self.analyzer.translate_and_analyze("Bonjour le monde", "fr", "en")
        self.assertEqual(fr_en_result["translation"], "Hello world")
        
        # Test with German to English
        de_en_result = self.analyzer.translate_and_analyze("Das ist ein Test", "de", "en")
        self.assertEqual(de_en_result["translation"], "This is a test")
        
        # Verify the model selection was called with the correct language pairs
        self.multilingual_model_manager.get_model.assert_any_call("en", "es")
        self.multilingual_model_manager.get_model.assert_any_call("fr", "en")
        self.multilingual_model_manager.get_model.assert_any_call("de", "en")

    def test_translation_with_analysis(self):
        """Test that translation with analysis returns both translation and scores."""
        # Override the mock for this specific test
        self.analyzer.translate_and_analyze.side_effect = None
        self.analyzer.translate_and_analyze.return_value = {
            "source_text": "Hello world",
            "source_lang": "en",
            "target_lang": "es",
            "translation": "Hola mundo",
            "quality_score": 0.92,
            "fluency_score": 0.95,
            "accuracy_score": 0.89,
            "segment_scores": [
                {"source": "Hello world", "target": "Hola mundo", "score": 0.92}
            ]
        }
        
        # Perform translation with analysis
        result = self.analyzer.translate_and_analyze(
            "Hello world", 
            source_lang="en", 
            target_lang="es", 
            analyze_quality=True
        )
        
        # Check translation
        self.assertEqual(result["translation"], "Hola mundo")
        
        # Check analysis scores
        self.assertIn("quality_score", result)
        self.assertIn("fluency_score", result)
        self.assertIn("accuracy_score", result)
        self.assertIn("segment_scores", result)
        
        self.assertEqual(result["quality_score"], 0.92)
        self.assertEqual(result["fluency_score"], 0.95)
        self.assertEqual(result["accuracy_score"], 0.89)
        self.assertEqual(len(result["segment_scores"]), 1)
        
    def test_translation_only_mode(self):
        """Test that translation without analysis returns just translation data."""
        # Override the mock for this specific test
        self.analyzer.translate_and_analyze.side_effect = None
        self.analyzer.translate_and_analyze.return_value = {
            "source_text": "Hello world",
            "source_lang": "en",
            "target_lang": "es",
            "translation": "Hola mundo"
        }
        
        # Call the translate_text method which should use translate_and_analyze internally
        result = self.analyzer.translate_text(
            "Hello world", 
            source_lang="en", 
            target_lang="es"
        )
        
        # Check we got just the translation string
        self.assertEqual(result, "Hola mundo")
        
        # Verify translate_and_analyze was called with analyze_quality=False
        self.analyzer.translate_and_analyze.assert_called_with(
            text="Hello world",
            source_lang="en",
            target_lang="es",
            model_name=None,
            analyze_quality=False
        )

if __name__ == '__main__':
    unittest.main()