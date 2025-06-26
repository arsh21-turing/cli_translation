import unittest
import os
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
import sys
from io import StringIO
import random

# Import the module to test
from quality_learning_engine import QualityLearningEngine, FeedbackData
from config_manager import ConfigManager


class TestQualityLearningEngineFeedback(unittest.TestCase):
    """Test the feedback system in the QualityLearningEngine."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock config manager
        self.config_manager = MagicMock(spec=ConfigManager)
        self.config_manager.get.return_value = {}
        
        # Initialize the engine with test directories
        self.engine = QualityLearningEngine(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            config=self.config_manager
        )
        
        # Create sample feedback data
        self.sample_feedback = FeedbackData(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            similarity_score=0.85,
            groq_rating=0.78,
            combined_score=0.82,
            human_rating=0.80
        )
        
        # Create multiple feedback entries for testing batch operations
        self.feedback_batch = []
        languages = [("en", "fr"), ("en", "es"), ("de", "en"), ("fr", "es")]
        
        for i in range(20):
            lang_pair = languages[i % len(languages)]
            similarity = random.uniform(0.6, 0.95)
            groq = random.uniform(0.5, 0.9)
            combined = 0.6 * similarity + 0.4 * groq
            human = combined + random.uniform(-0.1, 0.1)  # Human rating close to combined but with some variance
            
            feedback = FeedbackData(
                source_text=f"Sample source text {i}",
                translation=f"Sample translation {i}",
                source_lang=lang_pair[0],
                target_lang=lang_pair[1],
                similarity_score=similarity,
                groq_rating=groq,
                combined_score=combined,
                human_rating=human if i % 5 != 0 else None  # Some entries without human rating
            )
            self.feedback_batch.append(feedback)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_feedback_data_creation(self):
        """Test creating FeedbackData objects."""
        # Test creating from constructor
        feedback = FeedbackData(
            source_text="Test",
            translation="Test translation",
            source_lang="en",
            target_lang="fr",
            similarity_score=0.9,
            groq_rating=0.85,
            combined_score=0.88
        )
        
        self.assertEqual(feedback.source_text, "Test")
        self.assertEqual(feedback.translation, "Test translation")
        self.assertEqual(feedback.source_lang, "en")
        self.assertEqual(feedback.target_lang, "fr")
        self.assertEqual(feedback.similarity_score, 0.9)
        self.assertEqual(feedback.groq_rating, 0.85)
        self.assertEqual(feedback.combined_score, 0.88)
        self.assertIsNone(feedback.human_rating)
        
        # Test converting to dict
        feedback_dict = feedback.to_dict()
        self.assertIsInstance(feedback_dict, dict)
        self.assertEqual(feedback_dict["source_text"], "Test")
        self.assertEqual(feedback_dict["similarity_score"], 0.9)
        
        # Test creating from dictionary
        dict_data = {
            "source_text": "From dict",
            "translation": "Dict translation",
            "source_lang": "de",
            "target_lang": "en",
            "similarity_score": 0.75,
            "groq_rating": 0.7,
            "combined_score": 0.73,
            "human_rating": 0.8
        }
        
        feedback_from_dict = FeedbackData(**dict_data)
        self.assertEqual(feedback_from_dict.source_text, "From dict")
        self.assertEqual(feedback_from_dict.human_rating, 0.8)
    
    def test_add_feedback(self):
        """Test adding feedback to the engine."""
        # Add single feedback entry
        result = self.engine.add_feedback(self.sample_feedback)
        
        self.assertTrue(result)
        self.assertEqual(len(self.engine.feedback_data), 1)
        self.assertEqual(self.engine.feedback_data[0]['source_text'], "Hello world")
        self.assertEqual(self.engine.feedback_data[0]['similarity_score'], 0.85)
        
        # Add feedback as dictionary
        dict_feedback = {
            "source_text": "Second test",
            "translation": "Zweiter Test",
            "source_lang": "en",
            "target_lang": "de",
            "similarity_score": 0.8,
            "groq_rating": 0.7,
            "combined_score": 0.76,
            "human_rating": 0.75
        }
        
        result = self.engine.add_feedback(dict_feedback)
        
        self.assertTrue(result)
        self.assertEqual(len(self.engine.feedback_data), 2)
        self.assertEqual(self.engine.feedback_data[1]['source_text'], "Second test")
    
    def test_save_load_feedback(self):
        """Test saving and loading feedback data."""
        # Add feedback and ensure it's saved
        self.engine.add_feedback(self.sample_feedback)
        
        # Check that the file exists
        feedback_path = self.output_dir / "feedback" / "feedback_data.json"
        self.assertTrue(feedback_path.exists())
        
        # Check file contents
        with open(feedback_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(len(saved_data), 1)
        self.assertEqual(saved_data[0]['source_text'], "Hello world")
        
        # Create a new engine instance and verify it loads the data
        new_engine = QualityLearningEngine(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            config=self.config_manager
        )
        
        self.assertEqual(len(new_engine.feedback_data), 1)
        self.assertEqual(new_engine.feedback_data[0]['source_text'], "Hello world")
    
    @patch('quality_learning_engine.QualityLearningEngine._update_from_feedback')
    def test_update_from_feedback_called(self, mock_update):
        """Test that _update_from_feedback is called when adding feedback."""
        # Add feedback
        self.engine.add_feedback(self.sample_feedback)
        
        # Verify that _update_from_feedback was called
        mock_update.assert_called_once()
    
    def test_adapt_thresholds_from_feedback(self):
        """Test adapting thresholds based on feedback."""
        # Set up initial thresholds
        self.engine.thresholds = {
            "similarity": {
                "excellent": 0.85,
                "good": 0.75,
                "acceptable": 0.65,
                "poor": 0.55,
                "very_poor": 0.0
            },
            "groq_rating": {
                "excellent": 0.85,
                "good": 0.75,
                "acceptable": 0.65,
                "poor": 0.55,
                "very_poor": 0.0
            }
        }
        
        # Add feedback batch
        for feedback in self.feedback_batch:
            self.engine.add_feedback(feedback)
        
        # Force threshold adaptation directly (normally called by add_feedback)
        df = pd.DataFrame(self.engine.feedback_data)
        self.engine._adapt_thresholds_from_feedback(df)
        
        # Verify thresholds were adapted
        self.assertNotEqual(self.engine.thresholds["similarity"]["excellent"], 0.85)
        self.assertNotEqual(self.engine.thresholds["groq_rating"]["excellent"], 0.85)
        
        # Verify logical ordering of thresholds
        for metric in ["similarity", "groq_rating"]:
            thresholds = self.engine.thresholds[metric]
            self.assertGreater(thresholds["excellent"], thresholds["good"])
            self.assertGreater(thresholds["good"], thresholds["acceptable"])
            self.assertGreater(thresholds["acceptable"], thresholds["poor"])
            self.assertGreater(thresholds["poor"], thresholds["very_poor"])
        
        # Verify config was updated
        self.config_manager.set.assert_called()
        self.config_manager.save_config.assert_called()
    
    def test_analyze_feedback_data(self):
        """Test analyzing feedback data."""
        # Add feedback batch
        for feedback in self.feedback_batch:
            self.engine.feedback_data.append(feedback.to_dict())
        
        # Run analysis
        analysis = self.engine.analyze_feedback_data()
        
        # Verify basic analysis metrics
        self.assertIn('total_feedback', analysis)
        self.assertEqual(analysis['total_feedback'], 20)
        
        self.assertIn('language_pairs', analysis)
        self.assertEqual(analysis['language_pairs'], 4)
        
        # Verify metric statistics
        for metric in ['similarity_score_mean', 'groq_rating_mean', 'combined_score_mean']:
            self.assertIn(metric, analysis)
            self.assertIsInstance(analysis[metric], float)
        
        # Verify correlations
        for corr in ['sim_human_corr', 'groq_human_corr', 'sim_groq_corr']:
            self.assertIn(corr, analysis)
            self.assertIsInstance(analysis[corr], float)
        
        # Verify language-specific analysis
        self.assertIn('language_specific', analysis)
        self.assertIsInstance(analysis['language_specific'], dict)
    
    def test_get_dynamic_thresholds(self):
        """Test getting dynamic thresholds for language pairs."""
        # Set up language-specific thresholds
        self.engine.language_specific_thresholds = {
            "en-fr": {
                "similarity": {
                    "excellent": 0.9,
                    "good": 0.8,
                    "acceptable": 0.7,
                    "poor": 0.6,
                    "very_poor": 0.0
                }
            }
        }
        
        # Test getting thresholds for language pair with specific settings
        thresholds_en_fr = self.engine.get_dynamic_thresholds("en", "fr")
        self.assertEqual(thresholds_en_fr["similarity"]["excellent"], 0.9)
        
        # Test getting thresholds for language pair without specific settings
        thresholds_en_es = self.engine.get_dynamic_thresholds("en", "es")
        self.assertNotEqual(thresholds_en_es["similarity"]["excellent"], 0.9)
    
    def test_weights_update_from_feedback(self):
        """Test updating weights based on feedback correlations."""
        # Create feedback with strong correlation between groq and human ratings
        feedback_batch = []
        for i in range(20):
            # High correlation between groq and human, lower for similarity
            groq = random.uniform(0.5, 0.9)
            human = groq + random.uniform(-0.05, 0.05)  # Human close to groq
            similarity = random.uniform(0.6, 0.95)  # Less correlated with human
            
            feedback = FeedbackData(
                source_text=f"Source text {i}",
                translation=f"Translation {i}",
                source_lang="en",
                target_lang="fr",
                similarity_score=similarity,
                groq_rating=groq,
                human_rating=human,
                combined_score=(similarity + groq) / 2
            )
            feedback_batch.append(feedback)
        
        # Add feedback to engine
        for feedback in feedback_batch:
            self.engine.add_feedback(feedback)
        
        # Calculate weights
        weights = self.engine.get_learned_scoring_weights()
        
        # Verify groq weight is higher than similarity weight
        self.assertGreater(weights['groq_weight'], weights['similarity_weight'])
        
        # Verify weights sum to 1.0
        self.assertAlmostEqual(weights['groq_weight'] + weights['similarity_weight'], 1.0)
    
    def test_empty_feedback_handling(self):
        """Test handling of empty feedback data."""
        # Clear feedback data
        self.engine.feedback_data = []
        
        # Run analysis
        analysis = self.engine.analyze_feedback_data()
        
        # Verify error message
        self.assertIn('error', analysis)
        self.assertEqual(analysis['error'], 'No feedback data available')
        
        # Verify empty feedback doesn't crash when calculating weights
        weights = self.engine.get_learned_scoring_weights()
        
        # Default weights should be returned
        self.assertEqual(weights['similarity_weight'], 0.6)
        self.assertEqual(weights['groq_weight'], 0.4)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_feedback_file_error_handling(self, mock_file):
        """Test handling of file errors when loading feedback."""
        # Mock file operation to raise an exception
        mock_file.side_effect = IOError("Test file error")
        
        # Attempt to load feedback
        result = self.engine._load_feedback_data()
        
        # Verify failure was handled gracefully
        self.assertFalse(result)
        self.assertEqual(self.engine.feedback_data, [])
    
    def test_feedback_with_segment_data(self):
        """Test feedback with segment-level alignment data."""
        # Create feedback with segment data
        segment_data = [
            {
                "source_segment": "Hello",
                "target_segment": "Hola",
                "similarity_score": 0.9,
                "groq_rating": 0.85
            },
            {
                "source_segment": "world",
                "target_segment": "mundo",
                "similarity_score": 0.8,
                "groq_rating": 0.75
            }
        ]
        
        feedback = FeedbackData(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            similarity_score=0.85,
            groq_rating=0.8,
            combined_score=0.83,
            human_rating=0.85,
            segment_level_data=segment_data
        )
        
        # Add feedback with segment data
        self.engine.add_feedback(feedback)
        
        # Verify segment data was saved
        self.assertEqual(len(self.engine.feedback_data), 1)
        self.assertIn('segment_level_data', self.engine.feedback_data[0])
        self.assertEqual(len(self.engine.feedback_data[0]['segment_level_data']), 2)
    
    def test_update_config_with_learned_weights(self):
        """Test that learned weights are applied to configuration."""
        # Add feedback to create correlations
        for i in range(10):
            feedback = FeedbackData(
                source_text=f"Source {i}",
                translation=f"Translation {i}",
                source_lang="en",
                target_lang="fr",
                similarity_score=0.8 + 0.01 * i,
                groq_rating=0.7 + 0.02 * i,
                combined_score=0.75 + 0.015 * i,
                human_rating=0.75 + 0.02 * i
            )
            self.engine.feedback_data.append(feedback.to_dict())
        
        # Create DataFrame for testing
        df = pd.DataFrame(self.engine.feedback_data)
        
        # Update weights based on feedback
        self.engine._update_from_feedback()
        
        # Verify config manager was called to update weights
        self.config_manager.set.assert_called()
        self.config_manager.save_config.assert_called()
    
    def test_thresholds_logical_consistency(self):
        """Test that thresholds remain logically consistent after adaptation."""
        # Create feedback that would make thresholds inconsistent if not fixed
        feedback_batch = []
        
        # Group 1: High similarity, lower human rating (excellent)
        for i in range(5):
            feedback = FeedbackData(
                source_text=f"Excellent source {i}",
                translation=f"Excellent translation {i}",
                source_lang="en",
                target_lang="fr",
                similarity_score=0.95,
                groq_rating=0.93,
                combined_score=0.94,
                human_rating=0.9
            )
            feedback_batch.append(feedback)
        
        # Group 2: Similar similarity, but rated as good
        for i in range(5):
            feedback = FeedbackData(
                source_text=f"Good source {i}",
                translation=f"Good translation {i}",
                source_lang="en",
                target_lang="fr",
                similarity_score=0.9,
                groq_rating=0.88,
                combined_score=0.89,
                human_rating=0.8
            )
            feedback_batch.append(feedback)
        
        # Group 3: Similar (inconsistent data that would make thresholds illogical)
        for i in range(5):
            feedback = FeedbackData(
                source_text=f"Poor source {i}",
                translation=f"Poor translation {i}",
                source_lang="en",
                target_lang="fr",
                similarity_score=0.92,  # Higher than "good" but rated poorly
                groq_rating=0.55,
                combined_score=0.7,
                human_rating=0.55
            )
            feedback_batch.append(feedback)
        
        # Add feedback
        for feedback in feedback_batch:
            self.engine.add_feedback(feedback)
        
        # Force threshold adaptation
        df = pd.DataFrame(self.engine.feedback_data)
        self.engine._adapt_thresholds_from_feedback(df)
        
        # Verify thresholds remain logically consistent
        for metric in ['similarity', 'combined_score', 'groq_rating']:
            if metric in self.engine.thresholds:
                thresholds = self.engine.thresholds[metric]
                self.assertGreater(thresholds["excellent"], thresholds["good"])
                self.assertGreater(thresholds["good"], thresholds["acceptable"])
                self.assertGreater(thresholds["acceptable"], thresholds["poor"])
                self.assertGreater(thresholds["poor"], thresholds["very_poor"])


class TestQualityLearningEngineIntegration(unittest.TestCase):
    """Integration tests for QualityLearningEngine with feedback system."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create sample batch data file
        sample_data = [
            {
                "source_text": "Hello world",
                "translation": "Hola mundo",
                "source_lang": "en",
                "target_lang": "es",
                "analysis": {
                    "similarity_score": 0.85,
                    "linguistic_assessment": {
                        "quality_score": 8.5,
                        "aspect_scores": {
                            "accuracy": 8.0,
                            "fluency": 9.0
                        }
                    },
                    "combined_score": 0.83
                }
            },
            {
                "source_text": "How are you?",
                "translation": "¿Cómo estás?",
                "source_lang": "en",
                "target_lang": "es",
                "analysis": {
                    "similarity_score": 0.82,
                    "linguistic_assessment": {
                        "quality_score": 9.0,
                        "aspect_scores": {
                            "accuracy": 8.5,
                            "fluency": 9.5
                        }
                    },
                    "combined_score": 0.86
                }
            }
        ]
        
        with open(self.data_dir / "sample_batch.processed", "w") as f:
            json.dump(sample_data, f)
        
        # Create a real config manager with minimal implementation
        class SimpleConfigManager:
            def __init__(self):
                self.config = {}
            
            def get(self, key, default=None):
                return self.config.get(key, default)
            
            def set(self, key, value):
                self.config[key] = value
            
            def save_config(self):
                pass
        
        self.config_manager = SimpleConfigManager()
        
        # Initialize the engine with test directories
        self.engine = QualityLearningEngine(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            config=self.config_manager
        )
        
        # Create feedback data
        self.feedback_entries = []
        for i in range(10):
            feedback = FeedbackData(
                source_text=f"Source {i}",
                translation=f"Translation {i}",
                source_lang="en",
                target_lang="es",
                similarity_score=0.75 + (i * 0.02),
                groq_rating=0.7 + (i * 0.03),
                combined_score=0.73 + (i * 0.025),
                human_rating=0.8 if i % 2 == 0 else None  # Half with human ratings
            )
            self.feedback_entries.append(feedback)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_full_learning_cycle_with_feedback(self):
        """Test a complete learning cycle that includes feedback data."""
        # Add feedback data
        for feedback in self.feedback_entries:
            self.engine.add_feedback(feedback)
        
        # Run data loading and extraction
        self.engine.load_batch_data()
        self.engine.extract_metrics_pairs()
        
        # Run correlation analysis
        correlations = self.engine.analyze_correlations()
        self.assertIsInstance(correlations, dict)
        
        # Build prediction model
        model_info = self.engine.build_prediction_model()
        self.assertIsInstance(model_info, dict)
        
        # Optimize thresholds
        thresholds = self.engine.optimize_thresholds()
        self.assertIsInstance(thresholds, dict)
        
        # Generate insights
        report_path = self.engine.generate_insights_report()
        self.assertTrue(os.path.exists(report_path))
        
        # Verify that feedback data was incorporated
        self.assertIn('feedback_analysis', self.engine.analyze_feedback_data())
        
        # Verify that model file was created
        model_files = list(Path(self.output_dir / "models").glob("*.joblib"))
        self.assertGreater(len(model_files), 0)
    
    def test_feedback_persistence_across_sessions(self):
        """Test that feedback persists across engine instances."""
        # Add feedback to first engine instance
        for feedback in self.feedback_entries[:5]:
            self.engine.add_feedback(feedback)
        
        # Create second engine instance
        engine2 = QualityLearningEngine(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            config=self.config_manager
        )
        
        # Verify feedback was loaded correctly
        self.assertEqual(len(engine2.feedback_data), 5)
        
        # Add more feedback to second instance
        for feedback in self.feedback_entries[5:]:
            engine2.add_feedback(feedback)
        
        # Create third engine instance
        engine3 = QualityLearningEngine(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            config=self.config_manager
        )
        
        # Verify all feedback was loaded
        self.assertEqual(len(engine3.feedback_data), 10)
        
        # Verify feedback data integrity
        source_texts = [entry['source_text'] for entry in engine3.feedback_data]
        self.assertIn("Source 0", source_texts)
        self.assertIn("Source 9", source_texts)
    
    def test_thresholds_evolution_over_time(self):
        """Test how thresholds evolve with more feedback over time."""
        # Initial thresholds
        self.engine.thresholds = {
            "similarity": {
                "excellent": 0.85,
                "good": 0.75,
                "acceptable": 0.65,
                "poor": 0.55,
                "very_poor": 0.0
            }
        }
        
        initial_thresholds = self.engine.thresholds["similarity"].copy()
        
        # Add first batch of feedback (5 entries)
        for feedback in self.feedback_entries[:5]:
            self.engine.add_feedback(feedback)
        
        # Record intermediate thresholds
        intermediate_thresholds = self.engine.thresholds["similarity"].copy()
        
        # Verify thresholds changed somewhat but not drastically
        for level in ["excellent", "good", "acceptable", "poor"]:
            self.assertNotEqual(intermediate_thresholds[level], initial_thresholds[level])
            # Change should be moderate since we only added 5 entries
            abs_diff = abs(intermediate_thresholds[level] - initial_thresholds[level])
            self.assertLess(abs_diff, 0.2)  # Less than 20% change
        
        # Add second batch of feedback (5 more entries)
        for feedback in self.feedback_entries[5:]:
            self.engine.add_feedback(feedback)
        
        # Record final thresholds
        final_thresholds = self.engine.thresholds["similarity"].copy()
        
        # Verify thresholds continued to evolve
        for level in ["excellent", "good", "acceptable", "poor"]:
            # Should be different from intermediate thresholds
            self.assertNotEqual(final_thresholds[level], intermediate_thresholds[level])
        
        # Verify threshold relationships are maintained
        self.assertGreater(final_thresholds["excellent"], final_thresholds["good"])
        self.assertGreater(final_thresholds["good"], final_thresholds["acceptable"])
        self.assertGreater(final_thresholds["acceptable"], final_thresholds["poor"])
        self.assertGreater(final_thresholds["poor"], final_thresholds["very_poor"])


if __name__ == '__main__':
    unittest.main() 