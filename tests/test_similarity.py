import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import time

from similarity_calculator import SimilarityCalculator, SimilarityMetric
from config_manager import ConfigManager

class TestSimilarityCalculator:
    """Tests for the SimilarityCalculator class."""
    
    @pytest.fixture
    def setup_components(self):
        """Set up necessary components for testing."""
        # Create mock objects
        config = MagicMock()
        config.get.return_value = 0.75  # Default threshold
        
        vector_generator = MagicMock()
        
        # Mock vector generation to return predictable vectors
        def mock_generate_vectors(texts, **kwargs):
            # Generate predictable vectors based on text length
            result = np.zeros((len(texts), 384))
            for i, text in enumerate(texts):
                # Use text length to influence vector (for testing)
                base = np.ones(384) * (len(text) % 10) / 10
                # Add some randomness but keep it deterministic
                seed = sum(ord(c) for c in text[:5]) if text else 0
                np.random.seed(seed)
                noise = np.random.random(384) * 0.1
                result[i] = base + noise
            return result
            
        def mock_cross_lingual_vectors(source_texts, target_texts, **kwargs):
            # Similar to above but returns tuple of source and target vectors
            source_vectors = mock_generate_vectors(source_texts)
            target_vectors = mock_generate_vectors(target_texts)
            return source_vectors, target_vectors
        
        vector_generator.generate_vectors.side_effect = mock_generate_vectors
        vector_generator.generate_cross_lingual_vectors.side_effect = mock_cross_lingual_vectors
        
        similarity_calculator = SimilarityCalculator(vector_generator, config)
        
        return {
            'config': config,
            'vector_generator': vector_generator,
            'similarity_calculator': similarity_calculator
        }
    
    def test_calculate_similarity(self, setup_components):
        """Test basic similarity calculation."""
        comps = setup_components
        calculator = comps['similarity_calculator']
        
        # Test with identical text
        identical_text = "This is a test sentence."
        similarity = calculator.calculate_similarity(identical_text, identical_text)
        assert np.isclose(similarity, 1.0)  # Should be exactly 1.0 for identical text
        
        # Test with similar texts
        similar_text1 = "This is a test sentence about similarity."
        similar_text2 = "This sentence tests similarity concepts."
        similarity = calculator.calculate_similarity(similar_text1, similar_text2)
        assert 0 <= similarity <= 1  # Should be between 0 and 1
        
        # Test with different metrics
        for metric in [
            SimilarityMetric.COSINE, 
            SimilarityMetric.EUCLIDEAN, 
            SimilarityMetric.DOT
        ]:
            similarity = calculator.calculate_similarity(
                similar_text1, similar_text2, metric=metric
            )
            assert 0 <= similarity <= 1  # All metrics should produce values in range [0,1]
    
    def test_pairwise_similarity(self, setup_components):
        """Test pairwise similarity calculation."""
        comps = setup_components
        calculator = comps['similarity_calculator']
        
        # Create test texts
        texts1 = [
            "This is the first text.",
            "Here is another example.",
            "A third sample for testing."
        ]
        
        texts2 = [
            "The first text is here.",
            "Another example follows.",
            "Testing with a third sample.",
            "Extra text not in first list."
        ]
        
        # Calculate similarity matrix
        similarity_matrix = calculator.calculate_pairwise_similarity(texts1, texts2)
        
        # Check shape
        assert similarity_matrix.shape == (3, 4)
        
        # Check values are in range
        assert np.all(similarity_matrix >= 0)
        assert np.all(similarity_matrix <= 1)
    
    def test_find_most_similar(self, setup_components):
        """Test finding most similar texts."""
        comps = setup_components
        calculator = comps['similarity_calculator']
        
        # Create query and candidates
        query = "Natural language processing is fascinating."
        candidates = [
            "I find natural language processing very interesting.",
            "Machine learning is a broad field of AI.",
            "NLP and computational linguistics explore language.",
            "Data science includes statistics and programming."
        ]
        
        # Find most similar
        results = calculator.find_most_similar(query, candidates)
        
        # Check structure
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'index' in results[0]
        assert 'text' in results[0]
        assert 'similarity' in results[0]
        
        # Results should be sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i]['similarity'] >= results[i+1]['similarity']
        
        # Test with top_k
        limited = calculator.find_most_similar(query, candidates, top_k=2)
        assert len(limited) <= 2
        
        # Test with threshold
        high_threshold = calculator.find_most_similar(query, candidates, threshold=0.9)
        assert all(r['similarity'] >= 0.9 for r in high_threshold)
    
    def test_classify_semantic_match(self, setup_components):
        """Test semantic match classification."""
        comps = setup_components
        calculator = comps['similarity_calculator']
        
        # Define the test examples with expected classes
        test_cases = [
            # Identical text -> "exact"
            ("This is a test sentence.", "This is a test sentence.", "exact"),
            
            # Very minor changes -> "high"
            ("Testing semantic similarity.", "Testing the semantic similarity.", "high"),
            
            # Similar topic but different wording -> "moderate"
            ("The cat sat on the mat.", "A feline was resting on a rug.", "moderate"),
            
            # Vaguely related -> "low"
            ("Python is a programming language.", "Snakes are reptiles.", "low"),
            
            # Unrelated -> "unrelated"
            ("The car is blue.", "Pizza tastes good.", "unrelated")
        ]
        
        # Override the similarity calculation to provide predictable values for testing
        def mock_calculate_similarity(text1, text2, **kwargs):
            # Return different values based on test case
            if text1 == text2:
                return 1.0  # Exact match
            elif text1.startswith("Testing") and text2.startswith("Testing"):
                return 0.85  # High similarity
            elif "cat" in text1 and "feline" in text2:
                return 0.65  # Moderate similarity
            elif "Python" in text1 and "Snakes" in text2:
                return 0.45  # Low similarity
            else:
                return 0.2  # Unrelated
        
        calculator.calculate_similarity = mock_calculate_similarity
        
        # Test each case
        for text1, text2, expected_class in test_cases:
            result = calculator.classify_semantic_match(text1, text2)
            assert result == expected_class, f"Expected {expected_class} for '{text1}' and '{text2}'" 