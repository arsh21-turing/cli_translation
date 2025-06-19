# tests/test_sliding_window_alignment.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the module to test - assuming it exists in a module called segment_alignment
from segment_alignment import SlidingWindowAligner


class TestSlidingWindowAligner:
    """Tests for the SlidingWindowAligner class."""

    @pytest.fixture
    def mock_embedding_generator(self):
        """Create a mock embedding generator that returns predictable embeddings."""
        generator = MagicMock()

        # For identical texts, return identical embeddings
        def mock_generate_embedding(text):
            # Simple deterministic embedding based on the text content
            # Same text will produce same embedding
            if text == "Hello world":
                return np.array([0.1, 0.2, 0.3])
            elif text == "This is a test":
                return np.array([0.4, 0.5, 0.6])
            elif text == "Another example":
                return np.array([0.7, 0.8, 0.9])
            else:
                # Hash the text to a consistent embedding
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()[:3]  # Take first 3 bytes
                # Convert to float array in 0-1 range
                return np.array([b / 255.0 for b in hash_bytes])

        generator.generate_embedding.side_effect = mock_generate_embedding
        return generator

    @pytest.fixture
    def aligner(self, mock_embedding_generator):
        """Create a SlidingWindowAligner instance with the mock embedding generator."""
        return SlidingWindowAligner(embedding_generator=mock_embedding_generator)

    def test_perfect_alignment_identical_texts(self, aligner):
        """Test that identical texts get a perfect alignment score."""
        source_text = "Hello world. This is a test. Another example."
        translation = "Hello world. This is a test. Another example."

        result = aligner.calculate_alignment_score(source_text, translation)

        # The score should be very close to 1.0 for identical texts
        assert result['alignment_score'] > 0.99
        assert result['alignment_quality'] == 'excellent'

        # Check segment count
        assert len(result['segment_alignments']) == 3

        # All segments should have near-perfect similarity
        for alignment in result['segment_alignments']:
            assert alignment['similarity'] > 0.99

    def test_perfect_alignment_different_order(self, aligner):
        """Test that perfectly aligned texts with different segment order still get high scores."""
        source_text = "Hello world. This is a test. Another example."
        translation = "This is a test. Hello world. Another example."

        result = aligner.calculate_alignment_score(source_text, translation)

        # The score should still be high but not perfect due to order change
        assert result['alignment_score'] >= 0.9

        # Check segment count
        assert len(result['segment_alignments']) == 3

        # Either the segments should be matched to their counterparts or have high similarity
        alignments = {a['source']: a['target'] for a in result['segment_alignments']}

        # At least one segment should be correctly aligned
        correctly_aligned = False
        for source, target in alignments.items():
            if source == "Another example." and target == "Another example.":
                correctly_aligned = True
                break

        assert correctly_aligned

    def test_alignment_with_missing_segment(self, aligner):
        """Test alignment when a segment is missing from the translation."""
        source_text = "Hello world. This is a test. Another example."
        translation = "Hello world. Another example."

        result = aligner.calculate_alignment_score(source_text, translation)

        # The score should be lower but still decent
        assert result['alignment_score'] < 0.9
        assert result['alignment_score'] > 0.6

        # There should be a penalty for the missing segment
        assert result['missing_segments'] == 1

    def test_alignment_with_extra_segment(self, aligner):
        """Test alignment when the translation has an extra segment."""
        source_text = "Hello world. Another example."
        translation = "Hello world. This is a test. Another example."

        result = aligner.calculate_alignment_score(source_text, translation)

        # The score should be lower but still decent
        assert result['alignment_score'] < 0.9
        assert result['alignment_score'] > 0.6

        # There should be a penalty for the extra segment
        assert result['extra_segments'] == 1

    def test_alignment_with_partial_match(self, aligner):
        """Test alignment when segments are similar but not identical."""
        source_text = "Hello world. This is a test. Another example."
        translation = "Hello world! This is just a test. One more example."

        with patch('segment_alignment.cosine_similarity') as mock_similarity:
            # Mock similarity for similar but not identical texts
            mock_similarity.side_effect = lambda x, y: 0.85

            result = aligner.calculate_alignment_score(source_text, translation)

        # The score should reflect partial matches
        assert result['alignment_score'] >= 0.8
        assert result['alignment_score'] < 0.95

        # All segments should be found but with lower similarity
        for alignment in result['segment_alignments']:
            assert alignment['similarity'] == 0.85

    def test_completely_different_texts(self, aligner):
        """Test alignment between completely different texts."""
        source_text = "Hello world. This is a test. Another example."
        translation = "The quick brown fox. Jumps over the lazy dog. Something else entirely."

        with patch('segment_alignment.cosine_similarity') as mock_similarity:
            # Mock very low similarity for different texts
            mock_similarity.side_effect = lambda x, y: 0.2

            result = aligner.calculate_alignment_score(source_text, translation)

        # The score should be very low
        assert result['alignment_score'] < 0.3
        assert result['alignment_quality'] in ['poor', 'critical']

    def test_large_text_performance(self, aligner):
        """Test that the algorithm scales reasonably with larger texts."""
        # Create larger texts with repeated segments
        source_segments = ["Segment " + str(i) + "." for i in range(50)]
        source_text = " ".join(source_segments)

        # Create a translation with the same segments but 10 percent modified
        translation_segments = source_segments.copy()
        import random
        random.seed(42)  # For reproducibility
        for _ in range(5):  # Modify five segments
            idx = random.randint(0, 49)
            translation_segments[idx] = "Modified " + translation_segments[idx]

        translation_text = " ".join(translation_segments)

        # Time the alignment calculation
        import time
        start_time = time.time()
        result = aligner.calculate_alignment_score(source_text, translation_text)
        end_time = time.time()

        # The score should be high despite the modifications
        assert result['alignment_score'] > 0.8

        # The calculation should be reasonably fast adjust threshold as needed
        assert end_time - start_time < 2.0  # Should complete in less than two seconds

    def test_different_languages(self, aligner):
        """Test alignment between different languages."""
        source_text = "Hello world. This is a test. Another example."
        translation = "Hola mundo. Esta es una prueba. Otro ejemplo."

        # For cross lingual testing mock the cosine similarity function
        # to return reasonable values for translated equivalents
        with patch('segment_alignment.cosine_similarity') as mock_similarity:
            # Define a custom similarity function for this test
            def custom_similarity(vec1, vec2):
                # Map specific segments to their expected similarity scores
                if "Hello world" in vec1.mock_text and "Hola mundo" in vec2.mock_text:
                    return 0.85
                elif "This is a test" in vec1.mock_text and "Esta es una prueba" in vec2.mock_text:
                    return 0.82
                elif "Another example" in vec1.mock_text and "Otro ejemplo" in vec2.mock_text:
                    return 0.87
                else:
                    return 0.5  # Default similarity

            # Add mock_text attribute to identify vectors in the custom similarity function
            def mock_generate_embedding(text):
                embedding = MagicMock()
                embedding.mock_text = text
                return embedding

            aligner.embedding_generator.generate_embedding.side_effect = mock_generate_embedding
            mock_similarity.side_effect = custom_similarity

            result = aligner.calculate_alignment_score(source_text, translation)

        # The score should be high for well translated content
        assert result['alignment_score'] > 0.8
        assert result['alignment_quality'] in ['good', 'excellent']

    def test_sliding_window_effect(self, aligner):
        """Test that the sliding window approach properly handles non exact segment matches."""
        source_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        # Translation splits and merges some sentences
        translation = "First sentence revised. Second and third sentences combined. Fourth sentence."

        # Mock the cosine similarity function to handle the sliding window
        with patch('segment_alignment.cosine_similarity') as mock_similarity:
            # Define segment specific similarities
            similarity_map = {
                ("First sentence.", "First sentence revised."): 0.9,
                ("Second sentence. Third sentence.", "Second and third sentences combined."): 0.85,
                ("Third sentence.", "Second and third sentences combined."): 0.7,
                ("Second sentence.", "Second and third sentences combined."): 0.7,
                ("Fourth sentence.", "Fourth sentence."): 0.95,
            }

            def similarity_function(vec1, vec2):
                # In a real implementation this would compare the embeddings
                # Here we look up predefined similarities based on text
                for (src, tgt), score in similarity_map.items():
                    if src in vec1.mock_text and tgt in vec2.mock_text:
                        return score
                return 0.3  # Default low similarity for non matching segments

            # Add mock_text attribute to identify vectors
            def mock_generate_embedding(text):
                embedding = MagicMock()
                embedding.mock_text = text
                return embedding

            aligner.embedding_generator.generate_embedding.side_effect = mock_generate_embedding
            mock_similarity.side_effect = similarity_function

            result = aligner.calculate_alignment_score(source_text, translation)

        # The alignment should handle the split merged sentences intelligently
        assert result['alignment_score'] >= 0.8

        # The number of matched segments should be appropriate
        assert len(result['segment_alignments']) >= 3

    def test_segment_alignments_format(self, aligner):
        """Test the format of returned segment alignments."""
        source_text = "Hello world. This is a test."
        translation = "Hello world. This is a test."

        result = aligner.calculate_alignment_score(source_text, translation)

        # Check the structure of the segment alignments
        assert 'segment_alignments' in result
        assert isinstance(result['segment_alignments'], list)

        for alignment in result['segment_alignments']:
            assert 'source' in alignment
            assert 'target' in alignment
            assert 'similarity' in alignment
            assert 0 <= alignment['similarity'] <= 1

    def test_alignment_quality_mapping(self, aligner):
        """Test that alignment scores are correctly mapped to quality labels."""
        # Mock the core alignment calculation to return controlled scores
        with patch.object(aligner, '_calculate_raw_alignment', autospec=True) as mock_calc:
            test_cases = [
                (0.95, 'excellent'),
                (0.85, 'good'),
                (0.75, 'acceptable'),
                (0.65, 'fair'),
                (0.55, 'poor'),
                (0.35, 'critical')
            ]

            for score, expected_quality in test_cases:
                mock_calc.return_value = {
                    'raw_score': score,
                    'segment_alignments': [],
                    'extra_segments': 0,
                    'missing_segments': 0,
                    'unaligned_segments': 0
                }

                result = aligner.calculate_alignment_score("Test.", "Test.")
                assert result['alignment_quality'] == expected_quality, f"Score {score} should map to {expected_quality}" 