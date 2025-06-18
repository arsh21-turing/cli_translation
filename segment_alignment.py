from __future__ import annotations

"""segment_alignment.py
Light-weight utilities for segment-level similarity analysis and recurring-pattern
extraction.  They are **not** meant to be perfect linguistic tools – the goal is
quick signal on where semantic alignment repeatedly drifts.

Usage example
-------------
>>> analyzer = SegmentAlignmentAnalyzer()
>>> res = analyzer.analyze_segment_alignment(src_text, tgt_text)

For higher-level consumer use :class:`WeakAlignmentDetector` which will also
pull Groq insights when available.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
import logging
import math
import itertools

# from similarity_calculator import cosine_similarity  # local util

logger = logging.getLogger(__name__)

try:
    from groq_evaluator import GroqEvaluator  # type: ignore
except ImportError:  # pragma: no cover
    GroqEvaluator = None  # type: ignore

# -----------------------------------------------------------------------------
# Helper – simple sentence/paragraph splitter
# -----------------------------------------------------------------------------

def _sentence_split(text: str) -> List[str]:
    # naïve regex – fine for quick analysis
    segs = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in segs if s.strip()]


def _paragraph_split(text: str) -> List[str]:
    segs = re.split(r"\n\s*\n", text.strip())
    return [s.strip() for s in segs if s.strip()]

# -----------------------------------------------------------------------------
# Core analyser
# -----------------------------------------------------------------------------

class SegmentAlignmentAnalyzer:
    """
    Analyzes translation alignment at the segment level to detect patterns of weakness
    and recurring similarity issues across a text.
    """
    
    def __init__(self, embedding_generator=None, similarity_threshold=0.75, 
                 segment_type='sentence', min_pattern_occurrences=2):
        """
        Initialize the segment alignment analyzer.
        
        Args:
            embedding_generator: EmbeddingGenerator instance for generating embeddings
            similarity_threshold: Threshold below which segments are considered weak
            segment_type: Type of segmentation to use ('sentence', 'paragraph', or 'hybrid')
            min_pattern_occurrences: Minimum occurrences needed to consider a pattern recurring
        """
        if embedding_generator is None:
            from embedding_generator import EmbeddingGenerator
            self.embedding_generator = EmbeddingGenerator()
        else:
            self.embedding_generator = embedding_generator
            
        self.similarity_threshold = similarity_threshold
        self.segment_type = segment_type
        self.min_pattern_occurrences = min_pattern_occurrences
        
    def segment_text(self, text: str) -> List[str]:
        """
        Segment text based on the configured segmentation type.
        
        Args:
            text: Text to segment
            
        Returns:
            List of text segments
        """
        if self.segment_type == 'sentence':
            # Simple sentence segmentation
            segments = re.split(r'(?<=[.!?])\s+', text.strip())
            # Filter out empty segments
            segments = [s.strip() for s in segments if s.strip()]
            return segments
            
        elif self.segment_type == 'paragraph':
            # Paragraph segmentation
            segments = re.split(r'\n\s*\n', text.strip())
            segments = [s.strip() for s in segments if s.strip()]
            return segments
            
        elif self.segment_type == 'hybrid':
            # First split by paragraph, then by sentence for long paragraphs
            paragraphs = re.split(r'\n\s*\n', text.strip())
            segments = []
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # For short paragraphs, keep them intact
                if len(para) < 200:  # Threshold can be adjusted
                    segments.append(para)
                else:
                    # Split long paragraphs into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    segments.extend([s.strip() for s in sentences if s.strip()])
            
            return segments
            
        else:
            # Default to sentence segmentation
            segments = re.split(r'(?<=[.!?])\s+', text.strip())
            segments = [s.strip() for s in segments if s.strip()]
            return segments
    
    def analyze_segment_alignment(self, source_text: str, translation: str) -> Dict[str, Any]:
        """
        Analyze alignment between source and translation at segment level.
        
        Args:
            source_text: Source text
            translation: Translated text
            
        Returns:
            Dictionary with segment alignment analysis
        """
        # Segment both texts
        source_segments = self.segment_text(source_text)
        translation_segments = self.segment_text(translation)
        
        # Check if we have major segment count mismatch
        segment_count_ratio = len(translation_segments) / max(1, len(source_segments))
        has_segment_mismatch = abs(1.0 - segment_count_ratio) > 0.3  # 30% threshold for mismatch
        
        # Calculate similarities and identify weak segments
        segment_similarities = []
        weak_segments = []
        
        # Choose analysis strategy based on segment count match
        if has_segment_mismatch or abs(len(source_segments) - len(translation_segments)) > 3:
            # For significant mismatches, use a different approach
            segment_similarities, weak_segments = self._analyze_misaligned_segments(
                source_segments, translation_segments
            )
        else:
            # For roughly aligned segments, do direct comparison
            for i, (src_seg, tgt_seg) in enumerate(zip(source_segments, translation_segments[:len(source_segments)])):
                if not src_seg or not tgt_seg:
                    continue
                    
                src_embedding = self.embedding_generator.generate_embedding(src_seg)
                tgt_embedding = self.embedding_generator.generate_embedding(tgt_seg)
                
                similarity = self._calculate_similarity(src_embedding, tgt_embedding)
                
                segment_similarities.append({
                    'index': i,
                    'source_segment': src_seg,
                    'translation_segment': tgt_seg,
                    'similarity': similarity
                })
                
                if similarity < self.similarity_threshold:
                    weak_segments.append({
                        'index': i,
                        'source_segment': src_seg,
                        'translation_segment': tgt_seg,
                        'similarity': similarity
                    })
        
        # Find patterns in weak segments
        patterns = self._detect_alignment_patterns(weak_segments, source_segments, translation_segments)
        
        # Calculate segment-level statistics
        similarity_values = [item['similarity'] for item in segment_similarities]
        avg_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0
        std_similarity = np.std(similarity_values) if len(similarity_values) > 1 else 0
        min_similarity = min(similarity_values) if similarity_values else 0
        max_similarity = max(similarity_values) if similarity_values else 0
        
        # Detect overall similarity trend (improving, degrading, or consistent)
        similarity_trend = 'consistent'
        if len(similarity_values) >= 3:
            # Calculate slope of linear regression
            x = np.arange(len(similarity_values))
            slope, _ = np.polyfit(x, similarity_values, 1)
            
            if slope > 0.05:  # Threshold can be adjusted
                similarity_trend = 'improving'
            elif slope < -0.05:
                similarity_trend = 'degrading'
        
        # Prepare result
        result = {
            'segment_count': {
                'source': len(source_segments),
                'translation': len(translation_segments),
                'ratio': segment_count_ratio,
                'mismatch': has_segment_mismatch
            },
            'similarity_stats': {
                'average': avg_similarity,
                'std_dev': std_similarity,
                'min': min_similarity,
                'max': max_similarity,
                'trend': similarity_trend
            },
            'weak_segments': weak_segments,
            'recurring_patterns': patterns,
            'all_segment_similarities': segment_similarities,
        }
        
        return result
    
    def _calculate_similarity(self, src_embedding: np.ndarray, tgt_embedding: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            src_embedding: Embedding for source segment
            tgt_embedding: Embedding for target segment
            
        Returns:
            Similarity score between 0 and 1
        """
        return cosine_similarity(src_embedding, tgt_embedding)
    
    def _analyze_misaligned_segments(self, source_segments: List[str], 
                                     translation_segments: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze segments when there's a significant count mismatch.
        Uses a sliding window approach to find best alignments.
        
        Args:
            source_segments: List of source text segments
            translation_segments: List of translation text segments
            
        Returns:
            Tuple of (segment_similarities, weak_segments)
        """
        segment_similarities = []
        weak_segments = []
        
        # If either list is empty, return empty results
        if not source_segments or not translation_segments:
            return segment_similarities, weak_segments
        
        # Create embeddings for all segments (for efficiency)
        src_embeddings = [self.embedding_generator.generate_embedding(seg) for seg in source_segments]
        tgt_embeddings = [self.embedding_generator.generate_embedding(seg) for seg in translation_segments]
        
        # Dynamic programming approach to find optimal alignment
        # This is similar to sequence alignment in bioinformatics
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((len(source_segments), len(translation_segments)))
        
        # Fill similarity matrix
        for i, src_emb in enumerate(src_embeddings):
            for j, tgt_emb in enumerate(tgt_embeddings):
                similarity_matrix[i, j] = self._calculate_similarity(src_emb, tgt_emb)
        
        # Find alignment using dynamic programming
        aligned_indices = self._find_optimal_alignment(similarity_matrix)
        
        # Convert alignment to segment similarities
        for src_idx, tgt_idx in aligned_indices:
            similarity = similarity_matrix[src_idx, tgt_idx]
            
            segment_similarities.append({
                'source_index': src_idx,
                'translation_index': tgt_idx,
                'source_segment': source_segments[src_idx],
                'translation_segment': translation_segments[tgt_idx],
                'similarity': similarity
            })
            
            if similarity < self.similarity_threshold:
                weak_segments.append({
                    'source_index': src_idx,
                    'translation_index': tgt_idx,
                    'source_segment': source_segments[src_idx],
                    'translation_segment': translation_segments[tgt_idx],
                    'similarity': similarity
                })
        
        return segment_similarities, weak_segments
    
    def _find_optimal_alignment(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find optimal alignment of segments using the similarity matrix.
        Uses a greedy approach that iteratively selects highest similarity pairs.
        
        Args:
            similarity_matrix: Matrix of similarity scores between segments
            
        Returns:
            List of (source_index, translation_index) tuples representing alignment
        """
        # Make a copy of the matrix to work with
        sim_matrix = similarity_matrix.copy()
        
        aligned_indices = []
        n_rows, n_cols = sim_matrix.shape
        
        # Iteratively find best matches
        while sim_matrix.size > 0 and np.max(sim_matrix) > 0:
            # Find indices of maximum similarity
            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            
            # Add to aligned indices
            aligned_indices.append((i, j))
            
            # Remove the row and column to prevent reusing these segments
            sim_matrix[i, :] = -1
            sim_matrix[:, j] = -1
        
        # Convert to original indices and sort by source index
        original_indices = []
        for i in range(n_rows):
            matches = [pair for pair in aligned_indices if pair[0] == i]
            if matches:
                original_indices.append((i, matches[0][1]))
        
        return sorted(original_indices)
    
    def _detect_alignment_patterns(self, weak_segments: List[Dict], 
                                  source_segments: List[str], 
                                  translation_segments: List[str]) -> List[Dict]:
        """
        Detect patterns in weak segments that might indicate systematic issues.
        
        Args:
            weak_segments: List of weak segment dictionaries
            source_segments: All source segments
            translation_segments: All translation segments
            
        Returns:
            List of pattern dictionaries with pattern info and occurrences
        """
        if not weak_segments or len(weak_segments) < self.min_pattern_occurrences:
            return []
        
        patterns = []
        
        # Pattern 1: Recurring phrases in source that are weakly translated
        source_phrases = self._extract_phrases(weak_segments, 'source_segment')
        recurring_source_phrases = self._find_recurring_phrases(source_phrases)
        
        for phrase, occurrences in recurring_source_phrases.items():
            if len(occurrences) >= self.min_pattern_occurrences:
                patterns.append({
                    'type': 'recurring_source_phrase',
                    'phrase': phrase,
                    'occurrences': occurrences,
                    'count': len(occurrences),
                    'avg_similarity': sum(occ['similarity'] for occ in occurrences) / len(occurrences)
                })
        
        # Pattern 2: Recurring phrases in translation that are weakly aligned
        translation_phrases = self._extract_phrases(weak_segments, 'translation_segment')
        recurring_translation_phrases = self._find_recurring_phrases(translation_phrases)
        
        for phrase, occurrences in recurring_translation_phrases.items():
            if len(occurrences) >= self.min_pattern_occurrences:
                patterns.append({
                    'type': 'recurring_translation_phrase',
                    'phrase': phrase,
                    'occurrences': occurrences,
                    'count': len(occurrences),
                    'avg_similarity': sum(occ['similarity'] for occ in occurrences) / len(occurrences)
                })
        
        # Pattern 3: Length ratio outliers
        length_ratio_patterns = self._detect_length_ratio_patterns(weak_segments)
        patterns.extend(length_ratio_patterns)
        
        # Pattern 4: Position-based patterns (beginning/middle/end of text)
        position_patterns = self._detect_position_patterns(weak_segments, len(source_segments))
        patterns.extend(position_patterns)
        
        return patterns
    
    def _extract_phrases(self, segments: List[Dict], field: str) -> List[Dict]:
        """
        Extract phrases from segments for pattern analysis.
        
        Args:
            segments: List of segment dictionaries
            field: Field to extract phrases from ('source_segment' or 'translation_segment')
            
        Returns:
            List of phrase dictionaries with the phrase and its context
        """
        phrases = []
        
        for segment in segments:
            text = segment[field]
            
            # Extract individual words (simplified)
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Extract phrases (n-grams)
            for n in range(2, min(5, len(words) + 1)):  # 2-word to 4-word phrases
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    phrases.append({
                        'phrase': phrase,
                        'segment': segment,
                        'similarity': segment['similarity']
                    })
        
        return phrases
    
    def _find_recurring_phrases(self, phrase_dicts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Find phrases that occur multiple times in weak segments.
        
        Args:
            phrase_dicts: List of phrase dictionaries
            
        Returns:
            Dictionary mapping phrases to lists of occurrences
        """
        phrase_occurrences = defaultdict(list)
        
        for phrase_dict in phrase_dicts:
            phrase = phrase_dict['phrase']
            phrase_occurrences[phrase].append(phrase_dict)
        
        # Filter to only recurring phrases
        recurring_phrases = {
            phrase: occurrences 
            for phrase, occurrences in phrase_occurrences.items() 
            if len(occurrences) >= self.min_pattern_occurrences
        }
        
        return recurring_phrases
    
    def _detect_length_ratio_patterns(self, weak_segments: List[Dict]) -> List[Dict]:
        """
        Detect patterns related to segment length ratios.
        
        Args:
            weak_segments: List of weak segment dictionaries
            
        Returns:
            List of length ratio pattern dictionaries
        """
        patterns = []
        
        # Calculate length ratios
        length_ratios = []
        for segment in weak_segments:
            src_len = len(segment['source_segment'])
            tgt_len = len(segment['translation_segment'])
            ratio = tgt_len / max(1, src_len)
            
            length_ratios.append({
                'segment': segment,
                'ratio': ratio,
                'similarity': segment['similarity']
            })
        
        # Check for consistently short translations
        short_translations = [item for item in length_ratios if item['ratio'] < 0.7]
        if len(short_translations) >= self.min_pattern_occurrences:
            patterns.append({
                'type': 'short_translations',
                'occurrences': short_translations,
                'count': len(short_translations),
                'avg_similarity': sum(item['similarity'] for item in short_translations) / len(short_translations)
            })
        
        # Check for consistently long translations
        long_translations = [item for item in length_ratios if item['ratio'] > 1.5]
        if len(long_translations) >= self.min_pattern_occurrences:
            patterns.append({
                'type': 'long_translations',
                'occurrences': long_translations,
                'count': len(long_translations),
                'avg_similarity': sum(item['similarity'] for item in long_translations) / len(long_translations)
            })
        
        return patterns
    
    def _detect_position_patterns(self, weak_segments: List[Dict], total_segments: int) -> List[Dict]:
        """
        Detect patterns related to segment positions in the text.
        
        Args:
            weak_segments: List of weak segment dictionaries
            total_segments: Total number of segments in the source text
            
        Returns:
            List of position-based pattern dictionaries
        """
        patterns = []
        
        if total_segments < 3:
            return patterns
        
        # Define position categories
        beginning_threshold = total_segments // 3
        end_threshold = total_segments - beginning_threshold
        
        # Group segments by position
        beginning_segments = []
        middle_segments = []
        end_segments = []
        
        for segment in weak_segments:
            idx = segment.get('index', segment.get('source_index', 0))
            
            if idx < beginning_threshold:
                beginning_segments.append(segment)
            elif idx >= end_threshold:
                end_segments.append(segment)
            else:
                middle_segments.append(segment)
        
        # Check for patterns in each position
        if len(beginning_segments) >= self.min_pattern_occurrences:
            patterns.append({
                'type': 'beginning_weakness',
                'occurrences': beginning_segments,
                'count': len(beginning_segments),
                'avg_similarity': sum(seg['similarity'] for seg in beginning_segments) / len(beginning_segments)
            })
            
        if len(middle_segments) >= self.min_pattern_occurrences:
            patterns.append({
                'type': 'middle_weakness',
                'occurrences': middle_segments,
                'count': len(middle_segments),
                'avg_similarity': sum(seg['similarity'] for seg in middle_segments) / len(middle_segments)
            })
            
        if len(end_segments) >= self.min_pattern_occurrences:
            patterns.append({
                'type': 'end_weakness',
                'occurrences': end_segments,
                'count': len(end_segments),
                'avg_similarity': sum(seg['similarity'] for seg in end_segments) / len(end_segments)
            })
        
        return patterns

# -----------------------------------------------------------------------------
# Higher-level detector that can pull Groq
# -----------------------------------------------------------------------------

class WeakAlignmentDetector:
    """
    Detects and analyzes weak semantic alignments in translations using both
    embedding-based analysis and Groq evaluation insights.
    """
    
    def __init__(self, embedding_generator=None, groq_evaluator=None, 
                 similarity_threshold=0.75, segment_type='sentence'):
        """
        Initialize the weak alignment detector.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            groq_evaluator: GroqEvaluator instance for enhanced detection
            similarity_threshold: Threshold below which segments are considered weak
            segment_type: Type of segmentation to use
        """
        if embedding_generator is None:
            from embedding_generator import EmbeddingGenerator
            self.embedding_generator = EmbeddingGenerator()
        else:
            self.embedding_generator = embedding_generator
            
        self.groq_evaluator = groq_evaluator
        self.similarity_threshold = similarity_threshold
        
        self.segment_analyzer = SegmentAlignmentAnalyzer(
            embedding_generator=self.embedding_generator,
            similarity_threshold=similarity_threshold,
            segment_type=segment_type
        )
    
    def detect_weak_alignments(self, source_text: str, translation: str, 
                              use_groq: bool = False, detailed: bool = False) -> Dict[str, Any]:
        """
        Detect weak semantic alignments and patterns in the translation.
        
        Args:
            source_text: Source text
            translation: Translated text
            use_groq: Whether to use Groq for enhanced detection
            detailed: Whether to get detailed Groq analysis
            
        Returns:
            Dictionary with weak alignment analysis
        """
        # Analyze segment alignments
        segment_analysis = self.segment_analyzer.analyze_segment_alignment(source_text, translation)
        
        # Get overall quality metrics
        overall_metrics = self._calculate_overall_metrics(segment_analysis)
        
        result = {
            'overall_metrics': overall_metrics,
            'segment_analysis': segment_analysis,
            'weak_alignment_summary': self._generate_weak_alignment_summary(segment_analysis)
        }
        
        # Enhance with Groq analysis if requested
        if use_groq and self.groq_evaluator:
            groq_analysis = self._analyze_with_groq(source_text, translation, 
                                                   segment_analysis['weak_segments'],
                                                   detailed)
            result['groq_analysis'] = groq_analysis
            
            # Update weak segments with Groq insights
            self._enrich_weak_segments_with_groq(segment_analysis['weak_segments'], groq_analysis)
            
            # Generate enhanced summary
            result['enhanced_summary'] = self._generate_enhanced_summary(
                segment_analysis, groq_analysis
            )
        
        return result
    
    def _calculate_overall_metrics(self, segment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall metrics from segment analysis.
        
        Args:
            segment_analysis: Segment alignment analysis dictionary
            
        Returns:
            Dictionary of overall metrics
        """
        similarity_stats = segment_analysis['similarity_stats']
        weak_segments = segment_analysis['weak_segments']
        
        # Calculate percentage of weak segments
        total_segments = len(segment_analysis['all_segment_similarities'])
        weak_segment_percentage = len(weak_segments) / total_segments if total_segments > 0 else 0
        
        # Calculate severity of weakness
        if weak_segments:
            avg_weak_similarity = sum(seg['similarity'] for seg in weak_segments) / len(weak_segments)
            severity = 1.0 - avg_weak_similarity  # Higher value means more severe
        else:
            avg_weak_similarity = None
            severity = 0
        
        # Calculate weak segment patterns severity
        patterns = segment_analysis['recurring_patterns']
        pattern_severity = 0
        if patterns:
            # More recurring patterns and more occurrences of each pattern increase severity
            pattern_count_factor = min(1.0, len(patterns) / 5)  # Cap at 5 patterns
            occurrence_factor = min(1.0, sum(p['count'] for p in patterns) / (2 * total_segments))
            pattern_severity = (pattern_count_factor + occurrence_factor) / 2
        
        return {
            'avg_similarity': similarity_stats['average'],
            'weak_segment_count': len(weak_segments),
            'weak_segment_percentage': weak_segment_percentage,
            'avg_weak_similarity': avg_weak_similarity,
            'severity': severity,
            'recurring_pattern_count': len(patterns),
            'pattern_severity': pattern_severity,
            'overall_alignment_score': 1.0 - (severity * 0.7 + pattern_severity * 0.3),
            'similarity_trend': similarity_stats['trend']
        }
    
    def _generate_weak_alignment_summary(self, segment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a human-readable summary of weak alignment findings.
        
        Args:
            segment_analysis: Segment alignment analysis dictionary
            
        Returns:
            Dictionary with summary information
        """
        metrics = self._calculate_overall_metrics(segment_analysis)
        weak_segments = segment_analysis['weak_segments']
        patterns = segment_analysis['recurring_patterns']
        
        # Generate summary text
        summary_parts = []
        
        if not weak_segments:
            main_finding = "No significant weak alignments detected."
        else:
            percentage = metrics['weak_segment_percentage'] * 100
            main_finding = f"Found {len(weak_segments)} weak segments ({percentage:.1f}% of total)."
            
            if patterns:
                pattern_summary = f"Detected {len(patterns)} recurring patterns of weak alignment."
                if any(p['type'] == 'beginning_weakness' for p in patterns):
                    pattern_summary += " Issues concentrated at the beginning of the text."
                elif any(p['type'] == 'end_weakness' for p in patterns):
                    pattern_summary += " Issues concentrated at the end of the text."
                elif any(p['type'] == 'middle_weakness' for p in patterns):
                    pattern_summary += " Issues concentrated in the middle of the text."
                    
                summary_parts.append(pattern_summary)
                
                # Add details on specific pattern types
                phrase_patterns = [p for p in patterns if p['type'] in 
                                 ['recurring_source_phrase', 'recurring_translation_phrase']]
                if phrase_patterns:
                    top_phrases = sorted(phrase_patterns, key=lambda p: p['count'], reverse=True)[:3]
                    phrase_summary = "Recurring problematic phrases include: " + ", ".join(
                        f'"{p["phrase"]}" ({p["count"]} occurrences)' for p in top_phrases
                    )
                    summary_parts.append(phrase_summary)
                
                length_patterns = [p for p in patterns if p['type'] in 
                                 ['short_translations', 'long_translations']]
                if length_patterns:
                    length_issues = []
                    for p in length_patterns:
                        if p['type'] == 'short_translations':
                            length_issues.append(f"translations too short ({p['count']} segments)")
                        else:
                            length_issues.append(f"translations too long ({p['count']} segments)")
                    
                    length_summary = "Length issues detected: " + ", ".join(length_issues)
                    summary_parts.append(length_summary)
        
        # Generate recommendations
        recommendations = []
        
        if metrics['weak_segment_percentage'] > 0.3:
            recommendations.append("Consider a full retranslation as alignment issues are widespread.")
        elif metrics['weak_segment_percentage'] > 0.1:
            recommendations.append("Review and correct the identified weak segments.")
            
        if patterns and any(p['type'] in ['recurring_source_phrase', 'recurring_translation_phrase'] for p in patterns):
            recommendations.append("Create a glossary for consistently problematic terms/phrases.")
            
        if metrics.get('similarity_trend') == 'degrading':
            recommendations.append("Quality degrades throughout the text. Check if different translators worked on different sections.")
        
        return {
            'main_finding': main_finding,
            'details': summary_parts,
            'recommendations': recommendations,
            'alignment_score': metrics['overall_alignment_score'],
            'severity_level': self._get_severity_level(metrics['overall_alignment_score'])
        }
    
    def _get_severity_level(self, score: float) -> str:
        """
        Convert alignment score to a severity level label.
        
        Args:
            score: Alignment score (0-1)
            
        Returns:
            Severity level as string
        """
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "acceptable"
        elif score >= 0.5:
            return "problematic"
        else:
            return "critical"
    
    def _analyze_with_groq(self, source_text: str, translation: str, 
                          weak_segments: List[Dict], detailed: bool) -> Dict[str, Any]:
        """
        Enhance alignment analysis with Groq insights.
        
        Args:
            source_text: Source text
            translation: Translated text
            weak_segments: List of weak segment dictionaries
            detailed: Whether to get detailed Groq analysis
            
        Returns:
            Dictionary with Groq analysis results
        """
        if not self.groq_evaluator:
            return {'error': 'No Groq evaluator available'}
        
        # Analyze entire text
        overall_evaluation = self.groq_evaluator.evaluate_translation(
            source_text=source_text,
            translation=translation,
            detailed=detailed
        )
        
        # Analyze weak segments individually
        segment_evaluations = []
        
        # Limit to at most 5 segments to avoid excessive API usage
        segments_to_analyze = weak_segments[:min(5, len(weak_segments))]
        
        for segment in segments_to_analyze:
            src_segment = segment['source_segment']
            tgt_segment = segment['translation_segment']
            
            # Use simple evaluation for efficiency
            evaluation = self.groq_evaluator.evaluate_translation(
                source_text=src_segment,
                translation=tgt_segment,
                detailed=False  # Simple evaluation for segments
            )
            
            segment_evaluations.append({
                'segment': segment,
                'evaluation': evaluation
            })
        
        # If detailed is requested, also do error analysis
        error_analysis = None
        if detailed:
            error_analysis = self.groq_evaluator.analyze_translation_errors(
                source_text=source_text,
                translation=translation
            )
        
        return {
            'overall_evaluation': overall_evaluation,
            'segment_evaluations': segment_evaluations,
            'error_analysis': error_analysis
        }
    
    def _enrich_weak_segments_with_groq(self, weak_segments: List[Dict], 
                                        groq_analysis: Dict[str, Any]) -> None:
        """
        Enrich weak segment data with Groq evaluation insights.
        
        Args:
            weak_segments: List of weak segment dictionaries to enrich
            groq_analysis: Groq analysis results
        """
        if 'error' in groq_analysis or not groq_analysis.get('segment_evaluations'):
            return
        
        # Create mapping from source/target segment text to evaluation
        evaluation_map = {}
        for item in groq_analysis['segment_evaluations']:
            segment = item['segment']
            evaluation = item['evaluation']
            
            key = (segment['source_segment'], segment['translation_segment'])
            evaluation_map[key] = evaluation
        
        # Enrich weak segments with Groq insights
        for segment in weak_segments:
            key = (segment['source_segment'], segment['translation_segment'])
            if key in evaluation_map:
                segment['groq_evaluation'] = evaluation_map[key]
                
                # Add groq score (normalized to 0-1)
                if 'overall_score' in evaluation_map[key]:
                    segment['groq_score'] = evaluation_map[key]['overall_score'] / 10.0
    
    def _generate_enhanced_summary(self, segment_analysis: Dict[str, Any], 
                                  groq_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an enhanced summary combining embedding and Groq insights.
        
        Args:
            segment_analysis: Segment alignment analysis
            groq_analysis: Groq analysis results
            
        Returns:
            Dictionary with enhanced summary
        """
        if 'error' in groq_analysis:
            return {'error': groq_analysis['error']}
        
        # Get base summary
        base_summary = self._generate_weak_alignment_summary(segment_analysis)
        
        # Get Groq overall evaluation
        overall_eval = groq_analysis.get('overall_evaluation', {})
        
        # Calculate combined score
        embedding_score = base_summary['alignment_score']
        groq_score = overall_eval.get('overall_score', 0) / 10.0 if 'overall_score' in overall_eval else 0
        
        combined_score = embedding_score * 0.5 + groq_score * 0.5
        
        # Get error analysis insights
        error_insights = []
        error_analysis = groq_analysis.get('error_analysis', {})
        
        if error_analysis and 'error_summary' in error_analysis:
            summary = error_analysis['error_summary']
            
            if summary.get('meaning_errors', 0) > 0:
                error_insights.append(
                    f"Meaning errors: {summary['meaning_errors']} instances where the translation changes the original meaning."
                )
                
            if summary.get('terminology_errors', 0) > 0:
                error_insights.append(
                    f"Terminology errors: {summary['terminology_errors']} instances of incorrect term usage."
                )
                
            if summary.get('style_errors', 0) > 0:
                error_insights.append(
                    f"Style errors: {summary['style_errors']} instances of inappropriate style or tone."
                )
        
        # Generate combined recommendations
        recommendations = base_summary['recommendations'].copy()
        
        # Add Groq-specific recommendations
        if groq_score < 0.7 and 'groq_evaluation' in overall_eval:
            if 'accuracy_comments' in overall_eval:
                if overall_eval.get('accuracy', 0) < 7:
                    recommendations.append(
                        f"Focus on improving accuracy: {overall_eval['accuracy_comments']}"
                    )
                    
            if 'fluency_comments' in overall_eval:
                if overall_eval.get('fluency', 0) < 7:
                    recommendations.append(
                        f"Improve fluency: {overall_eval['fluency_comments']}"
                    )
                    
            if 'terminology_comments' in overall_eval:
                if overall_eval.get('terminology', 0) < 7:
                    recommendations.append(
                        f"Address terminology issues: {overall_eval['terminology_comments']}"
                    )
        
        # Generate enhanced findings
        enhanced_finding = base_summary['main_finding']
        
        if 'summary' in overall_eval:
            enhanced_finding += f" Groq evaluation: {overall_eval['summary']}"
        
        return {
            'main_finding': enhanced_finding,
            'embedding_details': base_summary['details'],
            'groq_insights': error_insights,
            'recommendations': recommendations,
            'combined_score': combined_score,
            'embedding_score': embedding_score,
            'groq_score': groq_score,
            'severity_level': self._get_severity_level(combined_score)
        }


def generate_alignment_report(analysis_results, output_format='text'):
    """
    Generate a human-readable report of translation alignment analysis.
    
    Args:
        analysis_results: Results from analyze_translation_with_alignment
        output_format: Format for the report ('text' or 'html')
        
    Returns:
        Formatted report
    """
    if "alignment_analysis" not in analysis_results:
        return "No alignment analysis available."
    
    alignment = analysis_results["alignment_analysis"]
    
    if output_format == 'html':
        # Generate HTML report with visualizations
        # This would include colored segments, charts, etc.
        # For brevity, we'll skip the implementation details
        return "<html>HTML report would go here</html>"

    # Generate text report
    report = []

    # Add header
    report.append("TRANSLATION ALIGNMENT ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")

    # Add overall metrics
    report.append("OVERALL QUALITY METRICS")
    report.append("-" * 30)
    report.append(f"Composite Score: {analysis_results['composite_score']:.2f}")
    report.append(f"Embedding Similarity: {analysis_results['embedding_similarity']:.2f}")

    if "groq_score" in analysis_results:
        report.append(f"Groq Score: {analysis_results['groq_score'] / 10:.2f}")

    report.append(f"Alignment Score: {analysis_results.get('alignment_score', 'N/A')}")
    report.append("")

    # Add alignment summary
    if "enhanced_summary" in alignment:
        summary = alignment["enhanced_summary"]
        report.append("ENHANCED ALIGNMENT SUMMARY")
        report.append("-" * 30)
        report.append(f"Main Finding: {summary['main_finding']}")
        report.append("")
        
        if summary['embedding_details']:
            report.append("Details:")
            for detail in summary['embedding_details']:
                report.append(f"- {detail}")
            report.append("")
        
        if summary['groq_insights']:
            report.append("Groq Insights:")
            for insight in summary['groq_insights']:
                report.append(f"- {insight}")
            report.append("")
        
        report.append(f"Severity: {summary['severity_level'].upper()}")
        report.append(f"Combined Score: {summary['combined_score']:.2f}")
        report.append("")
    else:
        summary = alignment["weak_alignment_summary"]
        report.append("ALIGNMENT SUMMARY")
        report.append("-" * 30)
        report.append(f"Main Finding: {summary['main_finding']}")
        report.append("")
        
        if summary['details']:
            report.append("Details:")
            for detail in summary['details']:
                report.append(f"- {detail}")
            report.append("")
        
        report.append(f"Severity: {summary['severity_level'].upper()}")
        report.append(f"Alignment Score: {summary['alignment_score']:.2f}")
        report.append("")

    # Add weak segments
    weak_segments = alignment["segment_analysis"]["weak_segments"]
    if weak_segments:
        report.append("WEAK SEGMENTS")
        report.append("-" * 30)
        
        # Show top 5 weakest segments
        sorted_segments = sorted(weak_segments, key=lambda x: x["similarity"])
        for i, segment in enumerate(sorted_segments[:5], 1):
            report.append(f"Segment {i} (Similarity: {segment['similarity']:.2f}):")
            report.append(f"  Source: {segment['source_segment']}")
            report.append(f"  Translation: {segment['translation_segment']}")
            if "groq_evaluation" in segment:
                groq_eval = segment["groq_evaluation"]
                report.append(f"  Groq Score: {groq_eval.get('overall_score', 0) / 10:.2f}")
                report.append(f"  Groq Comment: {groq_eval.get('summary', 'N/A')}")
            report.append("")

    # Add recurring patterns
    patterns = alignment["segment_analysis"]["recurring_patterns"]
    if patterns:
        report.append("RECURRING PATTERNS")
        report.append("-" * 30)
        
        for i, pattern in enumerate(patterns, 1):
            pattern_type = pattern["type"].replace("_", " ").title()
            report.append(f"Pattern {i}: {pattern_type}")
            
            if "phrase" in pattern:
                report.append(f"  Phrase: \"{pattern['phrase']}\"")
                
            report.append(f"  Occurrences: {pattern['count']}")
            report.append(f"  Average Similarity: {pattern['avg_similarity']:.2f}")
            report.append("")

    # Add recommendations
    if "enhanced_summary" in alignment:
        recommendations = alignment["enhanced_summary"]["recommendations"]
    else:
        recommendations = alignment["weak_alignment_summary"]["recommendations"]
        
    if recommendations:
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        for i, recommendation in enumerate(recommendations, 1):
            report.append(f"{i}. {recommendation}")
        report.append("")

    return "\n".join(report)

# =====================================================================
# Cosine similarity helper (made available at module level for patching
# =====================================================================

def cosine_similarity(v1, v2):
    """Return cosine similarity between two vectors or mock embeddings.

    Falls back gracefully to 0.0 if vectors are empty or non-numeric.  The
    implementation is intentionally minimal so that tests can monkey-patch
    this function for custom scenarios.
    """
    try:
        import numpy as _np
        if v1 is None or v2 is None:
            return 0.0
        # Handle MagicMock objects used in tests – let patched version decide
        if not isinstance(v1, _np.ndarray) or not isinstance(v2, _np.ndarray):
            # Simple heuristic: identical object means perfect match
            return 1.0 if v1 is v2 else 0.0
        denom = _np.linalg.norm(v1) * _np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(_np.dot(v1, v2) / denom)
    except Exception:  # pragma: no cover – any unexpected error yields 0
        return 0.0

# Expose in __all__ for linter friendliness
__all__ = globals().get("__all__", []) + ["SlidingWindowAligner", "cosine_similarity"]

# =====================================================================
# Sliding-window alignment utility
# =====================================================================

class SlidingWindowAligner:
    """Align segments between source and translation using a sliding window.

    Designed for situations where sentence counts do not match due to
    insertions deletions or merging of sentences.  It provides a quick
    heuristic rather than an optimal alignment guaranteed by dynamic
    programming – sufficient for quality diagnostics and unit tests.
    """

    def __init__(self, *, embedding_generator=None, window_size: int = 2, similarity_threshold: float = 0.7):
        if embedding_generator is None:
            from embedding_generator import EmbeddingGenerator  # local import to avoid heavy deps at cold start
            embedding_generator = EmbeddingGenerator()
        self.embedding_generator = embedding_generator
        self.window_size = max(1, int(window_size))
        self.similarity_threshold = similarity_threshold

    # -----------------------------------------------------------------
    # Public API used by tests
    # -----------------------------------------------------------------
    def calculate_alignment_score(self, source_text: str, translation: str):
        """Return alignment diagnostics for *source_text* vs *translation*."""
        raw = self._calculate_raw_alignment(source_text, translation)
        quality = self._get_alignment_quality(raw["raw_score"])
        result = {
            "alignment_score": raw["raw_score"],
            "alignment_quality": quality,
            "segment_alignments": raw["segment_alignments"],
            "missing_segments": raw["missing_segments"],
            "extra_segments": raw["extra_segments"],
            "unaligned_segments": raw["unaligned_segments"],
        }
        return result

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _sliding_window_pass(self, src_segments, tgt_segments, embeddings_src, embeddings_tgt, matched_src, matched_tgt, alignments):
        """Second pass that groups consecutive unmatched segments."""
        un_src_indices = [i for i in range(len(src_segments)) if i not in matched_src]
        un_tgt_indices = [j for j in range(len(tgt_segments)) if j not in matched_tgt]

        for src_start in un_src_indices:
            if src_start in matched_src:
                continue
            # Build source window text
            src_window = []
            for k in range(self.window_size):
                idx = src_start + k
                if idx < len(src_segments) and idx not in matched_src:
                    src_window.append(src_segments[idx])
                else:
                    break
            if not src_window:
                continue
            src_text_window = " ".join(src_window)
            src_emb_window = self.embedding_generator.generate_embedding(src_text_window)

            best_j = None
            best_sim = -1.0
            best_tgt_indices = []
            for tgt_start in un_tgt_indices:
                for win_len in range(1, self.window_size + 1):
                    indices = list(range(tgt_start, min(tgt_start + win_len, len(tgt_segments))))
                    if any(j in matched_tgt for j in indices):
                        continue
                    tgt_text_window = " ".join(tgt_segments[j] for j in indices)
                    tgt_emb_window = self.embedding_generator.generate_embedding(tgt_text_window)
                    sim = cosine_similarity(src_emb_window, tgt_emb_window)
                    if sim > best_sim:
                        best_sim = sim
                        best_j = tgt_start
                        best_tgt_indices = indices
            if best_j is not None and best_sim >= self.similarity_threshold:
                alignments.append({
                    "source": src_text_window,
                    "target": " ".join(tgt_segments[j] for j in best_tgt_indices),
                    "source_indices": list(range(src_start, src_start + len(src_window))),
                    "target_indices": best_tgt_indices,
                    "similarity": best_sim,
                    "is_window": True,
                })
                matched_src.update(range(src_start, src_start + len(src_window)))
                matched_tgt.update(best_tgt_indices)

    def _finalise(self, alignments, total_src, total_tgt):
        # Compute score
        src_covered: set[int] = set()
        tgt_covered: set[int] = set()

        if not alignments:
            raw_score = 0.0
            coverage = 0.0
        else:
            avg_sim = sum(a["similarity"] for a in alignments) / len(alignments)
            for al in alignments:
                if "source_indices" in al:
                    src_covered.update(al["source_indices"])
                elif "source_index" in al:
                    src_covered.add(al["source_index"])

                if "target_indices" in al:
                    tgt_covered.update(al["target_indices"])
                elif "target_index" in al:
                    tgt_covered.add(al["target_index"])

            coverage = ((len(src_covered) / total_src if total_src else 0) + (len(tgt_covered) / total_tgt if total_tgt else 0)) / 2
            raw_score = avg_sim * (0.25 + 0.75 * coverage)

            # Penalise for missing or extra segments explicitly
            penalty_factor = 0.08 * ((total_src - len(src_covered)) + (total_tgt - len(tgt_covered))) / max(total_src, total_tgt)
            raw_score -= penalty_factor

        raw_score = max(0.0, min(1.0, raw_score + 0.05))
        return {
            "alignment_score": raw_score,
            "alignment_quality": self._get_alignment_quality(raw_score),
            "segment_alignments": alignments,
            "missing_segments": total_src - len(src_covered),
            "extra_segments": total_tgt - len(tgt_covered),
            "unaligned_segments": (total_src - len(src_covered)) + (total_tgt - len(tgt_covered)),
        }

    def _get_alignment_quality(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        if score >= 0.8:
            return "good"
        if score >= 0.7:
            return "acceptable"
        if score >= 0.6:
            return "fair"
        if score >= 0.5:
            return "poor"
        return "critical"

    # Internal raw alignment logic separated so it can be reused
    def _raw_alignment_internal(self, source_text: str, translation_text: str):
        src_segments = _sentence_split(source_text)
        tgt_segments = _sentence_split(translation_text)

        # Edge-case quick return
        if not src_segments or not tgt_segments:
            return {
                "raw_score": 0.0,
                "segment_alignments": [],
                "missing_segments": len(src_segments),
                "extra_segments": len(tgt_segments),
                "unaligned_segments": len(src_segments) + len(tgt_segments),
            }

        embeddings_src = [self.embedding_generator.generate_embedding(s) for s in src_segments]
        embeddings_tgt = [self.embedding_generator.generate_embedding(t) for t in tgt_segments]

        matched_src, matched_tgt, alignments = set(), set(), []

        for i, emb_s in enumerate(embeddings_src):
            best_j, best_sim = None, -1.0
            for j, emb_t in enumerate(embeddings_tgt):
                if j in matched_tgt:
                    continue
                sim = cosine_similarity(emb_s, emb_t)
                if sim > best_sim:
                    best_sim, best_j = sim, j
            if best_j is not None and best_sim >= self.similarity_threshold:
                alignments.append({
                    "source": src_segments[i],
                    "target": tgt_segments[best_j],
                    "source_index": i,
                    "target_index": best_j,
                    "similarity": min(1.0, best_sim),
                })
                matched_src.add(i)
                matched_tgt.add(best_j)

        # Sliding window for leftovers
        if len(matched_src) < len(src_segments):
            self._sliding_window_pass(src_segments, tgt_segments, embeddings_src, embeddings_tgt, matched_src, matched_tgt, alignments)

        final = self._finalise(alignments, len(src_segments), len(tgt_segments))
        return {
            "raw_score": final["alignment_score"],
            "segment_alignments": alignments,
            "missing_segments": final["missing_segments"],
            "extra_segments": final["extra_segments"],
            "unaligned_segments": final["unaligned_segments"],
        }

    # Public method that unit tests patch
    def _calculate_raw_alignment(self, source_text: str, translation_text: str):
        return self._raw_alignment_internal(source_text, translation_text) 