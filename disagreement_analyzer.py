"""
Analyzer for cases where embedding similarity scores disagree with AI quality ratings.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re

# Configure logging using unified configuration
from logger_config import get_logger
logger = get_logger("DisagreementAnalyzer", "disagreements")

class DisagreementAnalyzer:
    """
    Analyzes cases where metrics (especially embedding similarity and AI quality ratings) 
    disagree on translation quality.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the disagreement analyzer.
        
        Args:
            threshold: Threshold for considering metrics in disagreement
                      (normalized score difference above this value is a disagreement)
        """
        self.threshold = threshold
        self.data = None
        self.disagreements = None
        self.disagreement_categories = {}
        self.disagreement_scores = {}
        self.language_patterns = {}
        self.text_features = {}
        self.disagreement_metrics = {}
        
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load data for analysis.
        
        Args:
            data: DataFrame with translation quality metrics
        """
        self.data = data.copy()
        logger.info(f"Loaded data with {len(self.data)} rows")
        
        # Check for required columns
        required_columns = ["embedding_similarity", "groq_quality_score"]
        for col in required_columns:
            if col not in self.data.columns:
                logger.warning(f"Required column '{col}' not found in data")
        
    def identify_disagreements(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Identify cases where metrics disagree on quality.
        
        Args:
            metrics: Optional list of metrics to consider (defaults to embedding_similarity and groq_quality_score)
            
        Returns:
            DataFrame with disagreement cases
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return pd.DataFrame()
            
        # Set default metrics if none provided
        if metrics is None:
            metrics = ["embedding_similarity", "groq_quality_score"]
            
        # Check if metrics exist
        for metric in metrics:
            if metric not in self.data.columns:
                logger.error(f"Metric '{metric}' not found in data")
                return pd.DataFrame()
                
        # Create a copy of the data for processing
        data = self.data.copy()
        
        # Normalize metrics to 0-1 scale
        normalized_data = data.copy()
        for metric in metrics:
            if data[metric].nunique() > 1:  # Only normalize if we have variation
                min_val = data[metric].min()
                max_val = data[metric].max()
                if max_val > min_val:
                    normalized_data[metric] = (data[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_data[metric] = 0.5  # Default if all values are the same
        
        # Calculate pairwise differences between normalized metrics
        disagreements = []
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                diff_col = f"{metric1}_vs_{metric2}_diff"
                normalized_data[diff_col] = abs(normalized_data[metric1] - normalized_data[metric2])
                
                # Mark as disagreement if difference exceeds threshold
                is_disagreement = normalized_data[diff_col] > self.threshold
                
                # Find disagreement cases
                disagreement_cases = normalized_data[is_disagreement].copy()
                if not disagreement_cases.empty:
                    # Add original (non-normalized) metric values
                    for metric in metrics:
                        disagreement_cases[f"original_{metric}"] = data.loc[disagreement_cases.index, metric]
                        
                    # Add comparison details
                    disagreement_cases["metric1"] = metric1
                    disagreement_cases["metric2"] = metric2
                    disagreement_cases["disagreement_type"] = np.where(
                        data.loc[disagreement_cases.index, metric1] > data.loc[disagreement_cases.index, metric2],
                        f"{metric1}_higher",
                        f"{metric2}_higher"
                    )
                    
                    disagreements.append(disagreement_cases)
        
        # Combine all disagreements
        if disagreements:
            disagreement_df = pd.concat(disagreements, ignore_index=True)
            
            # Add severity score
            max_diff = disagreement_df[[f"{metric1}_vs_{metric2}_diff" for i, metric1 in enumerate(metrics) 
                                      for metric2 in metrics[i+1:]]].max(axis=1)
            disagreement_df["severity"] = max_diff
            
            # Sort by severity
            disagreement_df = disagreement_df.sort_values("severity", ascending=False)
            
            self.disagreements = disagreement_df
            
            logger.info(f"Identified {len(disagreement_df)} disagreement cases")
            
            # Log the most severe disagreements
            top_n = 5
            if len(disagreement_df) > 0:
                top_disagreements = disagreement_df.head(top_n)
                logger.info(f"Top {min(top_n, len(top_disagreements))} disagreements by severity:")
                for i, (idx, row) in enumerate(top_disagreements.iterrows(), 1):
                    logger.info(f"  {i}. {row['metric1']}={row[f'original_{row['metric1']}']:.2f} vs "
                                f"{row['metric2']}={row[f'original_{row['metric2']}']:.2f} "
                                f"(severity: {row['severity']:.2f})")
            
            return disagreement_df
        else:
            logger.info("No disagreements found above the threshold")
            self.disagreements = pd.DataFrame()
            return pd.DataFrame()
    
    def classify_disagreements(self) -> Dict[str, pd.DataFrame]:
        """
        Classify disagreements into different types.
        
        Returns:
            Dictionary mapping category names to DataFrame of cases
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
        
        # Classify by disagreement type
        type_groups = {}
        for disagreement_type, group in self.disagreements.groupby("disagreement_type"):
            type_groups[disagreement_type] = group
            
        # Classify by severity
        if "severity" in self.disagreements.columns:
            # Define severity categories
            low_threshold = 0.3
            medium_threshold = 0.5
            high_threshold = 0.7
            
            severity_groups = {
                "low_severity": self.disagreements[self.disagreements["severity"] < low_threshold],
                "medium_severity": self.disagreements[(self.disagreements["severity"] >= low_threshold) & 
                                                    (self.disagreements["severity"] < medium_threshold)],
                "high_severity": self.disagreements[(self.disagreements["severity"] >= medium_threshold) & 
                                                   (self.disagreements["severity"] < high_threshold)],
                "extreme_severity": self.disagreements[self.disagreements["severity"] >= high_threshold]
            }
            
            # Remove empty groups
            severity_groups = {k: v for k, v in severity_groups.items() if not v.empty}
            
            # Add to type groups
            type_groups.update(severity_groups)
        
        # Classify by language pair if available
        if all(col in self.disagreements.columns for col in ["source_language", "target_language"]):
            self.disagreements["language_pair"] = self.disagreements["source_language"] + "-" + self.disagreements["target_language"]
            
            # Get top language pairs by disagreement count
            top_pairs = self.disagreements["language_pair"].value_counts().head(5).index
            
            for lang_pair in top_pairs:
                group_key = f"language_{lang_pair}"
                type_groups[group_key] = self.disagreements[self.disagreements["language_pair"] == lang_pair]
        
        # Store the categorized disagreements
        self.disagreement_categories = type_groups
        
        # Log category counts
        logger.info(f"Classified disagreements into {len(type_groups)} categories:")
        for category, group in type_groups.items():
            logger.info(f"  {category}: {len(group)} cases")
            
        return type_groups 
    
    def score_disagreements(self) -> pd.DataFrame:
        """
        Score the severity of disagreements.
        
        Returns:
            DataFrame with scored disagreements
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return pd.DataFrame()
            
        # Get copy of disagreements
        scored_disagreements = self.disagreements.copy()
        
        # Add more detailed scoring
        # 1. Basic severity score - already calculated in identify_disagreements()
        
        # 2. Weighted severity - give more weight to larger absolute values
        if "embedding_similarity" in self.data.columns and "groq_quality_score" in self.data.columns:
            # Normalize groq score to 0-1 scale for comparison
            max_groq = self.data["groq_quality_score"].max()
            min_groq = self.data["groq_quality_score"].min()
            
            def calculate_weighted_severity(row):
                # Normalize groq score to 0-1
                if max_groq > min_groq:
                    norm_groq = (row["original_groq_quality_score"] - min_groq) / (max_groq - min_groq)
                else:
                    norm_groq = 0.5
                    
                # Calculate absolute values (higher values matter more)
                abs_embedding = row["original_embedding_similarity"]
                abs_groq = norm_groq
                
                # Calculate absolute difference
                abs_diff = abs(abs_embedding - abs_groq)
                
                # Higher values with big differences should be weighted more
                avg_value = (abs_embedding + abs_groq) / 2
                weighted_severity = abs_diff * (0.5 + avg_value)
                
                return weighted_severity
                
            # Apply weighted severity calculation where we have both metrics
            mask = (
                scored_disagreements["metric1"].isin(["embedding_similarity", "groq_quality_score"]) &
                scored_disagreements["metric2"].isin(["embedding_similarity", "groq_quality_score"])
            )
            
            scored_disagreements.loc[mask, "weighted_severity"] = scored_disagreements[mask].apply(
                calculate_weighted_severity, axis=1
            )
            
            # Fill missing weighted severity with regular severity
            scored_disagreements["weighted_severity"] = scored_disagreements["weighted_severity"].fillna(
                scored_disagreements["severity"]
            )
        
        # 3. Add confidence score based on the magnitude of the metrics
        # Higher absolute values have higher confidence
        scored_disagreements["confidence"] = scored_disagreements.apply(
            lambda row: (row[f"original_{row['metric1']}"] + row[f"original_{row['metric2']}"]) / 2,
            axis=1
        )
        
        # 4. Calculate overall priority score
        scored_disagreements["priority_score"] = (
            scored_disagreements["severity"] * 0.6 + 
            scored_disagreements.get("weighted_severity", scored_disagreements["severity"]) * 0.3 + 
            scored_disagreements["confidence"] * 0.1
        )
        
        # Sort by priority score
        scored_disagreements = scored_disagreements.sort_values("priority_score", ascending=False)
        
        # Save to instance
        self.disagreement_scores = scored_disagreements
        
        logger.info("Scored disagreements by severity and priority")
        
        return scored_disagreements
    
    def analyze_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze language patterns in disagreements.
        
        Returns:
            Dictionary with language pattern analysis
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
            
        # Check if language columns exist
        if not all(col in self.disagreements.columns for col in ["source_language", "target_language"]):
            logger.warning("Language columns not found in disagreement data")
            return {}
            
        # Create language pair column if not exists
        if "language_pair" not in self.disagreements.columns:
            self.disagreements["language_pair"] = self.disagreements["source_language"] + "-" + self.disagreements["target_language"]
            
        # Get language pair counts and proportions
        total_count = len(self.disagreements)
        lang_counts = self.disagreements["language_pair"].value_counts()
        lang_props = lang_counts / total_count
        
        # Calculate data stats for comparison
        if self.data is not None:
            self.data["language_pair"] = self.data["source_language"] + "-" + self.data["target_language"]
            data_lang_counts = self.data["language_pair"].value_counts()
            data_lang_props = data_lang_counts / len(self.data)
            
            # Calculate disagreement rate per language pair
            disagreement_rates = {}
            for lang_pair in lang_counts.index:
                if lang_pair in data_lang_counts:
                    rate = lang_counts[lang_pair] / data_lang_counts[lang_pair]
                    disagreement_rates[lang_pair] = rate
            
            # Sort by disagreement rate (high to low)
            disagreement_rates = dict(sorted(
                disagreement_rates.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        else:
            data_lang_props = pd.Series()
            disagreement_rates = {}
        
        # Get metrics by language pair
        language_metrics = {}
        for lang_pair, group in self.disagreements.groupby("language_pair"):
            
            # Calculate the common metric pairs and their average difference
            metric_diffs = {}
            for metric1, metric2 in set(zip(group["metric1"], group["metric2"])):
                # Filter for this metric pair
                pair_rows = group[(group["metric1"] == metric1) & (group["metric2"] == metric2)]
                
                if not pair_rows.empty:
                    # Calculate average difference and severity
                    diff_col = f"{metric1}_vs_{metric2}_diff"
                    avg_diff = pair_rows[diff_col].mean() if diff_col in pair_rows.columns else None
                    
                    # Check which direction the disagreement tends to go
                    direction = (
                        pair_rows["disagreement_type"].value_counts().idxmax() 
                        if not pair_rows["disagreement_type"].empty else "unknown"
                    )
                    
                    metric_diffs[f"{metric1}_vs_{metric2}"] = {
                        "avg_difference": avg_diff,
                        "count": len(pair_rows),
                        "common_direction": direction,
                        "avg_severity": pair_rows["severity"].mean()
                    }
            
            # Store language pair metrics
            language_metrics[lang_pair] = {
                "count": int(len(group)),
                "percentage": float(len(group) / total_count * 100),
                "metrics": metric_diffs,
                "avg_severity": float(group["severity"].mean()),
                "disagreement_rate": float(disagreement_rates.get(lang_pair, 0))
            }
        
        # Analyze patterns across language pairs
        lang_pattern_stats = {
            "total_languages": len(language_metrics),
            "highest_disagreement_rate": {
                "language_pair": max(disagreement_rates.items(), key=lambda x: x[1])[0] if disagreement_rates else None,
                "rate": max(disagreement_rates.values()) if disagreement_rates else 0
            },
            "lowest_disagreement_rate": {
                "language_pair": min(disagreement_rates.items(), key=lambda x: x[1])[0] if disagreement_rates else None,
                "rate": min(disagreement_rates.values()) if disagreement_rates else 0
            },
            "high_disagreement_languages": [
                {"language_pair": lang, "rate": rate}
                for lang, rate in list(disagreement_rates.items())[:5]
            ] if disagreement_rates else []
        }
        
        # Store the language pattern analysis
        self.language_patterns = {
            "language_metrics": language_metrics,
            "summary": lang_pattern_stats
        }
        
        logger.info(f"Analyzed language patterns across {len(language_metrics)} language pairs")
        if disagreement_rates:
            logger.info(f"Highest disagreement rate: {lang_pattern_stats['highest_disagreement_rate']['language_pair']} "
                      f"({lang_pattern_stats['highest_disagreement_rate']['rate']:.2%})")
        
        return self.language_patterns
    
    def analyze_text_features(self, max_samples: int = 500) -> Dict[str, Any]:
        """
        Analyze text features in disagreements.
        
        Args:
            max_samples: Maximum number of samples to analyze (for performance)
            
        Returns:
            Dictionary with text feature analysis
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
            
        # Check if text columns exist
        text_cols = ["source_text", "translated_text"]
        if not any(col in self.disagreements.columns for col in text_cols):
            logger.warning("No text columns found in disagreement data")
            return {}
            
        # Limit sample size for performance
        sample_data = self.disagreements
        if len(sample_data) > max_samples:
            sample_data = sample_data.sample(max_samples, random_state=42)
            
        # Initialize text analytics results
        text_features = {
            "length_analysis": {},
            "complexity_analysis": {},
            "term_frequency": {},
            "pattern_analysis": {}
        }
        
        # 1. Text Length Analysis
        for col in text_cols:
            if col in sample_data.columns:
                sample_data[f"{col}_length"] = sample_data[col].astype(str).apply(len)
                sample_data[f"{col}_word_count"] = sample_data[col].astype(str).apply(lambda x: len(x.split()))
                
                # Calculate statistics
                text_features["length_analysis"][col] = {
                    "avg_length": float(sample_data[f"{col}_length"].mean()),
                    "std_length": float(sample_data[f"{col}_length"].std()),
                    "avg_word_count": float(sample_data[f"{col}_word_count"].mean()),
                    "std_word_count": float(sample_data[f"{col}_word_count"].std())
                }
                
                # Check correlation with disagreement severity
                length_corr = sample_data[[f"{col}_length", "severity"]].corr().iloc[0, 1]
                word_corr = sample_data[[f"{col}_word_count", "severity"]].corr().iloc[0, 1]
                
                text_features["length_analysis"][f"{col}_correlations"] = {
                    "length_vs_severity": float(length_corr) if not pd.isna(length_corr) else 0,
                    "word_count_vs_severity": float(word_corr) if not pd.isna(word_corr) else 0
                }
                
        # 2. Complexity Analysis (if we have text)
        if "source_text" in sample_data.columns:
            # Average sentence length
            sample_data["avg_sentence_length"] = sample_data["source_text"].astype(str).apply(
                lambda x: np.mean([len(s.split()) for s in re.split(r'[.!?]', x) if s.strip()])
                if len(re.split(r'[.!?]', x)) > 0 else 0
            )
            
            # Average word length
            sample_data["avg_word_length"] = sample_data["source_text"].astype(str).apply(
                lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
            )
            
            # Analyze if complexity correlates with disagreement severity
            text_features["complexity_analysis"] = {
                "avg_sentence_length": float(sample_data["avg_sentence_length"].mean()),
                "avg_word_length": float(sample_data["avg_word_length"].mean()),
                "sentence_length_vs_severity": float(sample_data[["avg_sentence_length", "severity"]].corr().iloc[0, 1]),
                "word_length_vs_severity": float(sample_data[["avg_word_length", "severity"]].corr().iloc[0, 1])
            }
            
        # 3. Term Frequency Analysis (for the most severe disagreements)
        if any(col in sample_data.columns for col in text_cols):
            # Get texts to analyze
            texts = []
            for col in text_cols:
                if col in sample_data.columns:
                    texts.extend(sample_data[col].astype(str).tolist())
                    
            if texts:
                # Limit corpus size
                max_corpus_size = min(1000, len(texts))
                texts = texts[:max_corpus_size]
                
                # Perform TF-IDF analysis
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get most important terms
                    tfidf_sums = tfidf_matrix.sum(axis=0).A1
                    top_indices = tfidf_sums.argsort()[-20:][::-1]
                    top_terms = {feature_names[i]: float(tfidf_sums[i]) for i in top_indices}
                    
                    text_features["term_frequency"] = top_terms
                except Exception as e:
                    logger.warning(f"Error in TF-IDF analysis: {str(e)}")
                
        # 4. Pattern Analysis
        # Extract patterns that might cause disagreements, like punctuation, numbers, special characters
        pattern_checkers = {
            "contains_numbers": lambda x: bool(re.search(r'\d', x)),
            "contains_special_chars": lambda x: bool(re.search(r'[^\w\s]', x)),
            "contains_uppercase": lambda x: any(c.isupper() for c in x),
            "contains_urls": lambda x: bool(re.search(r'https?://\S+', x)),
            "contains_html": lambda x: bool(re.search(r'<[^>]+>', x))
        }
        
        pattern_results = {}
        for col in text_cols:
            if col in sample_data.columns:
                # Apply pattern checkers
                for pattern_name, checker_func in pattern_checkers.items():
                    col_pattern = f"{col}_{pattern_name}"
                    sample_data[col_pattern] = sample_data[col].astype(str).apply(checker_func)
                    
                    # Get counts
                    pattern_count = sample_data[col_pattern].sum()
                    pattern_percentage = pattern_count / len(sample_data) * 100
                    
                    # Calculate if pattern correlates with disagreement
                    # We convert boolean to int for correlation calculation
                    pattern_corr = sample_data[[col_pattern, "severity"]].astype({col_pattern: int}).corr().iloc[0, 1]
                    
                    pattern_results[col_pattern] = {
                        "count": int(pattern_count),
                        "percentage": float(pattern_percentage),
                        "correlation_with_severity": float(pattern_corr) if not pd.isna(pattern_corr) else 0
                    }
        
        text_features["pattern_analysis"] = pattern_results
        
        # Store results
        self.text_features = text_features
        
        logger.info(f"Analyzed text features in {len(sample_data)} disagreement samples")
        
        return text_features 
    
    def generate_disagreement_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive disagreement report.
        
        Returns:
            Dictionary with disagreement analysis report
        """
        if self.disagreements is None:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
            
        # Ensure we have all analysis completed
        if not self.disagreement_categories:
            self.classify_disagreements()
            
        if self.disagreement_scores is None or self.disagreement_scores.empty:
            self.score_disagreements()
            
        if not self.language_patterns:
            self.analyze_language_patterns()
            
        if not self.text_features:
            self.analyze_text_features()
            
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_samples": int(len(self.data)) if self.data is not None else 0,
                "total_disagreements": int(len(self.disagreements)),
                "disagreement_rate": float(len(self.disagreements) / len(self.data)) if self.data is not None and len(self.data) > 0 else 0,
                "avg_severity": float(self.disagreements["severity"].mean()) if "severity" in self.disagreements else 0,
                "max_severity": float(self.disagreements["severity"].max()) if "severity" in self.disagreements else 0,
            },
            "disagreement_types": {
                category: {
                    "count": int(len(df)),
                    "percentage": float(len(df) / len(self.disagreements) * 100),
                    "avg_severity": float(df["severity"].mean()) if "severity" in df else 0
                }
                for category, df in self.disagreement_categories.items()
            },
            "language_patterns": self.language_patterns.get("summary", {}),
            "text_features": {
                "length_impact": {
                    k: v for k, v in self.text_features.get("length_analysis", {}).items()
                    if k.endswith("_correlations")
                },
                "complexity_impact": {
                    k: v for k, v in self.text_features.get("complexity_analysis", {}).items()
                    if k.endswith("_severity")
                },
                "significant_patterns": {
                    pattern: details["correlation_with_severity"]
                    for pattern, details in self.text_features.get("pattern_analysis", {}).items()
                    if abs(details.get("correlation_with_severity", 0)) > 0.2  # Threshold for significance
                }
            },
            "recommendations": self._generate_recommendations(),
            "most_severe_cases": self.get_most_severe_disagreements(10)
        }
        
        # Calculate disagreement metrics
        disagreement_metrics = self.calculate_disagreement_metrics()
        report["disagreement_metrics"] = disagreement_metrics
        
        # Get systematic patterns
        systematic_patterns = self.get_systematic_disagreement_patterns()
        if systematic_patterns:
            report["systematic_patterns"] = systematic_patterns
            
        logger.info(f"Generated comprehensive disagreement report")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on disagreement analysis.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Only generate if we have disagreements
        if self.disagreements is None or self.disagreements.empty:
            return ["No disagreements found to generate recommendations."]
            
        # Recommendation based on disagreement rate
        if self.data is not None:
            disagreement_rate = len(self.disagreements) / len(self.data)
            if disagreement_rate > 0.2:
                recommendations.append(
                    f"High disagreement rate ({disagreement_rate:.1%}): "
                    f"Consider reviewing quality assessment approach"
                )
            elif disagreement_rate > 0.1:
                recommendations.append(
                    f"Moderate disagreement rate ({disagreement_rate:.1%}): "
                    f"Consider targeted improvements in specific areas"
                )
            else:
                recommendations.append(
                    f"Low disagreement rate ({disagreement_rate:.1%}): "
                    f"System appears to be working well, focus on specific edge cases"
                )
        
        # Recommendation based on consistent disagreement metrics
        if "embedding_similarity_vs_groq_quality_score_diff" in self.disagreements.columns:
            embedding_higher = (
                self.disagreements["disagreement_type"] == "embedding_similarity_higher"
            ).mean() > 0.6
            
            groq_higher = (
                self.disagreements["disagreement_type"] == "groq_quality_score_higher"
            ).mean() > 0.6
            
            if embedding_higher:
                recommendations.append(
                    "Embedding similarity consistently rates translations higher than Groq. "
                    "Consider recalibrating embedding similarity thresholds downward."
                )
            elif groq_higher:
                recommendations.append(
                    "Groq consistently rates translations higher than embedding similarity. "
                    "Consider recalibrating embedding similarity thresholds upward."
                )
        
        # Recommendations based on language patterns
        if self.language_patterns and "high_disagreement_languages" in self.language_patterns.get("summary", {}):
            high_disagreement_langs = self.language_patterns["summary"]["high_disagreement_languages"]
            if high_disagreement_langs:
                lang_list = ", ".join([item["language_pair"] for item in high_disagreement_langs[:3]])
                recommendations.append(
                    f"High disagreement rates for language pairs: {lang_list}. "
                    f"Consider creating language-specific thresholds for these pairs."
                )
        
        # Recommendations based on text features
        if self.text_features:
            # Length recommendations
            length_correlations = self.text_features.get("length_analysis", {}).get("source_text_correlations", {})
            if length_correlations.get("length_vs_severity", 0) > 0.3:
                recommendations.append(
                    "Longer texts show higher disagreement rates. "
                    "Consider segment-based analysis for long texts."
                )
                
            # Pattern recommendations
            pattern_analysis = self.text_features.get("pattern_analysis", {})
            significant_patterns = [
                (pattern, details["correlation_with_severity"])
                for pattern, details in pattern_analysis.items()
                if abs(details.get("correlation_with_severity", 0)) > 0.3
            ]
            
            if significant_patterns:
                top_pattern, corr = max(significant_patterns, key=lambda x: abs(x[1])) if significant_patterns else (None, 0)
                if top_pattern:
                    pattern_name = top_pattern.split("_", 1)[1]  # Remove column prefix
                    recommendations.append(
                        f"Texts containing {pattern_name} show {'higher' if corr > 0 else 'lower'} disagreement. "
                        f"Consider specialized handling for these cases."
                    )
        
        return recommendations
            
    def save_disagreement_cases(self, output_dir: str) -> str:
        """
        Save disagreement cases to files.
        
        Args:
            output_dir: Directory to save disagreement files
            
        Returns:
            Path to the saved files
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return ""
            
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete disagreement data
        full_path = output_path / f"all_disagreements_{timestamp}.json"
        
        # Convert to dict for JSON serialization
        disagreement_dict = self.disagreements.to_dict(orient="records")
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_disagreements": len(self.disagreements),
                "disagreement_threshold": self.threshold,
                "timestamp": timestamp,
                "disagreements": disagreement_dict
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved all {len(self.disagreements)} disagreement cases to {full_path}")
        
        # Save categorized disagreements
        if self.disagreement_categories:
            categories_dir = output_path / "categories"
            categories_dir.mkdir(exist_ok=True)
            
            for category, df in self.disagreement_categories.items():
                if not df.empty:
                    category_path = categories_dir / f"{category}_{timestamp}.json"
                    with open(category_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "category": category,
                            "count": len(df),
                            "timestamp": timestamp,
                            "disagreements": df.to_dict(orient="records")
                        }, f, indent=2, ensure_ascii=False)
                        
                    logger.info(f"Saved {len(df)} {category} disagreement cases to {category_path}")
        
        # Save most severe disagreements
        severe_cases = self.get_most_severe_disagreements(20)
        if severe_cases:
            severe_path = output_path / f"severe_disagreements_{timestamp}.json"
            with open(severe_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "count": len(severe_cases),
                    "timestamp": timestamp,
                    "disagreements": severe_cases
                }, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(severe_cases)} severe disagreement cases to {severe_path}")
        
        # Save disagreement report
        report = self.generate_disagreement_report()
        report_path = output_path / f"disagreement_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved disagreement report to {report_path}")
            
        return str(output_path)
    
    def visualize_disagreements(self) -> Dict[str, Any]:
        """
        Create visualization data for disagreements.
        
        Returns:
            Dictionary with visualization data
            (actual plotting is done by the QualityLearningEngine)
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
            
        # Prepare visualization data
        viz_data = {}
        
        # 1. Scatter plot data - embedding_similarity vs groq_quality_score
        if all(col in self.disagreements.columns for col in ["original_embedding_similarity", "original_groq_quality_score"]):
            viz_data["scatter_data"] = {
                "embedding_similarity": self.disagreements["original_embedding_similarity"].tolist(),
                "groq_quality_score": self.disagreements["original_groq_quality_score"].tolist(),
                "severity": self.disagreements["severity"].tolist()
            }
            
        # 2. Disagreement types distribution
        if "disagreement_type" in self.disagreements.columns:
            type_counts = self.disagreements["disagreement_type"].value_counts().to_dict()
            viz_data["type_distribution"] = type_counts
            
        # 3. Severity distribution
        if "severity" in self.disagreements.columns:
            severity_bins = [0, 0.3, 0.5, 0.7, 1.0]
            severity_labels = ["Low", "Medium", "High", "Extreme"]
            
            self.disagreements["severity_category"] = pd.cut(
                self.disagreements["severity"], 
                bins=severity_bins, 
                labels=severity_labels
            )
            
            severity_counts = self.disagreements["severity_category"].value_counts().to_dict()
            viz_data["severity_distribution"] = {str(k): v for k, v in severity_counts.items()}
            
        # 4. Language pair distribution (if available)
        if "language_pair" in self.disagreements.columns:
            lang_counts = self.disagreements["language_pair"].value_counts().head(10).to_dict()
            viz_data["language_distribution"] = lang_counts
            
        logger.info(f"Prepared visualization data for disagreements")
            
        return viz_data 
    
    def recommend_investigations(self, max_cases: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend cases for manual investigation.
        
        Args:
            max_cases: Maximum number of cases to recommend
            
        Returns:
            List of recommended cases with context
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return []
            
        # Ensure cases are scored
        if self.disagreement_scores is None or self.disagreement_scores.empty:
            scored_disagreements = self.score_disagreements()
        else:
            scored_disagreements = self.disagreement_scores
            
        # Sort by priority score
        if "priority_score" in scored_disagreements.columns:
            sorted_cases = scored_disagreements.sort_values("priority_score", ascending=False)
        else:
            sorted_cases = scored_disagreements.sort_values("severity", ascending=False)
            
        # Get diverse set of cases: different types, different language pairs
        recommended_cases = []
        added_types = set()
        added_languages = set()
        
        # First add the most severe cases regardless of type diversity
        top_severe = sorted_cases.head(max_cases // 3)
        for _, row in top_severe.iterrows():
            case = row.to_dict()
            case["reason"] = "Highest severity disagreement"
            recommended_cases.append(case)
            
            # Track added types and languages
            added_types.add(case.get("disagreement_type", "unknown"))
            added_languages.add(case.get("language_pair", "unknown"))
            
        # Then add cases to maximize type diversity
        remaining = sorted_cases.iloc[len(recommended_cases):]
        for disagree_type, group in remaining.groupby("disagreement_type"):
            if disagree_type not in added_types and len(recommended_cases) < max_cases:
                case = group.iloc[0].to_dict()
                case["reason"] = f"Representative of disagreement type: {disagree_type}"
                recommended_cases.append(case)
                
                added_types.add(disagree_type)
                added_languages.add(case.get("language_pair", "unknown"))
                
        # Then add cases to maximize language diversity
        remaining = sorted_cases.iloc[len(recommended_cases):]
        if "language_pair" in remaining.columns:
            for lang_pair, group in remaining.groupby("language_pair"):
                if lang_pair not in added_languages and len(recommended_cases) < max_cases:
                    case = group.iloc[0].to_dict()
                    case["reason"] = f"Representative of language pair: {lang_pair}"
                    recommended_cases.append(case)
                    
                    added_types.add(case.get("disagreement_type", "unknown"))
                    added_languages.add(lang_pair)
        
        # Add context to all recommendations
        recommendations = []
        for case in recommended_cases:
            # Extract only the most relevant fields
            rec = {
                "disagreement_type": case.get("disagreement_type"),
                "severity": case.get("severity"),
                "priority_score": case.get("priority_score"),
                "reason": case.get("reason"),
                "metrics": {
                    case.get("metric1"): case.get(f"original_{case.get('metric1')}"),
                    case.get("metric2"): case.get(f"original_{case.get('metric2')}")
                },
                "language_pair": case.get("language_pair")
            }
            
            # Add text if available
            for text_field in ["source_text", "translated_text"]:
                if text_field in case and case[text_field]:
                    # Truncate long texts
                    text = case[text_field]
                    if isinstance(text, str) and len(text) > 200:
                        rec[text_field] = text[:197] + "..."
                    else:
                        rec[text_field] = text
                        
            recommendations.append(rec)
            
        logger.info(f"Recommended {len(recommendations)} cases for investigation")
        
        return recommendations
    
    def calculate_disagreement_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics about disagreements.
        
        Returns:
            Dictionary with disagreement metrics
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
            
        # Basic metrics about disagreement extent
        if self.data is not None:
            disagreement_rate = len(self.disagreements) / len(self.data)
        else:
            disagreement_rate = None
            
        # Create metrics dictionary
        disagreement_metrics = {
            "total_disagreements": int(len(self.disagreements)),
            "disagreement_rate": float(disagreement_rate) if disagreement_rate is not None else None,
            "avg_severity": float(self.disagreements["severity"].mean()),
            "median_severity": float(self.disagreements["severity"].median()),
            "max_severity": float(self.disagreements["severity"].max())
        }
        
        # Average difference between metrics
        if "embedding_similarity_vs_groq_quality_score_diff" in self.disagreements.columns:
            disagreement_metrics["avg_sim_groq_diff"] = float(
                self.disagreements["embedding_similarity_vs_groq_quality_score_diff"].mean()
            )
            
        # Calculate disagreement type distribution
        if "disagreement_type" in self.disagreements.columns:
            type_counts = self.disagreements["disagreement_type"].value_counts()
            type_percentages = type_counts / len(self.disagreements) * 100
            
            disagreement_metrics["disagreement_types"] = {
                type_name: {
                    "count": int(count),
                    "percentage": float(type_percentages[type_name])
                }
                for type_name, count in type_counts.items()
            }
            
        # Calculate language-specific metrics if available
        if "language_pair" in self.disagreements.columns:
            lang_metrics = self.disagreements.groupby("language_pair")["severity"].agg(
                ["count", "mean", "max"]
            )
            
            disagreement_metrics["language_metrics"] = {
                lang_pair: {
                    "count": int(row["count"]),
                    "avg_severity": float(row["mean"]),
                    "max_severity": float(row["max"])
                }
                for lang_pair, row in lang_metrics.iterrows()
            }
            
        self.disagreement_metrics = disagreement_metrics
        
        logger.info("Calculated disagreement metrics")
        
        return disagreement_metrics
    
    def get_most_severe_disagreements(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most severe disagreement cases.
        
        Args:
            n: Number of cases to return
            
        Returns:
            List of the most severe disagreement cases
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return []
            
        # Ensure cases are scored
        if self.disagreement_scores is not None and not self.disagreement_scores.empty:
            df = self.disagreement_scores
            sort_col = "priority_score" if "priority_score" in df.columns else "severity"
        else:
            df = self.disagreements
            sort_col = "severity"
            
        # Get top N severe cases
        top_cases = df.sort_values(sort_col, ascending=False).head(n)
        
        # Convert to list of dictionaries
        result = []
        for _, row in top_cases.iterrows():
            case = {}
            
            # Add key information
            for field in ["metric1", "metric2", "disagreement_type", "severity", "priority_score"]:
                if field in row:
                    case[field] = row[field]
            
            # Add original metric values
            if row["metric1"] and f"original_{row['metric1']}" in row:
                case[row["metric1"]] = row[f"original_{row['metric1']}"]
                
            if row["metric2"] and f"original_{row['metric2']}" in row:
                case[row["metric2"]] = row[f"original_{row['metric2']}"]
                
            # Add language information if available
            for field in ["source_language", "target_language", "language_pair"]:
                if field in row:
                    case[field] = row[field]
                    
            # Add text samples if available (truncated)
            for field in ["source_text", "translated_text"]:
                if field in row and isinstance(row[field], str):
                    text = row[field]
                    if len(text) > 200:
                        case[field] = text[:197] + "..."
                    else:
                        case[field] = text
                        
            result.append(case)
            
        logger.info(f"Retrieved {len(result)} most severe disagreement cases")
        
        return result
    
    def get_systematic_disagreement_patterns(self) -> Dict[str, Any]:
        """
        Get patterns of systematic disagreement.
        
        Returns:
            Dictionary with systematic disagreement patterns
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return {}
            
        # Check for embedding vs groq disagreements
        has_both_metrics = all(
            col in self.disagreements.columns for col in 
            ["original_embedding_similarity", "original_groq_quality_score"]
        )
        
        if not has_both_metrics:
            logger.warning("Missing required metrics for pattern analysis")
            return {}
            
        # Create a copy to avoid modifying original
        df = self.disagreements.copy()
        
        # 1. Direction Analysis - Does one metric consistently rate higher?
        embedding_higher_count = sum(
            (df["original_embedding_similarity"] > df["original_groq_quality_score"]) &
            ((df["metric1"] == "embedding_similarity") | (df["metric2"] == "embedding_similarity")) &
            ((df["metric1"] == "groq_quality_score") | (df["metric2"] == "groq_quality_score"))
        )
        
        groq_higher_count = sum(
            (df["original_embedding_similarity"] < df["original_groq_quality_score"]) &
            ((df["metric1"] == "embedding_similarity") | (df["metric2"] == "embedding_similarity")) &
            ((df["metric1"] == "groq_quality_score") | (df["metric2"] == "groq_quality_score"))
        )
        
        total_comparisons = embedding_higher_count + groq_higher_count
        
        direction_pattern = {
            "embedding_higher": {
                "count": int(embedding_higher_count),
                "percentage": float(embedding_higher_count / total_comparisons * 100) if total_comparisons > 0 else 0
            },
            "groq_higher": {
                "count": int(groq_higher_count),
                "percentage": float(groq_higher_count / total_comparisons * 100) if total_comparisons > 0 else 0
            },
            "dominant_direction": "embedding_higher" if embedding_higher_count > groq_higher_count else "groq_higher"
        }
        
        # 2. Magnitude Analysis - Is disagreement worse at certain quality levels?
        # Create quality bins
        quality_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        quality_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        
        # Normalize quality scores to 0-1
        df["norm_embedding"] = df["original_embedding_similarity"]
        
        if df["original_groq_quality_score"].min() != df["original_groq_quality_score"].max():
            df["norm_groq"] = (
                (df["original_groq_quality_score"] - df["original_groq_quality_score"].min()) /
                (df["original_groq_quality_score"].max() - df["original_groq_quality_score"].min())
            )
        else:
            df["norm_groq"] = 0.5
        
        # Calculate average quality
        df["avg_quality"] = (df["norm_embedding"] + df["norm_groq"]) / 2
        
        # Bin by quality
        df["quality_bin"] = pd.cut(df["avg_quality"], bins=quality_bins, labels=quality_labels)
        
        # Calculate disagreement by quality bin
        quality_disagreements = df.groupby("quality_bin", observed=False)["severity"].agg(["count", "mean", "max"]).to_dict()
        
        # 3. Patterns by Language (if available)
        language_patterns = {}
        if "language_pair" in df.columns:
            for lang_pair, group in df.groupby("language_pair"):
                if len(group) >= 5:  # Only analyze if we have enough samples
                    # Calculate which metric tends to be higher for this language
                    embedding_higher = (
                        (group["original_embedding_similarity"] > group["original_groq_quality_score"]) &
                        ((group["metric1"] == "embedding_similarity") | (group["metric2"] == "embedding_similarity")) &
                        ((group["metric1"] == "groq_quality_score") | (group["metric2"] == "groq_quality_score"))
                    ).mean()
                    
                    language_patterns[lang_pair] = {
                        "count": int(len(group)),
                        "embedding_higher_rate": float(embedding_higher),
                        "groq_higher_rate": float(1 - embedding_higher),
                        "avg_severity": float(group["severity"].mean()),
                        "dominant_metric": "embedding_similarity" if embedding_higher > 0.5 else "groq_quality_score"
                    }
        
        # Compile all systematic patterns
        systematic_patterns = {
            "direction_pattern": direction_pattern,
            "quality_level_pattern": quality_disagreements,
            "language_specific_patterns": language_patterns
        }
        
        logger.info("Analyzed systematic disagreement patterns")
        
        return systematic_patterns
    
    def extract_context_for_disagreements(self) -> pd.DataFrame:
        """
        Extract context for disagreements.
        
        Returns:
            DataFrame with disagreements and extracted context
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return pd.DataFrame()
            
        # Check if text fields are available
        text_fields = ["source_text", "translated_text"]
        if not any(field in self.disagreements.columns for field in text_fields):
            logger.warning("No text fields available for context extraction")
            return self.disagreements.copy()
            
        # Create copy to avoid modifying original
        context_df = self.disagreements.copy()
        
        # Extract text snippets from source/translations
        for field in text_fields:
            if field in context_df.columns:
                # Extract snippets (first 50 chars, middle 50 chars, last 50 chars)
                context_df[f"{field}_start"] = context_df[field].astype(str).apply(
                    lambda x: x[:50].strip() if len(x) > 0 else ""
                )
                
                context_df[f"{field}_middle"] = context_df[field].astype(str).apply(
                    lambda x: x[max(0, len(x)//2-25):min(len(x), len(x)//2+25)].strip() if len(x) > 50 else ""
                )
                
                context_df[f"{field}_end"] = context_df[field].astype(str).apply(
                    lambda x: x[-50:].strip() if len(x) > 50 else ""
                )
                
                # Remove full text to save space
                if len(context_df[field].iloc[0]) > 500:
                    context_df[field] = context_df[field].apply(
                        lambda x: x[:497] + "..." if len(x) > 500 else x
                    )
        
        logger.info(f"Extracted context for {len(context_df)} disagreement cases")
        
        return context_df
    
    def predict_disagreement_correction(self, model=None) -> pd.DataFrame:
        """
        Predict correct quality for disagreements using a model.
        
        Args:
            model: Pre-trained model to use for prediction (optional)
            
        Returns:
            DataFrame with predicted corrections
        """
        if self.disagreements is None or self.disagreements.empty:
            logger.warning("No disagreements found. Call identify_disagreements() first.")
            return pd.DataFrame()
            
        # If no model provided, we'll use a simple heuristic approach
        # based on the systematic patterns we've discovered
        
        # Get copy of disagreements
        correction_df = self.disagreements.copy()
        
        # Check if we have the necessary columns
        has_embedding = "original_embedding_similarity" in correction_df.columns
        has_groq = "original_groq_quality_score" in correction_df.columns
        
        if not (has_embedding and has_groq):
            logger.warning("Missing necessary columns for correction prediction")
            return correction_df
            
        # Get systematic patterns to inform our correction
        systematic_patterns = self.get_systematic_disagreement_patterns()
        
        # Default weights if no systematic patterns
        embedding_weight = 0.5
        groq_weight = 0.5
        
        # Adjust weights based on patterns
        if systematic_patterns:
            direction = systematic_patterns.get("direction_pattern", {}).get("dominant_direction")
            if direction == "embedding_higher":
                # If embedding is systematically higher, give it less weight
                embedding_weight = 0.4
                groq_weight = 0.6
            elif direction == "groq_higher":
                # If groq is systematically higher, give it less weight
                embedding_weight = 0.6
                groq_weight = 0.4
        
        # Calculate weighted average prediction
        correction_df["predicted_quality"] = (
            correction_df["original_embedding_similarity"] * embedding_weight +
            correction_df["original_groq_quality_score"] * groq_weight  
        )
        
        # Calculate how much each metric deviates from prediction
        correction_df["embedding_deviation"] = abs(
            correction_df["original_embedding_similarity"] - correction_df["predicted_quality"]
        )
        
        correction_df["groq_deviation"] = abs(
            correction_df["original_groq_quality_score"] - correction_df["predicted_quality"]
        )
        
        # Identify which metric is likely more accurate (lower deviation)
        correction_df["likely_more_accurate"] = np.where(
            correction_df["embedding_deviation"] < correction_df["groq_deviation"],
            "embedding_similarity",
            "groq_quality_score"
        )
        
        # Calculate confidence in the prediction
        max_deviation = correction_df[["embedding_deviation", "groq_deviation"]].max(axis=1)
        correction_df["prediction_confidence"] = 1 - (max_deviation / 0.5).clip(0, 1)
        
        logger.info(f"Predicted quality corrections for {len(correction_df)} disagreement cases")
        logger.info(f"Average predicted quality: {correction_df['predicted_quality'].mean():.4f}")
        
        # Log which metric tends to be more accurate
        embedding_accurate_rate = (correction_df["likely_more_accurate"] == "embedding_similarity").mean()
        logger.info(f"Embedding similarity more accurate in {embedding_accurate_rate:.1%} of cases")
        logger.info(f"Average prediction confidence: {correction_df['prediction_confidence'].mean():.4f}")
        
        return correction_df 