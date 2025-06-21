"""
Optimizes quality thresholds based on learned correlation data.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from config_manager import ConfigManager

# Configure logging using unified configuration
from logger_config import get_logger
logger = get_logger("ThresholdOptimizer", "quality_learning")

class ThresholdOptimizer:
    """
    Optimizes quality thresholds from learned correlation data.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the threshold optimizer.
        
        Args:
            config_manager: Optional configuration manager
        """
        self.config_manager = config_manager or ConfigManager()
        self.correlation_data = None
        self.thresholds = {}
        self.quality_weights = {}
        self.language_thresholds = {}
        
    def load_correlation_data(self, data: Dict[str, Any]) -> None:
        """
        Load correlation data for threshold optimization.
        
        Args:
            data: Dictionary with correlation analysis results
        """
        self.correlation_data = data
        logger.info("Loaded correlation data for threshold optimization")
        
        # Log key aspects of loaded data
        if "correlation_matrix" in data:
            logger.info("Correlation data includes correlation matrix")
        
        if "similarity_thresholds" in data:
            logger.info(f"Correlation data includes predefined similarity thresholds")
            
        if "language_patterns" in data:
            n_langs = len(data["language_patterns"])
            logger.info(f"Correlation data includes patterns for {n_langs} language pairs")
            
    def find_optimal_thresholds(
        self, 
        metric: str, 
        quality_levels: int = 5
    ) -> Dict[str, List[float]]:
        """
        Find optimal thresholds for quality levels.
        
        Args:
            metric: Metric to find thresholds for
            quality_levels: Number of quality levels to define
            
        Returns:
            Dictionary with optimized thresholds
        """
        if not self.correlation_data:
            logger.warning("No correlation data loaded. Call load_correlation_data() first.")
            return {}
            
        # Check if thresholds already computed in correlation data
        if "similarity_thresholds" in self.correlation_data and metric in self.correlation_data["similarity_thresholds"]:
            thresholds = self.correlation_data["similarity_thresholds"][metric]
            logger.info(f"Using pre-computed thresholds from correlation analysis for {metric}")
            
            result = {
                metric: thresholds,
                "quality_levels": quality_levels,
                "method": "pre-computed"
            }
            
            self.thresholds[metric] = result
            return result
            
        # Check if we have cluster data
        if "cluster_stats" in self.correlation_data:
            clusters = self.correlation_data["cluster_stats"]
            
            # Sort clusters by quality mean
            sorted_clusters = sorted(clusters, key=lambda x: x["quality_mean"])
            
            if len(sorted_clusters) > 1:
                thresholds = []
                
                # Get boundary values for clusters
                for i in range(len(sorted_clusters)):
                    if i == 0:
                        # Lowest threshold is the minimum value
                        thresholds.append(sorted_clusters[i]["similarity_min"])
                    else:
                        # Other thresholds are midpoints between cluster means
                        prev_mean = sorted_clusters[i-1]["similarity_mean"]
                        curr_mean = sorted_clusters[i]["similarity_mean"]
                        thresholds.append((prev_mean + curr_mean) / 2)
                
                # Add the maximum as the final threshold
                thresholds.append(sorted_clusters[-1]["similarity_max"])
                
                logger.info(f"Derived {len(thresholds)} thresholds from cluster analysis for {metric}")
                
                result = {
                    metric: thresholds,
                    "quality_levels": len(thresholds),
                    "method": "cluster-based"
                }
                
                self.thresholds[metric] = result
                return result
                
        # If we reach here, we need to calculate thresholds directly
        # We'll use KMeans to find natural groupings in the data
        if "correlation_matrix" in self.correlation_data:
            matrix_data = self.correlation_data["correlation_matrix"]
            
            # Convert to DataFrame if necessary
            if isinstance(matrix_data, dict):
                matrix = pd.DataFrame(matrix_data)
            else:
                matrix = matrix_data
                
            # Check if we have the required metrics
            if metric not in matrix.columns or "groq_quality_score" not in matrix.columns:
                logger.warning(f"Required metrics not found in correlation matrix")
                return self._compute_default_thresholds(metric, quality_levels)
                
            # Get the data points
            data_points = pd.DataFrame({
                "similarity": matrix[metric],
                "quality": matrix["groq_quality_score"]
            }).dropna()
            
            if len(data_points) < quality_levels:
                logger.warning(f"Not enough data points for {quality_levels} thresholds")
                return self._compute_default_thresholds(metric, quality_levels)
                
            # Use KMeans to find clusters
            X = data_points[["similarity"]].values
            kmeans = KMeans(n_clusters=quality_levels, random_state=42)
            kmeans.fit(X)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_.flatten()
            centers.sort()
            
            # Calculate thresholds as midpoints between centers
            thresholds = [centers[0] - 0.1]  # Start below the first center
            for i in range(len(centers) - 1):
                thresholds.append((centers[i] + centers[i+1]) / 2)
            thresholds.append(centers[-1] + 0.1)  # End above the last center
            
            logger.info(f"Calculated {len(thresholds)} KMeans-based thresholds for {metric}")
            
            result = {
                metric: thresholds,
                "quality_levels": quality_levels,
                "method": "kmeans"
            }
            
            self.thresholds[metric] = result
            return result
            
        # If we still can't calculate thresholds, use default
        logger.warning(f"Insufficient correlation data for threshold calculation, using defaults")
        return self._compute_default_thresholds(metric, quality_levels)
        
    def _compute_default_thresholds(self, metric: str, quality_levels: int) -> Dict[str, List[float]]:
        """
        Compute default thresholds when correlation data is insufficient.
        
        Args:
            metric: Metric to find thresholds for
            quality_levels: Number of quality levels
            
        Returns:
            Dictionary with default thresholds
        """
        # For embedding similarity, use equally spaced thresholds from 0.3 to 1.0
        if metric == "embedding_similarity":
            step = (1.0 - 0.3) / (quality_levels - 1)
            thresholds = [0.3 + step * i for i in range(quality_levels)]
            
        # For quality score, use equally spaced thresholds from 1.0 to 10.0
        elif metric == "groq_quality_score":
            step = (10.0 - 1.0) / (quality_levels - 1)
            thresholds = [1.0 + step * i for i in range(quality_levels)]
            
        # For other metrics, use 0.2 to 1.0 range
        else:
            step = 0.8 / (quality_levels - 1)
            thresholds = [0.2 + step * i for i in range(quality_levels)]
            
        logger.info(f"Using default thresholds for {metric}: {thresholds}")
        
        result = {
            metric: thresholds,
            "quality_levels": quality_levels,
            "method": "default"
        }
        
        self.thresholds[metric] = result
        return result
    
    def validate_thresholds(self, thresholds: Dict[str, List[float]], test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate threshold effectiveness on test data.
        
        Args:
            thresholds: Thresholds to validate
            test_data: Test data for validation
            
        Returns:
            Dictionary with validation metrics
        """
        if thresholds is None or not thresholds:
            logger.warning("No thresholds provided for validation")
            return {}
            
        # Get primary metric
        primary_metric = next(iter(thresholds.keys())) if isinstance(thresholds, dict) else "embedding_similarity"
        
        if primary_metric not in thresholds:
            logger.warning(f"Primary metric {primary_metric} not found in thresholds")
            return {}
            
        # Check if test data contains the metric and quality score
        if primary_metric not in test_data.columns or "groq_quality_score" not in test_data.columns:
            logger.warning(f"Required metrics not found in test data")
            return {}
            
        # Bin data according to thresholds
        bins = [-float('inf')] + thresholds[primary_metric] + [float('inf')]
        test_data['quality_bin'] = pd.cut(test_data[primary_metric], bins=bins, labels=False)
        
        # Calculate correlation between bins and actual quality
        bin_quality_corr = test_data['quality_bin'].corr(test_data['groq_quality_score'])
        
        # Calculate average quality within each bin
        bin_stats = test_data.groupby('quality_bin')['groq_quality_score'].agg(['mean', 'std', 'count'])
        
        # Calculate the standard deviation between bin means (higher is better)
        between_bin_std = np.std(bin_stats['mean'])
        
        # Calculate average within-bin std (lower is better)
        within_bin_std = np.mean(bin_stats['std'])
        
        # Calculate metrics
        validation_metrics = {
            "bin_quality_correlation": float(bin_quality_corr),
            "between_bin_std": float(between_bin_std),
            "within_bin_std": float(within_bin_std),
            "bin_separation_score": float(between_bin_std / within_bin_std if within_bin_std > 0 else 0),
            "bin_counts": bin_stats['count'].tolist()
        }
        
        logger.info(f"Threshold validation metrics: bin_quality_corr={bin_quality_corr:.4f}, separation_score={validation_metrics['bin_separation_score']:.4f}")
        
        return validation_metrics
    
    def optimize_quality_weights(self, metrics: List[str]) -> Dict[str, float]:
        """
        Optimize weights for quality metrics.
        
        Args:
            metrics: List of metrics to optimize weights for
            
        Returns:
            Dictionary mapping metrics to their optimal weights
        """
        if not self.correlation_data:
            logger.warning("No correlation data loaded. Call load_correlation_data() first.")
            return {}
            
        # Get correlation with quality
        quality_correlations = {}
        
        if "correlation_matrix" in self.correlation_data:
            matrix_data = self.correlation_data["correlation_matrix"]
            
            # Convert to DataFrame if necessary
            if isinstance(matrix_data, dict):
                matrix = pd.DataFrame(matrix_data)
            else:
                matrix = matrix_data
                
            # Get correlations with quality
            if "groq_quality_score" in matrix:
                for metric in metrics:
                    if metric in matrix.index and not pd.isna(matrix.loc[metric, "groq_quality_score"]):
                        quality_correlations[metric] = abs(matrix.loc[metric, "groq_quality_score"])
                        
        # If we have correlations, compute weights based on them
        if quality_correlations:
            # Normalize correlations to sum to 1.0
            total_corr = sum(quality_correlations.values())
            
            if total_corr > 0:
                weights = {metric: corr / total_corr for metric, corr in quality_correlations.items()}
            else:
                # Equal weights if no correlation
                weights = {metric: 1.0 / len(metrics) for metric in metrics}
                
            logger.info(f"Optimized quality weights based on correlations: {weights}")
            
            self.quality_weights = weights
            return weights
            
        # If correlation data doesn't contain what we need, check alternative source
        if "best_features" in self.correlation_data:
            best_features = self.correlation_data["best_features"]
            weights = {}
            
            for metric in metrics:
                if metric in best_features:
                    # Get absolute correlation value (first element in tuple)
                    weights[metric] = abs(best_features[metric][0])
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {metric: weight / total_weight for metric, weight in weights.items()}
            else:
                # Equal weights if no data
                weights = {metric: 1.0 / len(metrics) for metric in metrics}
                
            logger.info(f"Optimized quality weights based on best features: {weights}")
            
            self.quality_weights = weights
            return weights
            
        # If we still don't have data, use default weights
        logger.warning("Insufficient correlation data for weight optimization, using defaults")
        
        # Default weights prioritize Groq quality and then embedding similarity
        weights = {}
        for i, metric in enumerate(metrics):
            if metric == "groq_quality_score":
                weights[metric] = 0.6
            elif metric == "embedding_similarity":
                weights[metric] = 0.4
            else:
                weights[metric] = 0.0
                
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {metric: weight / total_weight for metric, weight in weights.items()}
        else:
            weights = {metric: 1.0 / len(metrics) for metric in metrics}
            
        logger.info(f"Using default quality weights: {weights}")
        
        self.quality_weights = weights
        return weights
    
    def apply_thresholds_to_config(self, config_path: Optional[str] = None) -> bool:
        """
        Update configuration with new thresholds.
        
        Args:
            config_path: Optional path to config file (uses config_manager default if None)
            
        Returns:
            True if config update was successful, False otherwise
        """
        if not self.thresholds:
            logger.warning("No thresholds available. Call find_optimal_thresholds() first.")
            return False
            
        if not self.quality_weights:
            logger.warning("No quality weights available. Call optimize_quality_weights() first.")
            
        # Get current config
        config = self.config_manager
        
        # Update quality thresholds
        for metric, threshold_data in self.thresholds.items():
            config_key = f"quality.thresholds.{metric}"
            
            # Only update if we have actual threshold values
            if metric in threshold_data and isinstance(threshold_data[metric], list):
                config.set(config_key, threshold_data[metric])
                logger.info(f"Updated config at {config_key} with new thresholds")
                
        # Update quality weights if available
        if self.quality_weights:
            for metric, weight in self.quality_weights.items():
                config_key = f"quality.weights.{metric}"
                config.set(config_key, weight)
                logger.info(f"Updated config at {config_key} with weight {weight:.4f}")
                
        # Update language-specific thresholds if available
        if self.language_thresholds:
            for lang_pair, thresholds in self.language_thresholds.items():
                config_key = f"quality.language_thresholds.{lang_pair}"
                config.set(config_key, thresholds)
                logger.info(f"Updated config at {config_key} with language-specific thresholds")
                
        # Save config
        if config_path:
            try:
                config.save_config(config_path)
                logger.info(f"Saved updated config to {config_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save config to {config_path}: {str(e)}")
                return False
        else:
            try:
                config.save_config()
                logger.info(f"Saved updated config to default path")
                return True
            except Exception as e:
                logger.error(f"Failed to save config to default path: {str(e)}")
                return False
    
    def generate_threshold_report(self) -> Dict[str, Any]:
        """
        Generate report on threshold optimization.
        
        Returns:
            Dictionary with threshold report
        """
        report = {
            "thresholds": self.thresholds,
            "quality_weights": self.quality_weights,
            "language_thresholds": self.language_thresholds,
            "recommendations": []
        }
        
        # Add recommendations
        if "embedding_similarity" in self.thresholds:
            similarity_thresholds = self.thresholds["embedding_similarity"]
            threshold_count = len(similarity_thresholds.get("embedding_similarity", []))
            method = similarity_thresholds.get("method", "unknown")
            
            report["recommendations"].append(
                f"Use {threshold_count} quality levels based on embedding similarity thresholds (method: {method})"
            )
            
        if self.quality_weights:
            top_metric = max(self.quality_weights.items(), key=lambda x: x[1])
            report["recommendations"].append(
                f"Prioritize {top_metric[0]} with weight {top_metric[1]:.2f} in quality assessment"
            )
            
        if self.language_thresholds:
            report["recommendations"].append(
                f"Apply language-specific thresholds for {len(self.language_thresholds)} language pairs"
            )
            
        logger.info(f"Generated threshold report with {len(report['recommendations'])} recommendations")
        
        return report
    
    def plot_threshold_effectiveness(self) -> Dict[str, Any]:
        """
        Create visualization data for threshold effectiveness.
        
        Returns:
            Dictionary with visualization data 
            (actual plotting is done by the QualityLearningEngine)
        """
        if not self.thresholds:
            logger.warning("No thresholds available for visualization.")
            return {}
            
        # Prepare visualization data for each metric
        visualization_data = {}
        
        for metric, threshold_data in self.thresholds.items():
            # Only proceed if we have actual thresholds
            if metric in threshold_data and isinstance(threshold_data[metric], list):
                thresholds = threshold_data[metric]
                
                # Create bins from thresholds
                bins = [-float('inf')] + thresholds + [float('inf')]
                bin_labels = [f"Level {i+1}" for i in range(len(bins)-1)]
                
                visualization_data[metric] = {
                    "thresholds": thresholds,
                    "bins": bins,
                    "bin_labels": bin_labels,
                    "method": threshold_data.get("method", "unknown")
                }
                
        logger.info(f"Prepared threshold visualization data for {len(visualization_data)} metrics")
        
        return visualization_data
    
    def get_language_specific_thresholds(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get thresholds specific to language pairs.
        
        Returns:
            Dictionary mapping language pairs to their threshold data
        """
        if not self.correlation_data:
            logger.warning("No correlation data loaded. Call load_correlation_data() first.")
            return {}
            
        # Check if we have language-specific patterns
        if "language_patterns" not in self.correlation_data:
            logger.warning("No language-specific patterns in correlation data.")
            return {}
            
        language_patterns = self.correlation_data["language_patterns"]
        
        # Only proceed if we have enough language pairs
        if not language_patterns or len(language_patterns) < 2:
            logger.info("Not enough language pairs for language-specific thresholds.")
            return {}
            
        # Calculate thresholds for each language pair
        language_thresholds = {}
        
        for lang_pair, pattern in language_patterns.items():
            # Check if we have enough data
            if "count" not in pattern or pattern["count"] < 10:
                continue
                
            # Check if we have similarity data
            if "embedding_similarity" not in pattern:
                continue
                
            similarity_data = pattern["embedding_similarity"]
            
            # Calculate thresholds
            # For simplicity, we use mean ± standard deviation as thresholds
            mean = similarity_data.get("mean", 0.5)
            std = similarity_data.get("std", 0.1)
            
            thresholds = {
                "embedding_similarity": [
                    max(0.0, mean - 2 * std),  # Very low
                    max(0.1, mean - std),     # Low
                    mean,                     # Medium
                    min(1.0, mean + std),     # High
                    min(1.0, mean + 2 * std)  # Very high
                ]
            }
            
            language_thresholds[lang_pair] = thresholds
            
        self.language_thresholds = language_thresholds
        
        logger.info(f"Generated language-specific thresholds for {len(language_thresholds)} language pairs")
        
        return language_thresholds
    
    def suggest_threshold_adjustments(self, current_thresholds: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Suggest adjustments to current thresholds.
        
        Args:
            current_thresholds: Current threshold values
            
        Returns:
            Dictionary with suggested threshold adjustments
        """
        if not self.thresholds:
            logger.warning("No optimized thresholds available. Call find_optimal_thresholds() first.")
            return {}
            
        suggested_adjustments = {}
        
        for metric, threshold_data in self.thresholds.items():
            # Skip if we don't have current thresholds for this metric
            if metric not in current_thresholds:
                continue
                
            # Skip if we don't have optimized thresholds
            if metric not in threshold_data or not isinstance(threshold_data[metric], list):
                continue
                
            current = current_thresholds[metric]
            optimized = threshold_data[metric]
            
            # We can only compare if both have the same number of thresholds
            if len(current) != len(optimized):
                logger.warning(f"Cannot suggest adjustments for {metric}: different number of thresholds")
                continue
                
            # Calculate adjustments
            adjustments = [opt - curr for curr, opt in zip(current, optimized)]
            
            suggested_adjustments[metric] = adjustments
            
            # Log suggestions
            logger.info(f"Suggested threshold adjustments for {metric}:")
            for i, adj in enumerate(adjustments):
                logger.info(f"  Level {i+1}: {current[i]:.4f} -> {optimized[i]:.4f} (adjustment: {adj:+.4f})")
                
        return suggested_adjustments
    
    def calculate_threshold_confidence(self, thresholds: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Calculate confidence in threshold values.
        
        Args:
            thresholds: Threshold values
            
        Returns:
            Dictionary with confidence scores for each threshold
        """
        if not self.correlation_data:
            logger.warning("No correlation data loaded. Call load_correlation_data() first.")
            return {}
            
        confidence_scores = {}
        
        for metric, threshold_list in thresholds.items():
            # Skip if no thresholds
            if not threshold_list:
                continue
                
            # Default confidence is moderate (0.5)
            confidence = [0.5] * len(threshold_list)
            
            # If we have cluster data, use it to calculate confidence
            if "cluster_stats" in self.correlation_data:
                clusters = self.correlation_data["cluster_stats"]
                
                # Sort clusters by similarity mean
                sorted_clusters = sorted(clusters, key=lambda x: x["similarity_mean"])
                
                if len(sorted_clusters) > 1:
                    # Calculate confidence based on cluster separation
                    for i, threshold in enumerate(threshold_list):
                        # Find closest clusters
                        closest_below = None
                        closest_above = None
                        
                        for cluster in sorted_clusters:
                            if cluster["similarity_mean"] <= threshold:
                                if closest_below is None or cluster["similarity_mean"] > closest_below["similarity_mean"]:
                                    closest_below = cluster
                            else:
                                if closest_above is None or cluster["similarity_mean"] < closest_above["similarity_mean"]:
                                    closest_above = cluster
                                    
                        # Can't calculate confidence if we don't have clusters on both sides
                        if closest_below is None or closest_above is None:
                            continue
                            
                        # Calculate confidence based on separation and cluster std
                        separation = closest_above["similarity_mean"] - closest_below["similarity_mean"]
                        avg_std = (closest_below["similarity_std"] + closest_above["similarity_std"]) / 2
                        
                        if avg_std > 0:
                            # Higher separation-to-noise ratio means higher confidence
                            confidence[i] = min(1.0, separation / (3 * avg_std))
                        else:
                            confidence[i] = 0.8  # Default high confidence if std is very low
            
            confidence_scores[metric] = confidence
            
            # Log confidence scores
            logger.info(f"Threshold confidence scores for {metric}:")
            for i, conf in enumerate(confidence):
                logger.info(f"  Threshold {threshold_list[i]:.4f}: {conf:.2f} confidence")
                
        return confidence_scores
    
    def export_thresholds(self, path: str) -> None:
        """
        Export thresholds to file.
        
        Args:
            path: Path to export thresholds to
        """
        # Prepare export data
        export_data = {
            "thresholds": self.thresholds,
            "quality_weights": self.quality_weights,
            "language_thresholds": self.language_thresholds,
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported thresholds to {path}") 