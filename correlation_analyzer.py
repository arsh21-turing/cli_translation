"""
Analyze statistical correlations in translation quality data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Configure logging using unified configuration
from logger_config import get_logger
logger = get_logger("CorrelationAnalyzer", "quality_learning")

class CorrelationAnalyzer:
    """
    Analyzes statistical correlations in translation quality data.
    """
    
    def __init__(self):
        """
        Initialize the correlation analyzer.
        """
        self.data = None
        self.numeric_data = None
        self.correlation_matrix = None
        self.clusters = None
        
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load data for analysis.
        
        Args:
            data: Pandas DataFrame with quality metrics
        """
        self.data = data.copy()
        
        # Extract only numeric columns for correlation analysis
        self.numeric_data = self.data.select_dtypes(include=['number'])
        
        logger.info(f"Loaded data with {len(self.data)} rows and {len(self.numeric_data.columns)} numeric features")
        
    def compute_pearson_correlation(self, x_metric: str, y_metric: str) -> Tuple[float, float]:
        """
        Compute Pearson correlation between two metrics.
        
        Args:
            x_metric: Name of first metric
            y_metric: Name of second metric
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return (0.0, 1.0)
            
        if x_metric not in self.numeric_data.columns or y_metric not in self.numeric_data.columns:
            logger.error(f"Metrics {x_metric} and/or {y_metric} not found in data.")
            return (0.0, 1.0)
            
        # Calculate Pearson correlation
        corr, p_value = stats.pearsonr(
            self.numeric_data[x_metric].dropna(),
            self.numeric_data[y_metric].dropna()
        )
        
        logger.info(f"Pearson correlation between {x_metric} and {y_metric}: {corr:.4f} (p={p_value:.4f})")
        
        return (corr, p_value)
    
    def compute_spearman_correlation(self, x_metric: str, y_metric: str) -> Tuple[float, float]:
        """
        Compute Spearman rank correlation between two metrics.
        
        Args:
            x_metric: Name of first metric
            y_metric: Name of second metric
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return (0.0, 1.0)
            
        if x_metric not in self.numeric_data.columns or y_metric not in self.numeric_data.columns:
            logger.error(f"Metrics {x_metric} and/or {y_metric} not found in data.")
            return (0.0, 1.0)
            
        # Calculate Spearman correlation
        corr, p_value = stats.spearmanr(
            self.numeric_data[x_metric].dropna(),
            self.numeric_data[y_metric].dropna()
        )
        
        logger.info(f"Spearman correlation between {x_metric} and {y_metric}: {corr:.4f} (p={p_value:.4f})")
        
        return (corr, p_value)
    
    def find_best_correlating_features(self, target_metric: str) -> Dict[str, Tuple[float, float]]:
        """
        Find features that correlate best with target metric.
        
        Args:
            target_metric: Target metric to find correlations for
            
        Returns:
            Dictionary of feature names to (correlation, p-value) tuples, sorted by absolute correlation
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        if target_metric not in self.numeric_data.columns:
            logger.error(f"Target metric {target_metric} not found in data.")
            return {}
        
        # Compute correlations with all other features
        correlations = {}
        for feature in self.numeric_data.columns:
            if feature != target_metric:
                corr, p_value = self.compute_pearson_correlation(feature, target_metric)
                correlations[feature] = (corr, p_value)
                
        # Sort by absolute correlation value (descending)
        sorted_correlations = dict(sorted(
            correlations.items(), 
            key=lambda x: abs(x[1][0]), 
            reverse=True
        ))
        
        # Log top correlations
        logger.info(f"Top correlations with {target_metric}:")
        for feature, (corr, p_value) in list(sorted_correlations.items())[:5]:
            logger.info(f"  {feature}: {corr:.4f} (p={p_value:.4f})")
        
        return sorted_correlations
    
    def analyze_feature_importance(self, target_metric: str) -> Dict[str, float]:
        """
        Analyze importance of features for predicting target metric.
        
        Args:
            target_metric: Target metric to analyze feature importance for
            
        Returns:
            Dictionary of feature names to importance scores
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        if target_metric not in self.numeric_data.columns:
            logger.error(f"Target metric {target_metric} not found in data.")
            return {}
        
        # Get feature correlations
        correlations = self.find_best_correlating_features(target_metric)
        
        # Convert correlations to importance scores (absolute value, normalized)
        total_corr = sum(abs(corr) for corr, _ in correlations.values())
        if total_corr == 0:
            return {feature: 0.0 for feature in correlations.keys()}
            
        importance = {
            feature: abs(corr) / total_corr 
            for feature, (corr, _) in correlations.items()
        }
        
        # Log feature importance
        logger.info(f"Feature importance for {target_metric}:")
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {feature}: {imp:.4f}")
        
        return importance
    
    def identify_similarity_quality_thresholds(self, n_thresholds: int = 5) -> Dict[str, List[float]]:
        """
        Identify threshold ranges for quality levels based on similarity scores.
        
        Args:
            n_thresholds: Number of threshold levels to identify
            
        Returns:
            Dictionary with threshold ranges for different metrics
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        # Check for required columns
        if "embedding_similarity" not in self.numeric_data.columns:
            logger.error("embedding_similarity column not found in data.")
            return {}
            
        if "groq_quality_score" not in self.numeric_data.columns:
            logger.error("groq_quality_score column not found in data.")
            return {}
        
        # Get similarity and quality data
        similarity = self.numeric_data["embedding_similarity"].dropna()
        quality = self.numeric_data["groq_quality_score"].dropna()
        
        # Ensure the data is aligned
        df = pd.DataFrame({
            "similarity": similarity,
            "quality": quality
        }).dropna()
        
        # Sort by quality score
        df = df.sort_values("quality")
        
        # Create equally sized quality bins
        bin_edges = []
        bin_size = len(df) // n_thresholds
        for i in range(n_thresholds):
            if i < n_thresholds - 1:
                idx = (i + 1) * bin_size
                bin_edges.append(df.iloc[idx]["quality"])
            else:
                bin_edges.append(df["quality"].max() + 0.001)  # Add small value to include max
        
        # Get corresponding similarity thresholds
        # Group quality scores into bins
        df["quality_bin"] = pd.cut(
            df["quality"], 
            bins=[df["quality"].min() - 0.001] + bin_edges, 
            labels=False
        )
        
        # Get similarity thresholds for each quality bin
        similarity_thresholds = []
        for i in range(n_thresholds):
            bin_df = df[df["quality_bin"] == i]
            if not bin_df.empty:
                similarity_thresholds.append(bin_df["similarity"].min())
        
        # Add max similarity as final threshold
        similarity_thresholds.append(df["similarity"].max())
                
        # Create threshold levels
        quality_levels = []
        for i in range(len(bin_edges)):
            if i == 0:
                quality_levels.append(df["quality"].min())
            else:
                quality_levels.append(bin_edges[i-1])
        
        quality_levels.append(df["quality"].max())
        
        # Log thresholds
        logger.info("Identified quality thresholds:")
        for i in range(n_thresholds):
            if i < len(similarity_thresholds) - 1:
                logger.info(f"  Level {i+1}: Similarity >= {similarity_thresholds[i]:.4f}, Quality >= {quality_levels[i]:.4f}")
        
        # Create threshold dictionary
        thresholds = {
            "embedding_similarity": similarity_thresholds,
            "groq_quality_score": quality_levels,
            "n_levels": n_thresholds
        }
        
        return thresholds
    
    def generate_correlation_matrix(self) -> pd.DataFrame:
        """
        Generate correlation matrix for all metrics.
        
        Returns:
            Pandas DataFrame with correlation matrix
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return pd.DataFrame()
            
        # Compute correlation matrix
        self.correlation_matrix = self.numeric_data.corr()
        
        # Log matrix dimensions
        logger.info(f"Generated correlation matrix with shape {self.correlation_matrix.shape}")
        
        return self.correlation_matrix
    
    def plot_correlation_heatmap(self) -> None:
        """
        Create correlation heatmap visualization.
        
        Note: This method only returns the data needed for plotting.
        The actual plotting is done by the QualityLearningEngine.
        """
        if self.correlation_matrix is None:
            self.generate_correlation_matrix()
            
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            logger.error("Could not generate correlation matrix.")
            return None
        
        return self.correlation_matrix
    
    def plot_scatter_with_regression(self, x_metric: str, y_metric: str) -> Dict[str, Any]:
        """
        Create scatter plot with regression line data.
        
        Args:
            x_metric: Name of metric for x-axis
            y_metric: Name of metric for y-axis
            
        Returns:
            Dictionary with data for plotting
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        if x_metric not in self.numeric_data.columns or y_metric not in self.numeric_data.columns:
            logger.error(f"Metrics {x_metric} and/or {y_metric} not found in data.")
            return {}
            
        # Get data for the two metrics
        x = self.numeric_data[x_metric].dropna()
        y = self.numeric_data[y_metric].dropna()
        
        # Ensure the data is aligned
        df = pd.DataFrame({x_metric: x, y_metric: y}).dropna()
        x = df[x_metric]
        y = df[y_metric]
        
        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Create regression line points
        x_min, x_max = x.min(), x.max()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        
        # Compile plot data
        plot_data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "x_line": x_line.tolist(),
            "y_line": y_line.tolist(),
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "x_label": x_metric,
            "y_label": y_metric
        }
        
        logger.info(f"Generated regression data for {x_metric} vs {y_metric}: R^2={r_value**2:.4f}")
        
        return plot_data
    
    def find_quality_clusters(self, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Identify clusters of similar quality patterns.
        
        Args:
            n_clusters: Number of clusters to find
            
        Returns:
            Dictionary with cluster analysis results
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        # Check for required columns
        if "embedding_similarity" not in self.numeric_data.columns:
            logger.error("embedding_similarity column not found in data.")
            return {}
            
        if "groq_quality_score" not in self.numeric_data.columns:
            logger.error("groq_quality_score column not found in data.")
            return {}
        
        # Get similarity and quality data
        similarity = self.numeric_data["embedding_similarity"].values.reshape(-1, 1)
        quality = self.numeric_data["groq_quality_score"].values.reshape(-1, 1)
        
        # Ensure no NaNs
        valid_mask = ~(np.isnan(similarity).any(axis=1) | np.isnan(quality).any(axis=1))
        similarity = similarity[valid_mask]
        quality = quality[valid_mask]
        
        # Combine features for clustering
        X = np.hstack([similarity, quality])
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(X)
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = self.clusters == i
            cluster_X = X[cluster_mask]
            
            if len(cluster_X) == 0:
                continue
                
            cluster_stats.append({
                "cluster_id": i,
                "size": int(np.sum(cluster_mask)),
                "similarity_mean": float(np.mean(cluster_X[:, 0])),
                "similarity_std": float(np.std(cluster_X[:, 0])),
                "quality_mean": float(np.mean(cluster_X[:, 1])),
                "quality_std": float(np.std(cluster_X[:, 1])),
                "similarity_min": float(np.min(cluster_X[:, 0])),
                "similarity_max": float(np.max(cluster_X[:, 0])),
                "quality_min": float(np.min(cluster_X[:, 1])),
                "quality_max": float(np.max(cluster_X[:, 1]))
            })
        
        # Sort clusters by quality mean
        cluster_stats.sort(key=lambda x: x["quality_mean"])
        
        # Log cluster information
        logger.info(f"Identified {len(cluster_stats)} quality clusters:")
        for i, stats in enumerate(cluster_stats):
            logger.info(f"  Cluster {i+1}: {stats['size']} items, Quality: {stats['quality_mean']:.2f}±{stats['quality_std']:.2f}, Similarity: {stats['similarity_mean']:.2f}±{stats['similarity_std']:.2f}")
        
        # Compile cluster results
        cluster_results = {
            "n_clusters": len(cluster_stats),
            "cluster_stats": cluster_stats,
            "centroids": kmeans.cluster_centers_.tolist() if hasattr(kmeans, 'cluster_centers_') else None,
            "inertia": float(kmeans.inertia_) if hasattr(kmeans, 'inertia_') else None
        }
        
        return cluster_results
    
    def detect_language_specific_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect patterns specific to language pairs.
        
        Returns:
            Dictionary mapping language pairs to their pattern data
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        # Check if language columns exist
        if 'source_language' not in self.data.columns or 'target_language' not in self.data.columns:
            logger.warning("Language columns not found in data.")
            return {}
            
        # Create language pair column
        self.data['language_pair'] = self.data['source_language'] + '-' + self.data['target_language']
        
        # Get language pairs with sufficient data
        language_pairs = self.data['language_pair'].value_counts()
        valid_pairs = language_pairs[language_pairs >= 10].index.tolist()
        
        if not valid_pairs:
            logger.warning("No language pairs with sufficient data found.")
            return {}
        
        # Analyze patterns for each language pair
        language_patterns = {}
        for pair in valid_pairs:
            pair_data = self.data[self.data['language_pair'] == pair]
            
            # Compute basic statistics
            stats_data = {
                "count": int(len(pair_data)),
                "embedding_similarity": {
                    "mean": float(pair_data['embedding_similarity'].mean()),
                    "std": float(pair_data['embedding_similarity'].std()),
                    "min": float(pair_data['embedding_similarity'].min()),
                    "max": float(pair_data['embedding_similarity'].max()),
                },
                "groq_quality_score": {
                    "mean": float(pair_data['groq_quality_score'].mean()),
                    "std": float(pair_data['groq_quality_score'].std()),
                    "min": float(pair_data['groq_quality_score'].min()),
                    "max": float(pair_data['groq_quality_score'].max()),
                }
            }
            
            # Compute correlation
            corr, p_value = stats.pearsonr(
                pair_data['embedding_similarity'],
                pair_data['groq_quality_score']
            )
            
            stats_data["correlation"] = {
                "pearson": float(corr),
                "p_value": float(p_value)
            }
            
            # Add to language patterns
            language_patterns[pair] = stats_data
        
        # Log language patterns
        logger.info(f"Detected patterns for {len(language_patterns)} language pairs:")
        for pair, pattern in language_patterns.items():
            logger.info(f"  {pair}: n={pattern['count']}, similarity={pattern['embedding_similarity']['mean']:.2f}±{pattern['embedding_similarity']['std']:.2f}, quality={pattern['groq_quality_score']['mean']:.2f}±{pattern['groq_quality_score']['std']:.2f}, corr={pattern['correlation']['pearson']:.2f}")
        
        return language_patterns
    
    def analyze_outliers(self, contamination: float = 0.05) -> Dict[str, Any]:
        """
        Analyze outlier cases in the data.
        
        Args:
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with outlier analysis results
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        # Check for required columns
        if "embedding_similarity" not in self.numeric_data.columns:
            logger.error("embedding_similarity column not found in data.")
            return {}
            
        if "groq_quality_score" not in self.numeric_data.columns:
            logger.error("groq_quality_score column not found in data.")
            return {}
        
        # Get similarity and quality data
        features = self.numeric_data[["embedding_similarity", "groq_quality_score"]].dropna()
        
        if len(features) < 10:
            logger.warning("Not enough data for outlier analysis.")
            return {}
        
        # Detect outliers using Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = clf.fit_predict(features)
        
        # -1 means outlier, 1 means inlier
        outlier_mask = outlier_labels == -1
        outlier_indices = np.where(outlier_mask)[0]
        
        # Get outlier data points
        outliers = features.iloc[outlier_indices]
        inliers = features.iloc[~outlier_mask]
        
        # Calculate statistics
        outlier_stats = {
            "count": int(len(outliers)),
            "percent": float(len(outliers) / len(features) * 100),
            "similarity_mean": float(outliers["embedding_similarity"].mean()),
            "similarity_std": float(outliers["embedding_similarity"].std()),
            "quality_mean": float(outliers["groq_quality_score"].mean()),
            "quality_std": float(outliers["groq_quality_score"].std()),
        }
        
        inlier_stats = {
            "count": int(len(inliers)),
            "percent": float(len(inliers) / len(features) * 100),
            "similarity_mean": float(inliers["embedding_similarity"].mean()),
            "similarity_std": float(inliers["embedding_similarity"].std()),
            "quality_mean": float(inliers["groq_quality_score"].mean()),
            "quality_std": float(inliers["groq_quality_score"].std()),
        }
        
        # Compile outlier results
        outlier_results = {
            "outlier_count": outlier_stats["count"],
            "outlier_percent": outlier_stats["percent"],
            "outlier_stats": outlier_stats,
            "inlier_stats": inlier_stats,
        }
        
        # Check if outliers have higher/lower metrics
        sim_diff = outlier_stats["similarity_mean"] - inlier_stats["similarity_mean"]
        qual_diff = outlier_stats["quality_mean"] - inlier_stats["quality_mean"]
        
        outlier_results["similarity_difference"] = float(sim_diff)
        outlier_results["quality_difference"] = float(qual_diff)
        
        # Log outlier information
        logger.info(f"Detected {outlier_stats['count']} outliers ({outlier_stats['percent']:.1f}%):")
        logger.info(f"  Outlier similarity: {outlier_stats['similarity_mean']:.4f}±{outlier_stats['similarity_std']:.4f}, quality: {outlier_stats['quality_mean']:.4f}±{outlier_stats['quality_std']:.4f}")
        logger.info(f"  Inlier similarity: {inlier_stats['similarity_mean']:.4f}±{inlier_stats['similarity_std']:.4f}, quality: {inlier_stats['quality_mean']:.4f}±{inlier_stats['quality_std']:.4f}")
        
        return outlier_results
    
    def compute_confidence_intervals(self, metric: str, confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute confidence intervals for a metric.
        
        Args:
            metric: Name of the metric
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Dictionary with confidence interval data
        """
        if self.numeric_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
            
        if metric not in self.numeric_data.columns:
            logger.error(f"Metric {metric} not found in data.")
            return {}
        
        # Get metric data
        values = self.numeric_data[metric].dropna()
        
        # Compute statistics
        mean = float(values.mean())
        std = float(values.std())
        n = len(values)
        
        # Compute confidence interval
        alpha = 1.0 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        margin_of_error = t_critical * std / np.sqrt(n)
        
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
        # Compile interval data
        interval_data = {
            "metric": metric,
            "mean": mean,
            "std": std,
            "n": n,
            "confidence": confidence,
            "t_critical": float(t_critical),
            "margin_of_error": float(margin_of_error),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
        
        logger.info(f"Computed {confidence*100}% confidence interval for {metric}: {mean:.4f} ± {margin_of_error:.4f} [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        return interval_data 