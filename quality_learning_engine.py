"""
Quality Learning Engine - Analyzes correlations between embedding similarity
and Groq quality ratings to improve translation quality assessment.
"""
import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from correlation_analyzer import CorrelationAnalyzer
from prediction_model_builder import PredictionModelBuilder
from threshold_optimizer import ThresholdOptimizer
from disagreement_analyzer import DisagreementAnalyzer
from config_manager import ConfigManager

# Configure logging using unified configuration
from logger_config import get_logger
logger = get_logger("QualityLearningEngine", "quality_learning")

class QualityLearningEngine:
    """
    Main engine for analyzing correlations between embedding similarity scores
    and Groq quality ratings to improve translation quality assessment.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        output_dir: str, 
        config: Optional[Union[Dict, ConfigManager]] = None
    ):
        """
        Initialize the quality learning engine.
        
        Args:
            data_dir: Directory containing batch processing results
            output_dir: Directory to write learning outputs
            config: Optional configuration dict or ConfigManager
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Set up directories
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizations_dir = self.output_dir / "visualizations"
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        self.thresholds_dir = self.output_dir / "thresholds"
        self.thresholds_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize config
        if isinstance(config, ConfigManager):
            self.config = config
        elif isinstance(config, dict):
            self.config = ConfigManager()
            for key, value in config.items():
                self.config.set(key, value)
        else:
            self.config = ConfigManager()
        
        # Initialize analyzers
        self.correlation_analyzer = CorrelationAnalyzer()
        self.model_builder = PredictionModelBuilder()
        self.threshold_optimizer = ThresholdOptimizer(self.config)
        self.disagreement_analyzer = DisagreementAnalyzer()
        
        # Initialize data
        self.raw_data = []
        self.metrics_df = None
        self.correlation_results = {}
        self.models = {}
        self.thresholds = {}
        
    def load_batch_data(
        self, 
        recursive: bool = True, 
        file_pattern: str = "*.processed"
    ) -> pd.DataFrame:
        """
        Load all processed batch files into a unified dataset.
        
        Args:
            recursive: Whether to search subdirectories recursively
            file_pattern: Glob pattern for matching result files
            
        Returns:
            Pandas DataFrame with combined data
        """
        pattern = f"**/{file_pattern}" if recursive else file_pattern
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern {pattern} in {self.data_dir}")
            return pd.DataFrame()
        
        logger.info(f"Loading {len(files)} batch result files")
        
        # Load all files
        raw_data = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    
                # Test if this is a translation analysis result
                if self._is_valid_translation_result(file_data):
                    raw_data.append(file_data)
            except Exception as e:
                logger.warning(f"Error loading file {file_path}: {str(e)}")
                
        logger.info(f"Successfully loaded {len(raw_data)} valid translation result files")
        self.raw_data = raw_data
        
        # Extract metrics into a DataFrame
        metrics_df = self.extract_metrics_pairs()
        return metrics_df
            
    def _is_valid_translation_result(self, data: Dict[str, Any]) -> bool:
        """
        Check if a data dict contains valid translation analysis results.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for expected keys in translation result
        required_keys = [
            "embedding_similarity", 
            "groq_quality_score"
        ]
        
        # Check optional but expected fields
        expected_keys = required_keys + [
            "source_text",
            "translated_text", 
            "source_language", 
            "target_language",
            "analysis_timestamp"
        ]
        
        # Must have all required keys
        has_required = all(key in data for key in required_keys)
        
        # Should have most expected keys
        expected_count = sum(1 for key in expected_keys if key in data)
        has_most_expected = expected_count >= len(expected_keys) * 0.7
        
        return has_required and has_most_expected
        
    def extract_metrics_pairs(self) -> pd.DataFrame:
        """
        Extract pairs of embedding similarity scores and quality ratings.
        
        Returns:
            DataFrame containing metric pairs
        """
        if not self.raw_data:
            logger.warning("No data loaded. Call load_batch_data() first.")
            return pd.DataFrame()
        
        metrics_data = []
        
        # Extract metrics from each result
        for result in self.raw_data:
            # Extract primary metrics
            embedding_similarity = result.get("embedding_similarity")
            groq_quality_score = result.get("groq_quality_score")
            
            # Skip if missing critical data
            if embedding_similarity is None or groq_quality_score is None:
                continue
                
            # Extract other metrics
            entry = {
                "embedding_similarity": float(embedding_similarity),
                "groq_quality_score": float(groq_quality_score),
                "source_language": result.get("source_language"),
                "target_language": result.get("target_language"),
            }
            
            # Add any other available metrics (may vary between results)
            for key, value in result.items():
                if key not in entry and isinstance(value, (int, float)):
                    entry[key] = value
            
            # Add detailed metrics if available
            groq_details = result.get("groq_quality_details", {})
            if isinstance(groq_details, dict):
                for detail_key, detail_value in groq_details.items():
                    if isinstance(detail_value, (int, float)):
                        entry[f"groq_detail_{detail_key}"] = detail_value
            
            metrics_data.append(entry)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        self.metrics_df = metrics_df
        
        logger.info(f"Extracted metrics from {len(metrics_df)} results")
        
        # Log basic statistics
        if not metrics_df.empty:
            logger.info(f"Embedding similarity range: {metrics_df['embedding_similarity'].min():.2f} - {metrics_df['embedding_similarity'].max():.2f}")
            logger.info(f"Groq quality score range: {metrics_df['groq_quality_score'].min():.2f} - {metrics_df['groq_quality_score'].max():.2f}")
            logger.info(f"Correlation: {metrics_df['embedding_similarity'].corr(metrics_df['groq_quality_score']):.4f}")
            
            # Log language pair counts
            if 'source_language' in metrics_df and 'target_language' in metrics_df:
                language_counts = metrics_df.groupby(['source_language', 'target_language']).size()
                logger.info(f"Language pair distribution:\n{language_counts}")
        
        return metrics_df
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between various metrics.
        
        Returns:
            Dictionary with correlation analysis results
        """
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No metrics data available. Call extract_metrics_pairs() first.")
            return {}
        
        logger.info("Analyzing correlations between metrics")
        self.correlation_analyzer.load_data(self.metrics_df)
        
        # Generate correlation matrix
        correlation_matrix = self.correlation_analyzer.generate_correlation_matrix()
        
        # Find best correlating features with Groq quality
        best_features = self.correlation_analyzer.find_best_correlating_features('groq_quality_score')
        
        # Identify similarity thresholds for quality levels
        similarity_thresholds = self.correlation_analyzer.identify_similarity_quality_thresholds()
        
        # Detect language-specific patterns
        language_patterns = self.correlation_analyzer.detect_language_specific_patterns()
        
        # Analyze outliers
        outliers = self.correlation_analyzer.analyze_outliers()
        
        # Compile results
        self.correlation_results = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "best_features": best_features,
            "similarity_thresholds": similarity_thresholds,
            "language_patterns": language_patterns,
            "outliers": outliers,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save correlation results
        self.save_correlation_data(self.correlation_results, "correlation_analysis")
        
        # Create visualizations
        self.visualize_correlations()
        
        return self.correlation_results
    
    def build_prediction_model(
        self, 
        model_type: str = "random_forest", 
        target: str = "groq_quality_score"
    ) -> Any:
        """
        Build and train a prediction model.
        
        Args:
            model_type: Type of model to build ("linear", "random_forest", "gradient_boosting", "neural_network")
            target: Target metric to predict
            
        Returns:
            Trained model
        """
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No metrics data available. Call extract_metrics_pairs() first.")
            return None
        
        logger.info(f"Building {model_type} prediction model for {target}")
        
        # Prepare dataset
        X_train, X_test, y_train, y_test = self.model_builder.prepare_dataset(
            self.metrics_df, 
            target_column=target
        )
        
        # Build appropriate model type
        model = None
        if model_type == "linear":
            model = self.model_builder.build_linear_model(X_train, y_train)
        elif model_type == "random_forest":
            model = self.model_builder.build_random_forest_model(X_train, y_train)
        elif model_type == "gradient_boosting":
            model = self.model_builder.build_gradient_boosting_model(X_train, y_train)
        elif model_type == "neural_network":
            model = self.model_builder.build_neural_network_model(X_train, y_train)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Evaluate model
        metrics = self.model_builder.evaluate_model(model, X_test, y_test)
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importance = self.model_builder.get_feature_importance(model, X_train.columns)
            logger.info(f"Feature importance: {importance}")
        
        # Store model
        model_entry = {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance if 'importance' in locals() else None,
            "model_type": model_type,
            "target": target,
            "timestamp": datetime.now().isoformat()
        }
        
        self.models[model_type] = model_entry
        
        # Save the model
        self.save_model(model, f"{model_type}_{target}_model")
        
        return model
    
    def save_model(self, model: Any, name: Optional[str] = None) -> str:
        """
        Save trained model to file.
        
        Args:
            model: Model to save
            name: Optional model name (defaults to timestamp)
            
        Returns:
            Path where model was saved
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"quality_model_{timestamp}"
        
        model_path = str(self.models_dir / f"{name}.pkl")
        self.model_builder.save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, name: str) -> Any:
        """
        Load a previously trained model.
        
        Args:
            name: Model name or path
            
        Returns:
            Loaded model
        """
        if not name.endswith(".pkl"):
            name = f"{name}.pkl"
        
        if not os.path.isabs(name):
            model_path = str(self.models_dir / name)
        else:
            model_path = name
        
        model = self.model_builder.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def evaluate_model(self, model: Any, test_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Model to evaluate
            test_data: Optional test data (uses 20% of metrics_df if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_data is None:
            if self.metrics_df is None or self.metrics_df.empty:
                logger.warning("No metrics data available and no test data provided.")
                return {}
                
            # Use 20% of data for testing
            _, X_test, _, y_test = self.model_builder.prepare_dataset(
                self.metrics_df, 
                target_column="groq_quality_score"
            )
        else:
            # Prepare provided test data
            X_test = test_data.drop(columns=["groq_quality_score"])
            y_test = test_data["groq_quality_score"]
        
        metrics = self.model_builder.evaluate_model(model, X_test, y_test)
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return metrics
    
    def optimize_thresholds(self) -> Dict[str, Any]:
        """
        Optimize quality thresholds based on learning.
        
        Returns:
            Dictionary with optimized thresholds
        """
        if self.correlation_results == {}:
            logger.warning("No correlation analysis results available. Call analyze_correlations() first.")
            return {}
        
        logger.info("Optimizing quality thresholds based on correlation analysis")
        
        # Load correlation data into threshold optimizer
        self.threshold_optimizer.load_correlation_data(self.correlation_results)
        
        # Find optimal thresholds for embedding similarity
        similarity_thresholds = self.threshold_optimizer.find_optimal_thresholds(
            "embedding_similarity", 
            quality_levels=5
        )
        
        # Optimize quality weights
        quality_weights = self.threshold_optimizer.optimize_quality_weights([
            "embedding_similarity",
            "groq_quality_score"
        ])
        
        # Get language-specific thresholds if data allows
        language_thresholds = self.threshold_optimizer.get_language_specific_thresholds()
        
        # Compile threshold results
        self.thresholds = {
            "similarity_thresholds": similarity_thresholds,
            "quality_weights": quality_weights,
            "language_thresholds": language_thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
        # Export thresholds
        threshold_path = self.thresholds_dir / "optimized_thresholds.json"
        self.threshold_optimizer.export_thresholds(str(threshold_path))
        
        logger.info(f"Optimized thresholds saved to {threshold_path}")
        
        return self.thresholds
    
    def apply_learned_thresholds(self, config_path: Optional[str] = None) -> bool:
        """
        Update configuration with learned thresholds.
        
        Args:
            config_path: Optional path to config file (uses config_manager default if None)
            
        Returns:
            True if config update was successful, False otherwise
        """
        if not self.thresholds:
            logger.warning("No threshold data available. Call optimize_thresholds() first.")
            return False
        
        logger.info("Applying learned thresholds to configuration")
        
        success = self.threshold_optimizer.apply_thresholds_to_config(config_path)
        
        if success:
            logger.info("Successfully applied thresholds to configuration")
        else:
            logger.warning("Failed to apply thresholds to configuration")
            
        return success
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights report.
        
        Returns:
            Dictionary with insights report data
        """
        if not self.correlation_results:
            logger.warning("No correlation analysis results available. Call analyze_correlations() first.")
            return {}
        
        if not self.thresholds:
            logger.warning("No threshold data available. Call optimize_thresholds() first.")
        
        logger.info("Generating insights report")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Compile report data
        report = {
            "title": "Translation Quality Assessment Insights Report",
            "timestamp": timestamp,
            "dataset_size": len(self.metrics_df) if self.metrics_df is not None else 0,
            "key_correlations": {},
            "quality_thresholds": self.thresholds.get("similarity_thresholds", {}),
            "optimized_weights": self.thresholds.get("quality_weights", {}),
            "language_specific_insights": {},
            "recommendations": []
        }
        
        # Add correlation highlights
        if self.correlation_results:
            matrix = pd.DataFrame(self.correlation_results.get("correlation_matrix", {}))
            if not matrix.empty and "groq_quality_score" in matrix:
                for col in matrix.index:
                    if col != "groq_quality_score" and not pd.isna(matrix.loc[col, "groq_quality_score"]):
                        report["key_correlations"][col] = matrix.loc[col, "groq_quality_score"]
        
        # Add language-specific insights
        language_patterns = self.correlation_results.get("language_patterns", {})
        for lang_pair, pattern in language_patterns.items():
            report["language_specific_insights"][lang_pair] = pattern
        
        # Add recommendations based on findings
        if "embedding_similarity" in report["key_correlations"]:
            corr = report["key_correlations"]["embedding_similarity"]
            if abs(corr) > 0.8:
                report["recommendations"].append(
                    "Embedding similarity is strongly correlated with quality ratings and can be weighted higher"
                )
            elif abs(corr) > 0.5:
                report["recommendations"].append(
                    "Embedding similarity shows good correlation with quality ratings"
                )
            else:
                report["recommendations"].append(
                    "Embedding similarity shows weak correlation with quality ratings and should be supplemented"
                )
        
        # Add model insights if available
        if self.models:
            model_insights = []
            for model_type, model_entry in self.models.items():
                metrics = model_entry.get("metrics", {})
                importance = model_entry.get("feature_importance", {})
                
                model_info = {
                    "model_type": model_type,
                    "prediction_accuracy": metrics.get("r2_score", 0),
                    "important_features": importance
                }
                model_insights.append(model_info)
                
            report["model_insights"] = model_insights
        
        # Save report to file
        report_path = self.output_dir / f"quality_insights_report_{timestamp.replace(' ', '_').replace(':', '-')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Insights report saved to {report_path}")
        
        return report
    
    def visualize_correlations(self) -> None:
        """
        Create visualizations of correlations between metrics.
        """
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No metrics data available for visualization.")
            return
        
        logger.info("Creating correlation visualizations")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories if they don't exist
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.metrics_df.select_dtypes(include=['number']).corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   square=True, linewidths=.5)
        plt.title("Correlation Matrix of Translation Quality Metrics")
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"correlation_heatmap_{timestamp}.png", dpi=300)
        plt.close()
        
        # 2. Embedding vs. Groq Quality Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.regplot(x="embedding_similarity", y="groq_quality_score", data=self.metrics_df)
        plt.title("Embedding Similarity vs. Groq Quality Score")
        plt.xlabel("Embedding Similarity")
        plt.ylabel("Groq Quality Score")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.visualizations_dir / f"similarity_vs_quality_{timestamp}.png", dpi=300)
        plt.close()
        
        # 3. Distribution of Metrics
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.metrics_df["embedding_similarity"], kde=True)
        plt.title("Distribution of Embedding Similarity")
        plt.xlabel("Embedding Similarity")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.metrics_df["groq_quality_score"], kde=True)
        plt.title("Distribution of Groq Quality Scores")
        plt.xlabel("Groq Quality Score")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"metric_distributions_{timestamp}.png", dpi=300)
        plt.close()
        
        # 4. Language Pair Influence
        if 'source_language' in self.metrics_df and 'target_language' in self.metrics_df:
            # Combine language pairs
            self.metrics_df['language_pair'] = self.metrics_df['source_language'] + '-' + self.metrics_df['target_language']
            
            # Get top N language pairs by count
            top_pairs = self.metrics_df['language_pair'].value_counts().head(10).index
            pair_data = self.metrics_df[self.metrics_df['language_pair'].isin(top_pairs)]
            
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='language_pair', y='embedding_similarity', data=pair_data)
            plt.title("Embedding Similarity by Language Pair")
            plt.xlabel("Language Pair")
            plt.ylabel("Embedding Similarity")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / f"similarity_by_language_{timestamp}.png", dpi=300)
            plt.close()
            
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='language_pair', y='groq_quality_score', data=pair_data)
            plt.title("Groq Quality Score by Language Pair")
            plt.xlabel("Language Pair")
            plt.ylabel("Groq Quality Score")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / f"quality_by_language_{timestamp}.png", dpi=300)
            plt.close()
            
        logger.info(f"Visualizations saved to {self.visualizations_dir}")
    
    def save_correlation_data(self, data: Dict[str, Any], name: str) -> str:
        """
        Save correlation data for future use.
        
        Args:
            data: Correlation data to save
            name: Base name for the file
            
        Returns:
            Path where data was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{name}_{timestamp}.json"
        file_path = self.output_dir / file_name
        
        # Ensure data is JSON serializable
        json_data = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                json_data[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
        logger.info(f"Correlation data saved to {file_path}")
        
        return str(file_path)
    
    def run_full_learning_cycle(self) -> Dict[str, Any]:
        """
        Run the complete learning process.
        
        Returns:
            Dictionary with results from the learning process
        """
        logger.info("Starting full learning cycle")
        
        # Step 1: Load batch data
        self.load_batch_data()
        
        # Step 2: Analyze correlations
        correlation_results = self.analyze_correlations()
        
        # Step 3: Build prediction models
        rf_model = self.build_prediction_model(model_type="random_forest")
        gb_model = self.build_prediction_model(model_type="gradient_boosting")
        
        # Step 4: Optimize thresholds
        thresholds = self.optimize_thresholds()
        
        # Step 5: Apply thresholds to config
        self.apply_learned_thresholds()
        
        # Step 6: Analyze disagreements (NEW!)
        disagreement_results = self.analyze_disagreements(threshold=0.4)  # Moderate threshold
        
        # Step 7: Generate insights report
        report = self.generate_insights_report()
        
        # Return comprehensive results
        results = {
            "correlation_results": correlation_results,
            "thresholds": thresholds,
            "disagreement_results": disagreement_results,
            "insights_report": report,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Full learning cycle completed")
        
        return results
    
    def get_learned_scoring_weights(self) -> Dict[str, float]:
        """
        Returns optimal scoring weights based on learning.
        
        Returns:
            Dictionary with learned weights for scoring metrics
        """
        if not self.thresholds or "quality_weights" not in self.thresholds:
            logger.warning("No learned weights available. Call optimize_thresholds() first.")
            return {}
            
        return self.thresholds["quality_weights"]
        
    def analyze_disagreements(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze cases where embedding similarity scores disagree with AI quality ratings.
        
        Args:
            threshold: Threshold for considering metrics in disagreement
            
        Returns:
            Dictionary with disagreement analysis results
        """
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No metrics data available. Call extract_metrics_pairs() first.")
            return {}
        
        logger.info(f"Analyzing disagreements with threshold {threshold}")
        
        # Initialize disagreement analyzer
        disagreement_analyzer = DisagreementAnalyzer(threshold=threshold)
        
        # Load data and identify disagreements
        disagreement_analyzer.load_data(self.metrics_df)
        disagreements = disagreement_analyzer.identify_disagreements()
        
        # Skip further analysis if no disagreements found
        if disagreements.empty:
            logger.info("No disagreements found above the threshold")
            return {"count": 0, "message": "No disagreements found"}
        
        # Perform full analysis
        disagreement_analyzer.classify_disagreements()
        disagreement_analyzer.score_disagreements()
        disagreement_analyzer.analyze_language_patterns()
        disagreement_analyzer.analyze_text_features()
        
        # Generate comprehensive report
        report = disagreement_analyzer.generate_disagreement_report()
        
        # Save disagreement cases
        disagreements_dir = self.output_dir / "disagreements"
        disagreements_dir.mkdir(parents=True, exist_ok=True)
        disagreement_analyzer.save_disagreement_cases(str(disagreements_dir))
        
        # Create visualizations
        self._create_disagreement_visualizations(disagreement_analyzer)
        
        # Store recommendations for further use
        if "recommendations" in report:
            self.disagreement_recommendations = report["recommendations"]
            
        logger.info(f"Completed disagreement analysis for {len(disagreements)} cases")
        
        return report

    def get_disagreement_recommendations(self) -> List[str]:
        """
        Get recommendations based on disagreement analysis.
        
        Returns:
            List of recommendations
        """
        if not hasattr(self, 'disagreement_recommendations') or not self.disagreement_recommendations:
            logger.warning("No disagreement recommendations available. Call analyze_disagreements() first.")
            return []
            
        return self.disagreement_recommendations

    def _create_disagreement_visualizations(self, disagreement_analyzer: DisagreementAnalyzer) -> None:
        """
        Create visualizations for disagreement analysis.
        
        Args:
            disagreement_analyzer: DisagreementAnalyzer instance with analyzed data
        """
        # Get visualization data
        viz_data = disagreement_analyzer.visualize_disagreements()
        if not viz_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # 1. Scatter plot of embedding_similarity vs groq_quality_score
        if "scatter_data" in viz_data:
            data = viz_data["scatter_data"]
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot with color based on severity
            plt.scatter(
                data["embedding_similarity"], 
                data["groq_quality_score"],
                c=data["severity"], 
                cmap="coolwarm",
                alpha=0.7,
                s=50
            )
            
            # Add the diagonal line (perfect agreement)
            min_val = min(plt.xlim()[0], plt.ylim()[0])
            max_val = max(plt.xlim()[1], plt.ylim()[1])
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect Agreement")
            
            plt.colorbar(label="Disagreement Severity")
            plt.xlabel("Embedding Similarity Score")
            plt.ylabel("Groq Quality Score")
            plt.title("Embedding Similarity vs. Groq Quality Score: Disagreement Cases")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.visualizations_dir / f"disagreement_scatter_{timestamp}.png", dpi=300)
            plt.close()
        
        # 2. Disagreement type distribution
        if "type_distribution" in viz_data:
            plt.figure(figsize=(10, 6))
            types = list(viz_data["type_distribution"].keys())
            counts = list(viz_data["type_distribution"].values())
            
            # Create bar chart
            plt.bar(types, counts, color="skyblue")
            plt.xlabel("Disagreement Type")
            plt.ylabel("Count")
            plt.title("Distribution of Disagreement Types")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.visualizations_dir / f"disagreement_types_{timestamp}.png", dpi=300)
            plt.close()
        
        # 3. Severity distribution
        if "severity_distribution" in viz_data:
            plt.figure(figsize=(8, 6))
            categories = list(viz_data["severity_distribution"].keys())
            counts = list(viz_data["severity_distribution"].values())
            
            # Create bar chart
            plt.bar(categories, counts, color="salmon")
            plt.xlabel("Severity Level")
            plt.ylabel("Count")
            plt.title("Distribution of Disagreement Severity")
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.visualizations_dir / f"disagreement_severity_{timestamp}.png", dpi=300)
            plt.close()
        
        # 4. Language pair distribution
        if "language_distribution" in viz_data:
            plt.figure(figsize=(12, 6))
            languages = list(viz_data["language_distribution"].keys())
            counts = list(viz_data["language_distribution"].values())
            
            # Sort by count
            sorted_data = sorted(zip(languages, counts), key=lambda x: x[1], reverse=True)
            languages = [item[0] for item in sorted_data]
            counts = [item[1] for item in sorted_data]
            
            # Create bar chart
            plt.bar(languages, counts, color="mediumseagreen")
            plt.xlabel("Language Pair")
            plt.ylabel("Count")
            plt.title("Disagreements by Language Pair")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.visualizations_dir / f"disagreement_languages_{timestamp}.png", dpi=300)
            plt.close()
        
        logger.info(f"Created disagreement visualizations in {self.visualizations_dir}") 