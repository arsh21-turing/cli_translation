#!/usr/bin/env python3
"""
Test script for the Quality Learning System.
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

def create_test_data(output_dir: str, num_files: int = 10) -> None:
    """
    Create test batch processing files with embedding similarity and Groq quality data.
    
    Args:
        output_dir: Directory to write test files
        num_files: Number of test files to create
    """
    import random
    import numpy as np
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_files} test files in {output_dir}")
    
    # Language pairs
    language_pairs = [
        ("en", "es"), ("en", "fr"), ("en", "de"), ("es", "en"),
        ("fr", "en"), ("de", "en"), ("en", "pt"), ("pt", "en")
    ]
    
    for i in range(num_files):
        # Generate correlated data (embedding similarity influences quality)
        embedding_similarity = random.uniform(0.2, 0.95)
        
        # Create correlation: higher similarity tends to higher quality (with noise)
        noise = random.gauss(0, 0.5)
        base_quality = 2 + (embedding_similarity * 6) + noise  # Scale to 2-8 range
        groq_quality_score = max(1.0, min(10.0, base_quality))  # Clamp to 1-10
        
        # Random language pair
        source_lang, target_lang = random.choice(language_pairs)
        
        # Create test translation data
        test_data = {
            "source_text": f"Test source text {i}",
            "translated_text": f"Test translated text {i}",
            "source_language": source_lang,
            "target_language": target_lang,
            "embedding_similarity": round(embedding_similarity, 4),
            "groq_quality_score": round(groq_quality_score, 2),
            "groq_quality_details": {
                "accuracy": round(groq_quality_score * 0.9 + random.uniform(-0.5, 0.5), 2),
                "fluency": round(groq_quality_score * 0.95 + random.uniform(-0.3, 0.3), 2),
                "consistency": round(groq_quality_score * 0.85 + random.uniform(-0.7, 0.7), 2)
            },
            "analysis_timestamp": f"2024-12-01T{i:02d}:00:00",
            "additional_metric_1": round(random.uniform(0.1, 0.9), 3),
            "additional_metric_2": round(random.uniform(0.2, 0.8), 3)
        }
        
        # Save to file
        file_path = os.path.join(output_dir, f"test_result_{i:03d}.processed")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
    
    print(f"Created {num_files} test files successfully")

def test_quality_learning_engine():
    """Test the complete quality learning system."""
    try:
        from quality_learning_engine import QualityLearningEngine
        print("✓ Successfully imported QualityLearningEngine")
    except ImportError as e:
        print(f"✗ Failed to import QualityLearningEngine: {e}")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = os.path.join(temp_dir, "test_data")
        output_dir = os.path.join(temp_dir, "output")
        
        # Create test data
        create_test_data(test_data_dir, num_files=25)
        
        # Initialize quality learning engine
        engine = QualityLearningEngine(
            data_dir=test_data_dir,
            output_dir=output_dir
        )
        
        try:
            # Test data loading
            print("Testing data loading...")
            metrics_df = engine.load_batch_data()
            assert metrics_df is not None and not metrics_df.empty, "Failed to load test data"
            
            # Test correlation analysis
            print("Testing correlation analysis...")
            correlation_results = engine.analyze_correlations()
            assert correlation_results, "Failed correlation analysis"
            
            # Test model building
            print("Testing model building...")
            model = engine.build_prediction_model(model_type="random_forest")
            assert model is not None, "Failed to build prediction model"
            
            # Test threshold optimization
            print("Testing threshold optimization...")
            thresholds = engine.optimize_thresholds()
            assert thresholds, "Failed threshold optimization"
            
            # Test report generation
            print("Testing report generation...")
            report = engine.generate_insights_report()
            assert report, "Failed to generate insights report"
            
            # Test full learning cycle
            print("Testing full learning cycle...")
            results = engine.run_full_learning_cycle()
            assert results, "Failed full learning cycle"
            
            # Check if files were created
            models_dir = Path(output_dir) / "models"
            visualizations_dir = Path(output_dir) / "visualizations"
            thresholds_dir = Path(output_dir) / "thresholds"
            
            if models_dir.exists() and any(models_dir.glob("*.pkl")):
                print("✓ Models were saved successfully")
            else:
                print("⚠ No model files found")
            
            if visualizations_dir.exists() and any(visualizations_dir.glob("*.png")):
                print("✓ Visualizations were created successfully")
            else:
                print("⚠ No visualization files found")
            
            if thresholds_dir.exists() and any(thresholds_dir.glob("*.json")):
                print("✓ Thresholds were exported successfully")
            else:
                print("⚠ No threshold files found")
                
        except Exception as e:
            print(f"✗ Error during quality learning test: {str(e)}")

def test_individual_components():
    """Test individual components of the quality learning system."""
    
    # Test CorrelationAnalyzer
    try:
        from correlation_analyzer import CorrelationAnalyzer
        print("✓ Successfully imported CorrelationAnalyzer")
        
        # Create simple test data
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            "embedding_similarity": np.random.uniform(0.3, 0.9, 50),
            "groq_quality_score": np.random.uniform(3.0, 8.0, 50),
            "source_language": ["en"] * 25 + ["es"] * 25,
            "target_language": ["es"] * 25 + ["en"] * 25
        })
        
        analyzer = CorrelationAnalyzer()
        analyzer.load_data(test_data)
        
        # Test correlation computation
        corr, p_value = analyzer.compute_pearson_correlation("embedding_similarity", "groq_quality_score")
        print(f"✓ Computed correlation: {corr:.4f} (p={p_value:.4f})")
        
        # Test correlation matrix
        matrix = analyzer.generate_correlation_matrix()
        if not matrix.empty:
            print("✓ Generated correlation matrix")
        
    except ImportError as e:
        print(f"✗ Failed to import CorrelationAnalyzer: {e}")
    except Exception as e:
        print(f"✗ Error testing CorrelationAnalyzer: {e}")
    
    # Test PredictionModelBuilder
    try:
        from prediction_model_builder import PredictionModelBuilder
        print("✓ Successfully imported PredictionModelBuilder")
        
        builder = PredictionModelBuilder()
        
        # Test dataset preparation
        X_train, X_test, y_train, y_test = builder.prepare_dataset(test_data, "groq_quality_score")
        print(f"✓ Prepared dataset: {len(X_train)} train, {len(X_test)} test samples")
        
        # Test model building
        model = builder.build_linear_model(X_train, y_train)
        print("✓ Built linear model")
        
        # Test model evaluation
        metrics = builder.evaluate_model(model, X_test, y_test)
        print(f"✓ Evaluated model: R²={metrics.get('r2_score', 0):.4f}")
        
    except ImportError as e:
        print(f"✗ Failed to import PredictionModelBuilder: {e}")
    except Exception as e:
        print(f"✗ Error testing PredictionModelBuilder: {e}")
    
    # Test ThresholdOptimizer
    try:
        from threshold_optimizer import ThresholdOptimizer
        print("✓ Successfully imported ThresholdOptimizer")
        
        optimizer = ThresholdOptimizer()
        
        # Create mock correlation data (only numeric columns)
        numeric_data = test_data.select_dtypes(include=['number'])
        correlation_data = {
            "correlation_matrix": numeric_data.corr().to_dict(),
            "best_features": {
                "embedding_similarity": (0.6, 0.01)
            }
        }
        
        optimizer.load_correlation_data(correlation_data)
        print("✓ Loaded correlation data")
        
        # Test threshold optimization
        thresholds = optimizer.find_optimal_thresholds("embedding_similarity", quality_levels=5)
        if thresholds:
            print("✓ Found optimal thresholds")
        
        # Test weight optimization
        weights = optimizer.optimize_quality_weights(["embedding_similarity", "groq_quality_score"])
        if weights:
            print(f"✓ Optimized weights: {weights}")
        
    except ImportError as e:
        print(f"✗ Failed to import ThresholdOptimizer: {e}")
    except Exception as e:
        print(f"✗ Error testing ThresholdOptimizer: {e}")

def main():
    """Run all quality learning system tests."""
    print("=" * 60)
    print("QUALITY LEARNING SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Test individual components first
    print("\n🔧 Testing Individual Components...")
    component_success = test_individual_components()
    
    if component_success:
        print("✓ All individual components passed!")
    else:
        print("✗ Some individual components failed!")
    
    # Test full system integration
    print("\n🔄 Testing Full System Integration...")
    system_success = test_quality_learning_engine()
    
    if system_success:
        print("\n🎉 ALL TESTS PASSED! Quality Learning System is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED! Please check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 