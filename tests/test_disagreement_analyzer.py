"""
Test script for the DisagreementAnalyzer functionality.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from disagreement_analyzer import DisagreementAnalyzer

def create_test_data():
    """Create sample data for testing disagreement analysis."""
    np.random.seed(42)
    
    # Create sample translation data with varying levels of disagreement
    n_samples = 100
    
    # Base quality scores
    quality_scores = np.random.uniform(1, 5, n_samples)
    
    # Create embedding similarity with some correlation but also some disagreement
    embedding_similarity = []
    disagreement_types = []
    
    for i, quality in enumerate(quality_scores):
        if i % 10 == 0:  # 10% high disagreement cases
            # Create high disagreement - opposite rating
            if quality > 3:
                sim = np.random.uniform(0.1, 0.4)  # Low similarity for high quality
                disagreement_types.append("groq_higher")
            else:
                sim = np.random.uniform(0.6, 0.9)  # High similarity for low quality
                disagreement_types.append("embedding_higher")
        elif i % 5 == 0:  # 10% moderate disagreement cases
            # Create moderate disagreement
            if quality > 3:
                sim = np.random.uniform(0.4, 0.6)
            else:
                sim = np.random.uniform(0.4, 0.6)
            disagreement_types.append("moderate")
        else:  # 80% agreement cases
            # Create correlated scores
            sim = 0.2 + (quality - 1) / 4 * 0.6 + np.random.normal(0, 0.1)
            sim = np.clip(sim, 0, 1)
            disagreement_types.append("agreement")
            
        embedding_similarity.append(sim)
    
    # Create sample text data
    source_texts = [f"Sample source text {i} with varying complexity" for i in range(n_samples)]
    translated_texts = [f"Sample translated text {i} with different quality" for i in range(n_samples)]
    
    # Language pairs
    language_pairs = ["en-es", "en-fr", "en-de", "es-en"] * (n_samples // 4)
    if len(language_pairs) < n_samples:
        language_pairs.extend(["en-es"] * (n_samples - len(language_pairs)))
    
    # Create DataFrame
    data = pd.DataFrame({
        "embedding_similarity": embedding_similarity,
        "groq_quality_score": quality_scores,
        "source_text": source_texts,
        "translated_text": translated_texts,
        "source_language": [pair.split("-")[0] for pair in language_pairs],
        "target_language": [pair.split("-")[1] for pair in language_pairs],
        "expected_disagreement_type": disagreement_types
    })
    
    return data

def test_disagreement_analyzer():
    """Test the DisagreementAnalyzer with sample data."""
    print("Testing DisagreementAnalyzer...")
    
    # Create test data
    test_data = create_test_data()
    print(f"Created test data with {len(test_data)} samples")
    
    # Initialize analyzer with moderate threshold
    analyzer = DisagreementAnalyzer(threshold=0.4)
    
    # Load data
    analyzer.load_data(test_data)
    print("✓ Data loaded successfully")
    
    # Identify disagreements
    disagreements = analyzer.identify_disagreements()
    if not disagreements.empty:
        print(f"✓ Identified {len(disagreements)} disagreement cases")
        print(f"  Disagreement rate: {len(disagreements)/len(test_data)*100:.1f}%")
    else:
        print("✗ No disagreements found")
    
    # Classify disagreements
    categories = analyzer.classify_disagreements()
    print(f"✓ Classified disagreements into {len(categories)} categories")
    for category, cases in categories.items():
        print(f"  {category}: {len(cases)} cases")
    
    # Score disagreements
    scored = analyzer.score_disagreements()
    if not scored.empty:
        print(f"✓ Scored disagreements with priority scores")
        print(f"  Average severity: {scored['severity'].mean():.3f}")
        print(f"  Max severity: {scored['severity'].max():.3f}")
    
    # Analyze language patterns
    lang_patterns = analyzer.analyze_language_patterns()
    if lang_patterns:
        print(f"✓ Analyzed language patterns")
        print(f"  Languages analyzed: {lang_patterns['summary']['total_languages']}")
    
    # Analyze text features
    text_features = analyzer.analyze_text_features()
    if text_features:
        print(f"✓ Analyzed text features")
        print(f"  Features analyzed: {len(text_features)}")
    
    # Generate comprehensive report
    report = analyzer.generate_disagreement_report()
    if report:
        print(f"✓ Generated comprehensive report")
        print(f"  Report sections: {len(report)}")
        
        # Print key findings
        if "summary" in report:
            summary = report["summary"]
            print(f"  Summary:")
            print(f"    Total disagreements: {summary['total_disagreements']}")
            print(f"    Disagreement rate: {summary['disagreement_rate']:.1%}")
            print(f"    Average severity: {summary['avg_severity']:.3f}")
    
    # Test recommendations
    recommendations = analyzer._generate_recommendations()
    if recommendations:
        print(f"✓ Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec[:80]}...")
    
    # Test most severe cases
    severe_cases = analyzer.get_most_severe_disagreements(5)
    if severe_cases:
        print(f"✓ Retrieved {len(severe_cases)} most severe cases")
        for i, case in enumerate(severe_cases[:3], 1):
            print(f"  {i}. Severity: {case.get('severity', 0):.3f}, Type: {case.get('disagreement_type', 'Unknown')}")
    
    # Test systematic patterns
    patterns = analyzer.get_systematic_disagreement_patterns()
    if patterns:
        print(f"✓ Analyzed systematic patterns")
        if "direction_pattern" in patterns:
            direction = patterns["direction_pattern"]["dominant_direction"]
            print(f"  Dominant direction: {direction}")
    
    # Test investigation recommendations
    investigations = analyzer.recommend_investigations(5)
    if investigations:
        print(f"✓ Recommended {len(investigations)} cases for investigation")
    
    # Test disagreement prediction/correction
    corrections = analyzer.predict_disagreement_correction()
    if not corrections.empty:
        print(f"✓ Predicted corrections for {len(corrections)} cases")
        print(f"  Average predicted quality: {corrections['predicted_quality'].mean():.3f}")
    
    # Ensure disagreements are identified
    assert not disagreements.empty, "No disagreements found"

    # Ensure report is generated
    assert report, "Failed to generate report"

    # Ensure corrections are predicted
    assert not corrections.empty, "Failed to predict corrections"

    print("\nAll tests completed successfully! ✓")

def test_output_directory():
    """Test saving disagreement cases to output directory."""
    print("\nTesting output functionality...")
    
    # Create test data
    test_data = create_test_data()
    analyzer = DisagreementAnalyzer(threshold=0.3)
    analyzer.load_data(test_data)
    
    # Run analysis
    analyzer.identify_disagreements()
    analyzer.classify_disagreements()
    analyzer.score_disagreements()
    analyzer.analyze_language_patterns()
    analyzer.analyze_text_features()
    
    # Use absolute path for output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_disagreement_output")
    
    # Save disagreement cases
    saved_path = analyzer.save_disagreement_cases(output_dir)
    
    if saved_path:
        print(f"✓ Saved disagreement analysis to {saved_path}")
        
        # Check if files were created
        files_created = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.json'):
                    files_created.append(file)
        
        print(f"✓ Created {len(files_created)} output files:")
        for file in files_created[:5]:  # Show first 5 files
            print(f"  - {file}")
        
        # Clean up test directory
        import shutil
        shutil.rmtree(output_dir)
        print("✓ Cleaned up test output directory")

        # Ensure output directory is created and files are saved
        assert saved_path, "Failed to save disagreement analysis"
        assert files_created, "No JSON files created in output directory"

if __name__ == "__main__":
    print("DisagreementAnalyzer Test Suite")
    print("=" * 50)
    
    try:
        # Run basic functionality tests
        test_disagreement_analyzer()
        
        if test_disagreement_analyzer():
            # Run output tests
            test_output_directory()
            
            print("\n" + "=" * 50)
            print("All tests passed! DisagreementAnalyzer is working correctly.")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("Some tests failed. Please check the implementation.")
            print("=" * 50)
            
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 