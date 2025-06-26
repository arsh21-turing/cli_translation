# tests/test_cache_statistics.py

import unittest
import sys
import os
import tempfile
from unittest.mock import MagicMock, patch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions we want to test
from main import format_cache_stats, calculate_cache_efficiency, interpret_cache_stats

class TestCacheStatistics(unittest.TestCase):
    """Test the cache statistics functionality."""
    
    def setUp(self):
        # Create sample cache statistics for testing
        self.sample_stats = {
            "memory_size": 150,
            "memory_max_size": 1000,
            "disk_size": 500,
            "disk_max_size": 5000,
            "memory_hits": 800,
            "memory_misses": 200,
            "disk_hits": 100,
            "disk_misses": 50,
            "memory_evictions": 50,
            "disk_evictions": 10,
            "api_calls_saved": 200,
            "computation_time_saved": 120.5,  # seconds
            "bytes_saved": 1048576  # 1MB
        }
    
    def test_format_cache_stats_json(self):
        """Test JSON formatting of cache statistics."""
        formatted = format_cache_stats(self.sample_stats, format='json')
        
        # Verify it's valid JSON
        parsed = json.loads(formatted)
        
        # Check some key values
        self.assertEqual(parsed["memory_size"], 150)
        self.assertEqual(parsed["api_calls_saved"], 200)
    
    def test_format_cache_stats_text(self):
        """Test text formatting of cache statistics."""
        formatted = format_cache_stats(self.sample_stats, format='text')
        
        # Verify key elements are in the text
        self.assertIn("SMART CACHE STATISTICS", formatted)
        self.assertIn("Memory Cache:", formatted)
        self.assertIn("API Calls Saved:", formatted)
        self.assertIn("800 hits", formatted)
    
    def test_format_cache_stats_markdown(self):
        """Test markdown formatting of cache statistics."""
        formatted = format_cache_stats(self.sample_stats, format='markdown')
        
        # Verify markdown formatting
        self.assertIn("# Smart Cache Statistics", formatted)
        self.assertIn("## Cache Utilization", formatted)
        self.assertIn("- **Memory Cache:**", formatted)
    
    def test_format_cache_stats_html(self):
        """Test HTML formatting of cache statistics."""
        formatted = format_cache_stats(self.sample_stats, format='html')
        
        # Verify HTML formatting
        self.assertIn("<!DOCTYPE html>", formatted)
        self.assertIn("<title>Smart Cache Statistics</title>", formatted)
        self.assertIn('<div class="stat-group">', formatted)

    def test_calculate_cache_efficiency(self):
        """Test cache efficiency calculation."""
        # Test with good stats
        good_stats = self.sample_stats.copy()
        efficiency = calculate_cache_efficiency(good_stats)
        self.assertGreaterEqual(efficiency, 7.0)  # Should be a good score
        
        # Test with poor stats
        poor_stats = self.sample_stats.copy()
        poor_stats["memory_hits"] = 50
        poor_stats["memory_misses"] = 950
        poor_stats["memory_evictions"] = 400
        poor_efficiency = calculate_cache_efficiency(poor_stats)
        
        self.assertLessEqual(poor_efficiency, 4.0)  # Should be a poor score
        self.assertLess(poor_efficiency, efficiency)  # Should be worse than good stats

    def test_interpret_cache_stats(self):
        """Test cache statistics interpretation."""
        interpretation = interpret_cache_stats(self.sample_stats)
        
        # Should contain "INTERPRETATION" section
        self.assertIn("INTERPRETATION:", interpretation)
        
        # Should contain "RECOMMENDATIONS" section
        self.assertIn("RECOMMENDATIONS:", interpretation)
        
        # Check with poor stats for different recommendations
        poor_stats = self.sample_stats.copy()
        poor_stats["memory_hits"] = 50
        poor_stats["memory_misses"] = 950
        poor_interpretation = interpret_cache_stats(poor_stats)
        
        self.assertIn("cache hit rate is low", poor_interpretation.lower())

    def test_empty_stats(self):
        """Test handling of empty or missing statistics."""
        empty_stats = {}
        
        # Should not crash with empty stats
        efficiency = calculate_cache_efficiency(empty_stats)
        self.assertGreaterEqual(efficiency, 0)
        self.assertLessEqual(efficiency, 10)
        
        interpretation = interpret_cache_stats(empty_stats)
        self.assertIn("INTERPRETATION:", interpretation)
        
        formatted = format_cache_stats(empty_stats, format='text')
        self.assertIn("SMART CACHE STATISTICS", formatted)
    
    def test_edge_cases(self):
        """Test edge cases like division by zero."""
        edge_stats = {
            "memory_size": 0,
            "memory_max_size": 0,
            "disk_size": 0,
            "disk_max_size": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "memory_evictions": 0,
            "disk_evictions": 0,
            "api_calls_saved": 0,
            "computation_time_saved": 0,
            "bytes_saved": 0
        }
        
        # Should handle division by zero gracefully
        efficiency = calculate_cache_efficiency(edge_stats)
        self.assertGreaterEqual(efficiency, 0)
        self.assertLessEqual(efficiency, 10)
        
        formatted = format_cache_stats(edge_stats, format='text')
        self.assertIn("0.00%", formatted)  # Should show 0% hit rates
        
    def test_high_performance_stats(self):
        """Test with high-performance cache statistics."""
        high_perf_stats = {
            "memory_size": 900,
            "memory_max_size": 1000,
            "disk_size": 4500,
            "disk_max_size": 5000,
            "memory_hits": 9500,
            "memory_misses": 500,
            "disk_hits": 450,
            "disk_misses": 50,
            "memory_evictions": 10,
            "disk_evictions": 5,
            "api_calls_saved": 1000,
            "computation_time_saved": 3600,  # 1 hour
            "bytes_saved": 1073741824  # 1GB
        }
        
        efficiency = calculate_cache_efficiency(high_perf_stats)
        self.assertGreaterEqual(efficiency, 8.0)  # Should be excellent
        
        interpretation = interpret_cache_stats(high_perf_stats)
        self.assertIn("extremely well", interpretation.lower())

if __name__ == "__main__":
    unittest.main() 