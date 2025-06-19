#!/usr/bin/env python3
"""
Tests for the performance monitoring system
"""

import pytest
import tempfile
import shutil
import time
import json
import os
from unittest.mock import Mock, patch

try:
    from performance_monitor import PerformanceMonitor
    performance_monitor_available = True
except ImportError:
    performance_monitor_available = False


@pytest.mark.skipif(not performance_monitor_available, reason="Performance monitor not available")
class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.monitor = PerformanceMonitor(log_dir=self.test_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'monitor'):
            self.monitor.cleanup()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_monitor_initialization(self):
        """Test that monitor initializes correctly."""
        assert self.monitor.log_dir == self.test_dir
        assert self.monitor.session_id.startswith("session_")
        assert isinstance(self.monitor.system_info, dict)
        
    def test_timer_operations(self):
        """Test timer start/stop functionality."""
        # Test basic timer
        self.monitor.start_timer("test_operation")
        time.sleep(0.1)  # Small delay
        elapsed = self.monitor.stop_timer("test_operation")
        
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be reasonable
        
        # Test analysis timer tracking
        self.monitor.start_timer("analysis_test")
        time.sleep(0.05)
        elapsed = self.monitor.stop_timer("analysis_test")
        
        assert len(self.monitor.analysis_times) == 1
        assert self.monitor.analysis_times[0] >= 0.05
        
    def test_api_response_recording(self):
        """Test API response metrics recording."""
        self.monitor.record_api_response(
            api_name="test_api",
            endpoint="test_endpoint", 
            response_time=1.5,
            success=True,
            status_code=200
        )
        
        key = "test_api_test_endpoint"
        assert key in self.monitor.api_timings
        assert len(self.monitor.api_timings[key]) == 1
        
        record = self.monitor.api_timings[key][0]
        assert record["api"] == "test_api"
        assert record["endpoint"] == "test_endpoint"
        assert record["response_time"] == 1.5
        assert record["success"] == True
        assert record["status_code"] == 200
        
    def test_component_execution_tracking(self):
        """Test component reliability tracking."""
        # Record successful execution
        self.monitor.record_component_execution("test_component", True, 0.5)
        
        # Record failed execution
        self.monitor.record_component_execution("test_component", False, 0.3)
        
        stats = self.monitor.component_stats["test_component"]
        assert stats["success"] == 1
        assert stats["failure"] == 1
        assert stats["total_time"] == 0.8
        
    def test_memory_usage_retrieval(self):
        """Test memory usage information."""
        memory_info = self.monitor.get_memory_usage()
        
        assert "rss" in memory_info
        assert "vms" in memory_info
        assert "percent" in memory_info
        assert "system_percent" in memory_info
        
        # Values should be reasonable
        assert memory_info["rss"] > 0
        assert memory_info["percent"] >= 0
        assert memory_info["system_percent"] >= 0
        
    def test_cpu_usage_retrieval(self):
        """Test CPU usage information."""
        cpu_info = self.monitor.get_cpu_usage()
        
        assert "process_percent" in cpu_info
        assert "system_percent" in cpu_info
        
        # Values should be reasonable
        assert cpu_info["process_percent"] >= 0
        assert cpu_info["system_percent"] >= 0
        
    def test_statistics_computation(self):
        """Test comprehensive statistics computation."""
        # Add some test data
        self.monitor.record_api_response("groq", "evaluate", 2.0, True)
        self.monitor.record_component_execution("embedding", True, 0.5)
        self.monitor.start_timer("analysis_test")
        time.sleep(0.1)
        self.monitor.stop_timer("analysis_test")
        
        stats = self.monitor.compute_statistics()
        
        assert "session_id" in stats
        assert "system_info" in stats
        assert "current_memory" in stats
        assert "current_cpu" in stats
        assert "component_reliability" in stats
        assert "api_performance" in stats
        assert "analysis_performance" in stats
        
        # Check component stats
        assert "embedding" in stats["component_reliability"]
        component_stats = stats["component_reliability"]["embedding"]
        assert component_stats["success_rate"] == 100.0
        assert component_stats["total_executions"] == 1
        
        # Check API stats
        assert "groq_evaluate" in stats["api_performance"]
        api_stats = stats["api_performance"]["groq_evaluate"]
        assert api_stats["success_rate"] == 100.0
        assert api_stats["avg_response_time"] == 2.0
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some analysis data
        self.monitor.analysis_times.extend([0.5, 1.0, 0.8])
        
        summary = self.monitor.get_performance_summary()
        
        assert "session_duration" in summary
        assert "memory_usage_mb" in summary
        assert "avg_analysis_time" in summary
        assert "memory_trend" in summary
        
        assert summary["avg_analysis_time"] > 0
        assert summary["memory_usage_mb"] > 0
        
    def test_statistics_saving(self):
        """Test statistics file saving."""
        # Add some test data
        self.monitor.record_component_execution("test_component", True, 0.5)
        
        # Save statistics
        filepath = self.monitor.save_statistics("test_stats.json")
        
        assert filepath != ""
        assert os.path.exists(filepath)
        
        # Load and verify the saved data
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        assert "session_id" in data
        assert "component_reliability" in data
        assert "test_component" in data["component_reliability"]
        
    def test_report_generation(self):
        """Test performance report generation."""
        # Add some test data
        self.monitor.record_api_response("groq", "evaluate", 1.5, True)
        self.monitor.record_component_execution("embedding", True, 0.5)
        
        # Test markdown report
        markdown_report = self.monitor.generate_performance_report(format='markdown')
        assert "# Performance Monitoring Report" in markdown_report
        assert "Component Reliability" in markdown_report
        assert "API Performance" in markdown_report
        
        # Test JSON report
        json_report = self.monitor.generate_performance_report(format='json')
        data = json.loads(json_report)
        assert "session_id" in data
        assert "component_reliability" in data
        
        # Test text report
        text_report = self.monitor.generate_performance_report(format='text')
        assert "Performance Monitoring Report" in text_report
        
    def test_recommendations_generation(self):
        """Test performance recommendations."""
        # Create a component with low reliability
        for i in range(10):
            success = i < 7  # 70% success rate
            self.monitor.record_component_execution("unreliable_component", success, 0.1)
            
        stats = self.monitor.compute_statistics()
        recommendations = self.monitor._generate_recommendations(stats)
        
        # Should recommend investigating the unreliable component
        assert len(recommendations) > 0
        assert any("unreliable_component" in rec for rec in recommendations)
        
    def test_memory_monitoring_thread(self):
        """Test background memory monitoring."""
        # Wait a bit for memory snapshots to be collected
        time.sleep(6)  # Wait longer than monitoring interval
        
        # Should have at least one memory snapshot
        assert len(self.monitor.memory_snapshots) > 0
        
        snapshot = self.monitor.memory_snapshots[0]
        assert "timestamp" in snapshot
        assert "rss" in snapshot
        assert "percent" in snapshot
        
    def test_slow_api_detection(self):
        """Test slow API response detection."""
        # Mock logger to capture warnings
        with patch.object(self.monitor.logger, 'warning') as mock_warning:
            # Record a slow response (above threshold)
            self.monitor.record_api_response("groq", "evaluate", 10.0, True)
            
            # Should log a warning for slow response
            mock_warning.assert_called_once()
            assert "Slow groq response" in str(mock_warning.call_args)
            
    def test_cleanup(self):
        """Test proper cleanup of resources."""
        # Add some data
        self.monitor.record_component_execution("test", True, 0.1)
        
        # Cleanup should save data and stop monitoring
        self.monitor.cleanup()
        
        # Should have saved statistics
        stats_files = [f for f in os.listdir(self.test_dir) if f.startswith("stats_")]
        assert len(stats_files) > 0
        
        # Monitoring thread should be stopped
        assert self.monitor.stop_monitoring == True


@pytest.mark.skipif(not performance_monitor_available, reason="Performance monitor not available")
def test_monitor_with_config_manager():
    """Test monitor initialization with config manager."""
    mock_config = Mock()
    mock_config.get.return_value = "/tmp/test_logs"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = PerformanceMonitor(config_manager=mock_config, log_dir=temp_dir)
        
        assert monitor.config_manager == mock_config
        assert monitor.log_dir == temp_dir
        
        monitor.cleanup()


if __name__ == "__main__":
    pytest.main([__file__]) 