# performance_monitor.py

import time
import os
import json
import logging
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import threading
import statistics
from collections import defaultdict, deque
import traceback

class PerformanceMonitor:
    """
    Monitors and records performance metrics for the translation evaluator system including:
    - Analysis speed
    - Memory usage
    - API response times
    - Component reliability
    - System resource utilization
    """
    
    def __init__(self, config_manager=None, log_dir="./logs/performance"):
        """
        Initialize the performance monitor.
        
        Args:
            config_manager: Optional configuration manager for settings
            log_dir: Directory for storing performance logs
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Set up log directory
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Current system info
        self.process = psutil.Process(os.getpid())
        self.system_info = self._get_system_info()
        
        # Performance metrics storage
        self.timers = {}  # For tracking operation times
        self.api_timings = defaultdict(list)  # API response times by endpoint
        self.memory_snapshots = []  # Memory usage over time
        self.component_stats = defaultdict(lambda: {"success": 0, "failure": 0, "total_time": 0})
        
        # Moving averages for real-time metrics
        self.window_size = 20
        self.api_response_times = defaultdict(lambda: deque(maxlen=self.window_size))
        self.analysis_times = deque(maxlen=self.window_size)
        
        # Session information
        self.session_start = datetime.now()
        self.session_id = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}"
        
        # Start memory monitoring thread
        self.monitoring_interval = 5  # seconds
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._memory_monitor_thread, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Performance monitoring initialized. Session ID: {self.session_id}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context."""
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory": psutil.virtual_memory().total,
                "available_memory": psutil.virtual_memory().available
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    def _memory_monitor_thread(self):
        """Background thread to periodically record memory usage."""
        while not self.stop_monitoring:
            try:
                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "rss": self.process.memory_info().rss,  # Resident Set Size
                    "vms": self.process.memory_info().vms,  # Virtual Memory Size
                    "percent": self.process.memory_percent(),
                    "system_percent": psutil.virtual_memory().percent
                }
                self.memory_snapshots.append(snapshot)
                
                # Periodically save snapshots to prevent memory buildup
                if len(self.memory_snapshots) > 1000:
                    self._save_memory_snapshots()
                    self.memory_snapshots = []
                    
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                
            time.sleep(self.monitoring_interval)
    
    def start_timer(self, operation_name: str):
        """
        Start a timer for the specified operation.
        
        Args:
            operation_name: Name of the operation being timed
        """
        self.timers[operation_name] = time.time()
    
    def stop_timer(self, operation_name: str) -> float:
        """
        Stop a timer and return the elapsed time.
        
        Args:
            operation_name: Name of the operation being timed
            
        Returns:
            Elapsed time in seconds
        """
        if operation_name not in self.timers:
            self.logger.warning(f"Timer '{operation_name}' was not started")
            return 0
            
        elapsed = time.time() - self.timers[operation_name]
        
        # Store analysis times for moving average
        if operation_name.startswith("analysis_"):
            self.analysis_times.append(elapsed)
            
        # Clean up timer
        del self.timers[operation_name]
        
        return elapsed
    
    def record_api_response(self, api_name: str, endpoint: str, response_time: float, success: bool, 
                          status_code: Optional[int] = None, error_message: Optional[str] = None):
        """
        Record API response metrics.
        
        Args:
            api_name: Name of the API (e.g., "groq", "embedding")
            endpoint: Specific endpoint or operation called
            response_time: Time taken for the response in seconds
            success: Whether the API call was successful
            status_code: Optional HTTP status code
            error_message: Optional error message if failure
        """
        key = f"{api_name}_{endpoint}"
        
        # Update moving average
        self.api_response_times[key].append(response_time)
        
        # Store detailed record
        record = {
            "timestamp": datetime.now().isoformat(),
            "api": api_name,
            "endpoint": endpoint,
            "response_time": response_time,
            "success": success
        }
        
        if status_code is not None:
            record["status_code"] = status_code
            
        if error_message is not None:
            record["error_message"] = error_message
            
        self.api_timings[key].append(record)
        
        # Log slow responses
        if response_time > self._get_threshold(api_name):
            self.logger.warning(f"Slow {api_name} response for {endpoint}: {response_time:.2f}s")
    
    def _get_threshold(self, api_name: str) -> float:
        """Get the threshold for considering a response slow."""
        thresholds = {
            "groq": 5.0,  # 5 seconds
            "embedding": 1.0,  # 1 second
            "default": 2.0  # 2 seconds
        }
        return thresholds.get(api_name, thresholds["default"])
    
    def record_component_execution(self, component_name: str, success: bool, execution_time: float):
        """
        Record component execution statistics for reliability tracking.
        
        Args:
            component_name: Name of the component
            success: Whether the execution was successful
            execution_time: Time taken to execute
        """
        stats = self.component_stats[component_name]
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        stats["total_time"] += execution_time
    
    def get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary of memory usage metrics
        """
        try:
            memory_info = self.process.memory_info()
            return {
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": self.process.memory_percent(),
                "system_percent": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """
        Get current CPU usage statistics.
        
        Returns:
            Dictionary of CPU usage metrics
        """
        try:
            return {
                "process_percent": self.process.cpu_percent(interval=0.1),
                "system_percent": psutil.cpu_percent(interval=0.1)
            }
        except Exception as e:
            self.logger.error(f"Error getting CPU usage: {e}")
            return {"error": str(e)}
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics across all metrics.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            "session_id": self.session_id,
            "session_duration": str(datetime.now() - self.session_start),
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "current_memory": self.get_memory_usage(),
            "current_cpu": self.get_cpu_usage(),
            "component_reliability": {},
            "api_performance": {},
            "analysis_performance": {}
        }
        
        # Component reliability
        for component, data in self.component_stats.items():
            total = data["success"] + data["failure"]
            if total > 0:
                reliability = data["success"] / total * 100
                avg_time = data["total_time"] / total if total > 0 else 0
                stats["component_reliability"][component] = {
                    "success_rate": round(reliability, 2),
                    "total_executions": total,
                    "failures": data["failure"],
                    "avg_execution_time": round(avg_time, 3)
                }
        
        # API performance
        for key, timings in self.api_timings.items():
            if not timings:
                continue
                
            response_times = [t["response_time"] for t in timings]
            success_count = sum(1 for t in timings if t["success"])
            
            api_stats = {
                "count": len(timings),
                "success_rate": round(success_count / len(timings) * 100, 2) if timings else 0,
                "avg_response_time": round(statistics.mean(response_times), 3) if response_times else 0,
                "min_response_time": round(min(response_times), 3) if response_times else 0,
                "max_response_time": round(max(response_times), 3) if response_times else 0
            }
            
            if len(response_times) > 1:
                api_stats["std_deviation"] = round(statistics.stdev(response_times), 3)
                
            if len(response_times) >= 5:
                api_stats["percentile_95"] = round(statistics.quantiles(response_times, n=20)[-1], 3)
                
            stats["api_performance"][key] = api_stats
        
        # Analysis performance
        analysis_times_list = list(self.analysis_times)
        if analysis_times_list:
            stats["analysis_performance"] = {
                "count": len(analysis_times_list),
                "avg_time": round(statistics.mean(analysis_times_list), 3),
                "min_time": round(min(analysis_times_list), 3),
                "max_time": round(max(analysis_times_list), 3)
            }
            
            if len(analysis_times_list) > 1:
                stats["analysis_performance"]["std_deviation"] = round(statistics.stdev(analysis_times_list), 3)
                
            if len(analysis_times_list) >= 5:
                stats["analysis_performance"]["percentile_95"] = round(statistics.quantiles(analysis_times_list, n=20)[-1], 3)
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a brief performance summary for quick display.
        
        Returns:
            Dictionary with key performance indicators
        """
        statistics_data = self.compute_statistics()
        
        # Extract memory trend
        memory_trend = "stable"
        if len(self.memory_snapshots) >= 10:
            first_five_avg = statistics.mean([s["rss"] for s in self.memory_snapshots[:5]])
            last_five_avg = statistics.mean([s["rss"] for s in self.memory_snapshots[-5:]])
            
            if last_five_avg > first_five_avg * 1.25:  # 25% increase
                memory_trend = "increasing"
            elif last_five_avg < first_five_avg * 0.9:  # 10% decrease
                memory_trend = "decreasing"
        
        # Get worst performing component
        component_reliability = statistics_data.get("component_reliability", {})
        worst_component = None
        lowest_reliability = 100
        
        for component, stats in component_reliability.items():
            if stats["success_rate"] < lowest_reliability:
                lowest_reliability = stats["success_rate"]
                worst_component = component
        
        # Quick summary
        summary = {
            "session_duration": statistics_data.get("session_duration"),
            "memory_usage_mb": round(statistics_data.get("current_memory", {}).get("rss", 0) / (1024 * 1024), 2),
            "memory_trend": memory_trend,
            "cpu_usage": statistics_data.get("current_cpu", {}).get("process_percent", 0),
            "avg_analysis_time": statistics_data.get("analysis_performance", {}).get("avg_time", 0),
            "critical_components": {}
        }
        
        # Add critical component info
        for component, stats in component_reliability.items():
            if stats["success_rate"] < 90:  # Less than 90% success rate is concerning
                summary["critical_components"][component] = {
                    "success_rate": stats["success_rate"],
                    "failures": stats["failures"]
                }
        
        if worst_component:
            summary["worst_component"] = {
                "name": worst_component,
                "success_rate": lowest_reliability
            }
        
        return summary
    
    def save_statistics(self, filename: Optional[str] = None) -> str:
        """
        Save current statistics to a file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"stats_{self.session_id}.json"
            
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            stats = self.compute_statistics()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
                
            self.logger.info(f"Performance statistics saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")
            return ""
    
    def _save_memory_snapshots(self):
        """Save memory snapshots to a file."""
        if not self.memory_snapshots:
            return
            
        filename = f"memory_{self.session_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            # If file exists, append to it; otherwise create new
            if os.path.exists(filepath):
                with open(filepath, 'r+', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    
                    data.extend(self.memory_snapshots)
                    
                    # Reset file pointer and write updated data
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.memory_snapshots, f, indent=2)
                    
            self.logger.debug(f"Memory snapshots saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving memory snapshots: {e}")
    
    def generate_performance_report(self, format='markdown') -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            format: Report format ('markdown', 'text', or 'json')
            
        Returns:
            Formatted performance report
        """
        stats = self.compute_statistics()
        
        if format == 'json':
            return json.dumps(stats, indent=2)
        
        if format == 'text':
            return self._generate_text_report(stats)
            
        # Default to markdown
        return self._generate_markdown_report(stats)
    
    def _generate_markdown_report(self, stats: Dict[str, Any]) -> str:
        """Generate markdown performance report."""
        report = []
        report.append("# Performance Monitoring Report")
        report.append(f"**Session ID:** {stats['session_id']}")
        report.append(f"**Duration:** {stats['session_duration']}")
        report.append(f"**Generated:** {datetime.now().isoformat()}")
        report.append("")
        
        report.append("## System Information")
        for key, value in stats['system_info'].items():
            # Format memory values in MB
            if 'memory' in key and isinstance(value, int):
                value_mb = value / (1024 * 1024)
                report.append(f"**{key.replace('_', ' ').title()}:** {value_mb:.2f} MB")
            else:
                report.append(f"**{key.replace('_', ' ').title()}:** {value}")
        report.append("")
        
        report.append("## Current Resource Usage")
        memory = stats['current_memory']
        report.append("### Memory")
        report.append(f"**RSS:** {memory.get('rss', 0) / (1024 * 1024):.2f} MB")
        report.append(f"**VMS:** {memory.get('vms', 0) / (1024 * 1024):.2f} MB")
        report.append(f"**Process Memory %:** {memory.get('percent', 0):.2f}%")
        report.append(f"**System Memory %:** {memory.get('system_percent', 0):.2f}%")
        report.append("")
        
        report.append("### CPU")
        cpu = stats['current_cpu']
        report.append(f"**Process CPU %:** {cpu.get('process_percent', 0):.2f}%")
        report.append(f"**System CPU %:** {cpu.get('system_percent', 0):.2f}%")
        report.append("")
        
        report.append("## Component Reliability")
        if not stats['component_reliability']:
            report.append("*No component data available*")
        else:
            report.append("| Component | Success Rate | Executions | Failures | Avg Time (s) |")
            report.append("|-----------|-------------|------------|----------|--------------|")
            for component, data in stats['component_reliability'].items():
                report.append(
                    f"| {component} | {data['success_rate']}% | "
                    f"{data['total_executions']} | {data['failures']} | "
                    f"{data['avg_execution_time']} |"
                )
        report.append("")
        
        report.append("## API Performance")
        if not stats['api_performance']:
            report.append("*No API data available*")
        else:
            report.append("| API | Count | Success Rate | Avg Time (s) | Min Time (s) | Max Time (s) | 95th % (s) |")
            report.append("|-----|-------|-------------|--------------|--------------|--------------|------------|")
            for api, data in stats['api_performance'].items():
                percentile = data.get('percentile_95', 'N/A')
                report.append(
                    f"| {api} | {data['count']} | {data['success_rate']}% | "
                    f"{data['avg_response_time']} | {data['min_response_time']} | "
                    f"{data['max_response_time']} | {percentile} |"
                )
        report.append("")
        
        report.append("## Analysis Performance")
        if not stats.get('analysis_performance'):
            report.append("*No analysis data available*")
        else:
            analysis = stats['analysis_performance']
            report.append(f"**Total Analyses:** {analysis['count']}")
            report.append(f"**Average Time:** {analysis['avg_time']} seconds")
            report.append(f"**Min Time:** {analysis['min_time']} seconds")
            report.append(f"**Max Time:** {analysis['max_time']} seconds")
            if 'std_deviation' in analysis:
                report.append(f"**Std Deviation:** {analysis['std_deviation']} seconds")
            if 'percentile_95' in analysis:
                report.append(f"**95th Percentile:** {analysis['percentile_95']} seconds")
        report.append("")
        
        report.append("## Recommendations")
        
        # Generate recommendations based on metrics
        recommendations = self._generate_recommendations(stats)
        if recommendations:
            for rec in recommendations:
                report.append(f"- {rec}")
        else:
            report.append("*No recommendations at this time*")
        
        return "\n".join(report)
    
    def _generate_text_report(self, stats: Dict[str, Any]) -> str:
        """Generate plain text performance report."""
        # Convert markdown to plain text
        md_report = self._generate_markdown_report(stats)
        text_report = md_report.replace('# ', '').replace('## ', '\n').replace('### ', '')
        text_report = text_report.replace('**', '').replace('*', '')
        
        # Remove markdown table formatting
        import re
        text_report = re.sub(r'\|.*\|\n\|[-|]*\|\n', '\n', text_report)
        text_report = re.sub(r'\|(.*?)\|', r'\1', text_report)
        
        return text_report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on statistics."""
        recommendations = []
        
        # Check component reliability
        for component, data in stats.get('component_reliability', {}).items():
            if data['success_rate'] < 80:
                recommendations.append(
                    f"Critical: {component} has very low reliability ({data['success_rate']}%). "
                    f"Investigate the {data['failures']} failures."
                )
            elif data['success_rate'] < 95:
                recommendations.append(
                    f"Warning: {component} has below-optimal reliability ({data['success_rate']}%). "
                    f"Consider improving error handling."
                )
        
        # Check API performance
        slow_apis = []
        for api, data in stats.get('api_performance', {}).items():
            # APIs with high latency variability
            if data.get('std_deviation', 0) > data['avg_response_time'] * 0.5 and data['count'] > 5:
                slow_apis.append(api)
                recommendations.append(
                    f"API performance: {api} has high response time variability "
                    f"(stddev: {data.get('std_deviation', 0)}s). Consider implementing retry mechanisms."
                )
        
        # Check memory usage
        memory = stats.get('current_memory', {})
        if memory.get('percent', 0) > 70:
            recommendations.append(
                "Memory usage is high (>70% of available memory). Consider optimizing memory usage "
                "or increasing available memory."
            )
        
        # Analysis performance
        analysis = stats.get('analysis_performance', {})
        if analysis.get('avg_time', 0) > 10:  # More than 10 seconds on average
            recommendations.append(
                f"Analysis is taking {analysis.get('avg_time', 0):.1f} seconds on average, which is slow. "
                "Consider optimizing or adding caching mechanisms."
            )
        
        # General recommendations if no specific issues found
        if not recommendations and stats.get('component_reliability'):
            recommendations.append(
                "All components are functioning well. Consider adding more logging for finer-grained performance insights."
            )
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources and save final data."""
        self.stop_monitoring = True
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        
        self._save_memory_snapshots()
        self.save_statistics()
        
        self.logger.info("Performance monitoring stopped and data saved") 