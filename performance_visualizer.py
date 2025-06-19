# performance_visualizer.py
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def load_performance_data(log_dir="./logs/performance"):
    """Load all performance data from log directory."""
    data = {
        "stats": [],
        "memory": []
    }
    
    # Load stats files
    for filename in os.listdir(log_dir):
        if filename.startswith("stats_") and filename.endswith(".json"):
            try:
                with open(os.path.join(log_dir, filename), 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    data["stats"].append(stats)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        # Load memory files
        elif filename.startswith("memory_") and filename.endswith(".json"):
            try:
                with open(os.path.join(log_dir, filename), 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    data["memory"].extend(memory_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return data

def plot_memory_usage(memory_data, output_file=None):
    """Plot memory usage over time."""
    if not memory_data:
        print("No memory data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(memory_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Convert memory values to MB
    df['rss_mb'] = df['rss'] / (1024 * 1024)
    df['vms_mb'] = df['vms'] / (1024 * 1024)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['rss_mb'], label='RSS (MB)')
    plt.plot(df['timestamp'], df['percent'], label='Process Memory %')
    plt.plot(df['timestamp'], df['system_percent'], label='System Memory %')
    
    plt.title("Memory Usage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Usage")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Memory usage plot saved to {output_file}")
    else:
        plt.show()

def plot_api_performance(stats_data, output_file=None):
    """Plot API performance metrics."""
    if not stats_data:
        print("No stats data available")
        return
    
    # Extract API performance data
    api_data = []
    
    for stat in stats_data:
        timestamp = datetime.fromisoformat(stat.get('timestamp', '2023-01-01T00:00:00'))
        session_id = stat.get('session_id', 'unknown')
        
        for api_name, perf in stat.get('api_performance', {}).items():
            api_data.append({
                'timestamp': timestamp,
                'session_id': session_id,
                'api_name': api_name,
                'count': perf.get('count', 0),
                'success_rate': perf.get('success_rate', 0),
                'avg_response_time': perf.get('avg_response_time', 0),
                'max_response_time': perf.get('max_response_time', 0)
            })
    
    if not api_data:
        print("No API performance data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(api_data)
    
    # Group by API name
    grouped = df.groupby('api_name')
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot average response times
    plt.subplot(2, 1, 1)
    for name, group in grouped:
        plt.plot(group['timestamp'], group['avg_response_time'], marker='o', linestyle='-', label=name)
    
    plt.title("API Average Response Times")
    plt.xlabel("Time")
    plt.ylabel("Response Time (s)")
    plt.legend()
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(2, 1, 2)
    for name, group in grouped:
        plt.plot(group['timestamp'], group['success_rate'], marker='o', linestyle='-', label=name)
    
    plt.title("API Success Rates")
    plt.xlabel("Time")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"API performance plot saved to {output_file}")
    else:
        plt.show()

def plot_component_reliability(stats_data, output_file=None):
    """Plot component reliability metrics."""
    if not stats_data:
        print("No stats data available")
        return
    
    # Extract component reliability data
    component_data = []
    
    for stat in stats_data:
        timestamp = datetime.fromisoformat(stat.get('timestamp', '2023-01-01T00:00:00'))
        session_id = stat.get('session_id', 'unknown')
        
        for component_name, reliability in stat.get('component_reliability', {}).items():
            component_data.append({
                'timestamp': timestamp,
                'session_id': session_id,
                'component_name': component_name,
                'success_rate': reliability.get('success_rate', 0),
                'total_executions': reliability.get('total_executions', 0),
                'failures': reliability.get('failures', 0),
                'avg_execution_time': reliability.get('avg_execution_time', 0)
            })
    
    if not component_data:
        print("No component reliability data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(component_data)
    
    # Group by component name
    grouped = df.groupby('component_name')
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot success rates
    for name, group in grouped:
        ax1.plot(group['timestamp'], group['success_rate'], marker='o', linestyle='-', label=name)
    
    ax1.set_title("Component Success Rates")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Success Rate (%)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot average execution times
    for name, group in grouped:
        ax2.plot(group['timestamp'], group['avg_execution_time'], marker='o', linestyle='-', label=name)
    
    ax2.set_title("Component Average Execution Times")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Execution Time (s)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Component reliability plot saved to {output_file}")
    else:
        plt.show()

def plot_analysis_performance(stats_data, output_file=None):
    """Plot analysis performance metrics."""
    if not stats_data:
        print("No stats data available")
        return
    
    # Extract analysis performance data
    analysis_data = []
    
    for stat in stats_data:
        timestamp = datetime.fromisoformat(stat.get('timestamp', '2023-01-01T00:00:00'))
        session_id = stat.get('session_id', 'unknown')
        
        performance = stat.get('analysis_performance', {})
        if performance:
            analysis_data.append({
                'timestamp': timestamp,
                'session_id': session_id,
                'count': performance.get('count', 0),
                'avg_time': performance.get('avg_time', 0),
                'min_time': performance.get('min_time', 0),
                'max_time': performance.get('max_time', 0),
                'std_deviation': performance.get('std_deviation', 0)
            })
    
    if not analysis_data:
        print("No analysis performance data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['timestamp'], df['avg_time'], marker='o', linestyle='-', label='Avg Time')
    plt.plot(df['timestamp'], df['min_time'], marker='s', linestyle='--', label='Min Time')
    plt.plot(df['timestamp'], df['max_time'], marker='^', linestyle='--', label='Max Time')
    
    plt.title("Analysis Performance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Duration (s)")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Analysis performance plot saved to {output_file}")
    else:
        plt.show()

def generate_dashboard(data, output_dir="./performance_dashboard"):
    """Generate a comprehensive performance dashboard with all plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_memory_usage(data["memory"], os.path.join(output_dir, "memory_usage.png"))
    plot_api_performance(data["stats"], os.path.join(output_dir, "api_performance.png"))
    plot_component_reliability(data["stats"], os.path.join(output_dir, "component_reliability.png"))
    plot_analysis_performance(data["stats"], os.path.join(output_dir, "analysis_performance.png"))
    
    # Create HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Translation Evaluator Performance Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #333; }}
        .plot {{ margin-bottom: 30px; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        .plot h2 {{ color: #555; }}
    </style>
</head>
<body>
    <h1>Translation Evaluator Performance Dashboard</h1>
    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="plot">
        <h2>Memory Usage</h2>
        <img src="memory_usage.png" alt="Memory Usage">
    </div>
    
    <div class="plot">
        <h2>API Performance</h2>
        <img src="api_performance.png" alt="API Performance">
    </div>
    
    <div class="plot">
        <h2>Component Reliability</h2>
        <img src="component_reliability.png" alt="Component Reliability">
    </div>
    
    <div class="plot">
        <h2>Analysis Performance</h2>
        <img src="analysis_performance.png" alt="Analysis Performance">
    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print(f"Performance dashboard generated at {output_dir}/index.html")

def main():
    parser = argparse.ArgumentParser(description="Translation Evaluator Performance Visualizer")
    parser.add_argument("--log-dir", default="./logs/performance", help="Directory containing performance logs")
    parser.add_argument("--output-dir", default="./performance_dashboard", help="Output directory for dashboard")
    parser.add_argument("--memory", action="store_true", help="Plot memory usage")
    parser.add_argument("--api", action="store_true", help="Plot API performance")
    parser.add_argument("--components", action="store_true", help="Plot component reliability")
    parser.add_argument("--analysis", action="store_true", help="Plot analysis performance")
    parser.add_argument("--dashboard", action="store_true", help="Generate full dashboard")
    
    args = parser.parse_args()
    
    # Load data
    data = load_performance_data(args.log_dir)
    
    # Generate plots based on arguments
    if args.dashboard:
        generate_dashboard(data, args.output_dir)
    else:
        if args.memory:
            plot_memory_usage(data["memory"])
        if args.api:
            plot_api_performance(data["stats"])
        if args.components:
            plot_component_reliability(data["stats"])
        if args.analysis:
            plot_analysis_performance(data["stats"])
        
        # If no specific plot was requested, show all
        if not (args.memory or args.api or args.components or args.analysis):
            plot_memory_usage(data["memory"])
            plot_api_performance(data["stats"])
            plot_component_reliability(data["stats"])
            plot_analysis_performance(data["stats"])

if __name__ == "__main__":
    main() 