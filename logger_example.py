"""
Example demonstrating the unified logging configuration.
"""
import os
import time
import random
from logger_config import get_logger, LoggerConfig

# Get loggers for different components
system_logger = get_logger(__name__, "system")
batch_logger = get_logger(__name__, "batch_processing")
quality_logger = get_logger(__name__, "quality_learning")
api_logger = get_logger(__name__, "api")

def simulate_system_activity():
    """Simulate normal system activity."""
    system_logger.info("System initialized with standard configuration")
    system_logger.debug("Loading configuration parameters from default settings")
    
    if random.random() < 0.2:
        system_logger.warning("System resources running higher than expected")
    
    system_logger.info("System activity completed successfully")

def simulate_batch_processing():
    """Simulate batch processing activity."""
    batch_size = random.randint(50, 200)
    batch_logger.info(f"Starting batch processing with batch size {batch_size}")
    
    for i in range(3):
        files_processed = random.randint(10, 50)
        success_rate = random.uniform(0.9, 1.0)
        batch_logger.info(f"Processed {files_processed} files in batch segment {i+1}, success rate: {success_rate:.2%}")
        
        if success_rate < 0.95:
            batch_logger.warning(f"Success rate below threshold in batch segment {i+1}")
            
    total_time = random.uniform(5, 15)
    batch_logger.info(f"Batch processing completed in {total_time:.2f} seconds")

def simulate_quality_learning():
    """Simulate quality learning system activity."""
    quality_logger.info("Starting quality learning analysis")
    
    metrics = {
        "embedding_similarity": random.uniform(0.7, 0.95),
        "groq_quality_score": random.uniform(7.0, 9.5),
        "agreement_rate": random.uniform(0.8, 0.98)
    }
    
    quality_logger.info(f"Initial metrics: {metrics}")
    
    # Simulate learning iterations
    for i in range(3):
        quality_logger.debug(f"Running learning iteration {i+1}")
        
        # Update metrics
        metrics["embedding_similarity"] += random.uniform(-0.05, 0.05)
        metrics["groq_quality_score"] += random.uniform(-0.3, 0.3)
        metrics["agreement_rate"] += random.uniform(-0.02, 0.04)
        
        quality_logger.info(f"Updated metrics after iteration {i+1}: {metrics}")
    
    if metrics["agreement_rate"] < 0.85:
        quality_logger.warning(f"Agreement rate is concerning: {metrics['agreement_rate']:.2%}")
        
    quality_logger.info("Quality learning analysis completed")

def simulate_api_calls():
    """Simulate API client activity."""
    api_logger.info("Initializing API client")
    
    # Simulate API calls
    for i in range(3):
        endpoint = random.choice(["translate", "analyze", "evaluate", "compare"])
        latency = random.uniform(0.1, 2.0)
        api_logger.debug(f"Calling {endpoint} API endpoint")
        
        # Simulate occasional API errors
        if random.random() < 0.1:
            api_logger.error(f"API call to {endpoint} failed with timeout after {latency:.2f}s")
        else:
            api_logger.info(f"API call to {endpoint} completed in {latency:.2f}s")
    
    api_logger.info("API client operations completed")

def main():
    """Main function demonstrating logging across different components."""
    system_logger.info("=== Starting logging demonstration ===")
    
    # Demonstrate system logging
    system_logger.info("Simulating system activity...")
    simulate_system_activity()
    
    # Demonstrate batch processing logging
    system_logger.info("Simulating batch processing...")
    simulate_batch_processing()
    
    # Demonstrate quality learning logging
    system_logger.info("Simulating quality learning...")
    simulate_quality_learning()
    
    # Demonstrate API client logging
    system_logger.info("Simulating API calls...")
    simulate_api_calls()
    
    # Show available log files
    log_files = LoggerConfig.get_available_log_files()
    system_logger.info(f"Log files created: {len(log_files)}")
    for log_name, log_path in list(log_files.items())[:5]:  # Show first 5 only
        system_logger.info(f"  - {log_name}: {log_path}")
    
    system_logger.info("=== Logging demonstration completed ===")

if __name__ == "__main__":
    main() 