"""
Simple batch processor example that works with existing codebase components.
"""
import os
import argparse
import logging
import time
import json
from typing import Dict, Any

from batch_processor import BatchProcessor
from progress_tracker import ProgressTracker
from error_handler import ErrorHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/batch_processing/simple_example.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SimpleBatchProcessor")

def simple_text_processor(file_path: str) -> Dict[str, Any]:
    """
    Simple processor function that doesn't depend on complex components.
    Performs basic text analysis without embedding or AI evaluation.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing file: {file_path}")
    
    # Simulate occasional errors (for testing error handling)
    if os.path.basename(file_path).startswith('error_'):
        raise ValueError(f"Simulated error for file {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Skip empty files
        if not content:
            return {
                "file_path": file_path,
                "status": "skipped",
                "reason": "empty_file",
                "processed_timestamp": time.time()
            }
        
        # Parse file content - expect format: "source_text|||translation_text"
        if '|||' in content:
            source_text, translation_text = content.split('|||', 1)
            source_text = source_text.strip()
            translation_text = translation_text.strip()
        else:
            # If no separator, treat whole content as source text
            source_text = content
            translation_text = content  # For analysis
        
        # Basic text analysis
        source_words = source_text.split()
        translation_words = translation_text.split()
        
        # Simple similarity metric based on common words
        source_word_set = set(word.lower() for word in source_words)
        translation_word_set = set(word.lower() for word in translation_words)
        
        if source_word_set or translation_word_set:
            common_words = source_word_set.intersection(translation_word_set)
            all_words = source_word_set.union(translation_word_set)
            similarity_score = len(common_words) / len(all_words) if all_words else 0.0
        else:
            similarity_score = 0.0
        
        # Return comprehensive analysis results
        return {
            "file_path": file_path,
            "source_text": source_text[:100] + "..." if len(source_text) > 100 else source_text,
            "translation_text": translation_text[:100] + "..." if len(translation_text) > 100 else translation_text,
            "source_word_count": len(source_words),
            "translation_word_count": len(translation_words),
            "source_character_count": len(source_text),
            "translation_character_count": len(translation_text),
            "basic_similarity": similarity_score,
            "word_ratio": len(translation_words) / len(source_words) if source_words else 0.0,
            "status": "completed",
            "processed_timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        # Return error information
        return {
            "file_path": file_path,
            "status": "error",
            "error_message": str(e),
            "processed_timestamp": time.time()
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple batch processing for text files')
    parser.add_argument('--input', required=True, help='Input directory containing files to process')
    parser.add_argument('--output', required=True, help='Output directory for processed files')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of files to process in each batch')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--clean', action='store_true', help='Clear existing checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    args = parser.parse_args()
    
    # Set up the batch processor
    error_handler = ErrorHandler(log_file="logs/batch_processing/simple_errors.log")
    progress_tracker = ProgressTracker(log_file="logs/batch_processing/simple_progress.log")
    
    batch_processor = BatchProcessor(
        input_dir=args.input,
        output_dir=args.output,
        processor_func=simple_text_processor,
        batch_size=args.batch_size,
        max_workers=args.workers,
        error_handler=error_handler,
        progress_tracker=progress_tracker
    )
    
    # Clear checkpoints if requested
    if args.clean:
        batch_processor.clear_checkpoints()
        logger.info("Cleared checkpoints for fresh start")
    
    # Check if resumable
    if batch_processor.is_resumable() and not args.clean:
        logger.info("Found existing checkpoint. Will resume processing...")
    
    # Run the batch processor
    logger.info(f"Starting simple batch processing with batch size {args.batch_size} and {args.workers} workers")
    start_time = time.time()
    stats = batch_processor.process_directory()
    duration = time.time() - start_time
    
    # Log completion stats
    logger.info(f"Processing completed in {duration:.2f} seconds")
    logger.info(f"Processed {stats['processed_files']} files: {stats['successful_files']} success, {stats['failed_files']} failed")
    
    # Get error summary
    error_summary = error_handler.get_error_summary()
    if error_summary['total_errors'] > 0:
        logger.warning(f"Encountered {error_summary['total_errors']} errors")
        for error_type, count in error_summary['error_types'].items():
            logger.warning(f"  {error_type}: {count} occurrences")
        
        # Show recoverable vs unrecoverable errors
        recoverable = error_summary.get('recoverable_errors', 0)
        unrecoverable = error_summary.get('unrecoverable_errors', 0)
        logger.info(f"Error breakdown: {recoverable} recoverable, {unrecoverable} unrecoverable")
    else:
        logger.info("No errors encountered during processing")
    
    # Get progress summary
    progress_stats = progress_tracker.get_progress_stats()
    logger.info(f"Final progress: {progress_stats['progress_percentage']:.1f}% completed")
    if progress_stats['items_per_second'] > 0:
        logger.info(f"Average processing rate: {progress_stats['items_per_second']:.2f} files/second")
    
    # Show checkpoint information
    if batch_processor.is_resumable():
        logger.info(f"Checkpoint saved for future resumption")

if __name__ == '__main__':
    main() 