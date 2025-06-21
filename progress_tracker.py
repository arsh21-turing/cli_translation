"""
Progress tracking module for batch operations.
"""
import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProgressTracker")

class ProgressTracker:
    """
    Tracks and reports progress for batch operations.
    """
    
    def __init__(self, total_items: int = 0, log_file: Optional[str] = None):
        """
        Initialize the progress tracker.
        
        Args:
            total_items: Total number of items to process
            log_file: Path to log file (optional)
        """
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items: Dict[str, str] = {}  # item_id: error
        self.completed_with_metadata: Dict[str, Any] = {}  # item_id: metadata
        
        # Timing information
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.latest_status: Optional[str] = None
        
        # Set up logging
        self.log_file = log_file
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Add file handler if log file specified
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
    
    def start(self, total_items: Optional[int] = None) -> None:
        """
        Start or restart the progress tracking.
        
        Args:
            total_items: Optional new total items count
        """
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        if total_items is not None:
            self.total_items = total_items
            
        self.completed_items = 0
        self.failed_items = {}
        self.completed_with_metadata = {}
        
        logger.info(f"Progress tracking started with {self.total_items} total items")
    
    def update(self, count: int = 1, status: Optional[str] = None) -> None:
        """
        Update progress count and status.
        
        Args:
            count: New completed count (not incremental)
            status: Optional status message
        """
        self.completed_items = count
        self.last_update_time = time.time()
        
        if status:
            self.latest_status = status
            logger.info(f"Progress updated: {count}/{self.total_items} items ({self.get_progress_percentage():.1f}%) - {status}")
        else:
            logger.info(f"Progress updated: {count}/{self.total_items} items ({self.get_progress_percentage():.1f}%)")
    
    def increment(self, status: Optional[str] = None) -> None:
        """
        Increment the completed count by 1.
        
        Args:
            status: Optional status message
        """
        self.update(self.completed_items + 1, status)
    
    def mark_complete(self, item_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark specific item as complete.
        
        Args:
            item_id: Unique identifier for the item
            metadata: Optional metadata about the completion
        """
        self.increment(f"Completed item: {item_id}")
        
        if metadata:
            self.completed_with_metadata[item_id] = metadata
    
    def mark_failed(self, item_id: str, error: Optional[str] = None) -> None:
        """
        Mark specific item as failed.
        
        Args:
            item_id: Unique identifier for the item
            error: Optional error information
        """
        self.failed_items[item_id] = error or "Unknown error"
        logger.warning(f"Item failed: {item_id} - Error: {error}")
    
    def get_progress_percentage(self) -> float:
        """
        Get progress as percentage.
        
        Returns:
            Progress percentage between 0 and 100
        """
        if self.total_items == 0:
            return 100.0
        
        return (self.completed_items / self.total_items) * 100.0
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """
        Get detailed progress statistics.
        
        Returns:
            Dictionary with progress statistics
        """
        elapsed_time = 0
        remaining_time = None
        
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            
            # Calculate estimated remaining time
            if self.completed_items > 0 and self.completed_items < self.total_items:
                items_per_second = self.completed_items / elapsed_time
                remaining_items = self.total_items - self.completed_items
                remaining_time = remaining_items / items_per_second if items_per_second > 0 else 0
        
        return {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items_count": len(self.failed_items),
            "progress_percentage": self.get_progress_percentage(),
            "elapsed_time_seconds": elapsed_time,
            "estimated_remaining_time_seconds": remaining_time,
            "items_per_second": (self.completed_items / elapsed_time) if elapsed_time > 0 else 0,
            "latest_status": self.latest_status
        }
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """
        Estimate remaining time in seconds.
        
        Returns:
            Estimated seconds remaining or None if unknown
        """
        if not self.start_time or self.completed_items == 0:
            return None
            
        elapsed_time = time.time() - self.start_time
        items_per_second = self.completed_items / elapsed_time
        remaining_items = self.total_items - self.completed_items
        
        if items_per_second > 0:
            return remaining_items / items_per_second
        return None
    
    def get_failed_items(self) -> Dict[str, str]:
        """
        Get list of failed items.
        
        Returns:
            Dictionary of failed items with errors
        """
        return self.failed_items
    
    def reset(self) -> None:
        """Reset the progress tracker."""
        self.total_items = 0
        self.completed_items = 0
        self.failed_items = {}
        self.start_time = None
        self.last_update_time = None
        self.latest_status = None
        logger.info("Progress tracker reset")
    
    def log_progress(self) -> None:
        """Log the current progress."""
        stats = self.get_progress_stats()
        elapsed_str = f"{stats['elapsed_time_seconds']:.1f}s" if stats['elapsed_time_seconds'] is not None else "unknown"
        remaining_str = f"{stats['estimated_remaining_time_seconds']:.1f}s" if stats['estimated_remaining_time_seconds'] is not None else "unknown"
        
        logger.info(
            f"Progress: {stats['completed_items']}/{stats['total_items']} "
            f"({stats['progress_percentage']:.1f}%) - "
            f"Elapsed: {elapsed_str}, Remaining: {remaining_str}"
        )
    
    def create_progress_report(self) -> Dict[str, Any]:
        """
        Create a detailed progress report.
        
        Returns:
            Dictionary with comprehensive progress information
        """
        stats = self.get_progress_stats()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_stats": stats,
            "failed_items": self.failed_items,
            "completed_count": self.completed_items,
            "failed_count": len(self.failed_items),
            "total_items": self.total_items,
            "detailed_items": len(self.completed_with_metadata),
        }
        
        # Only include metadata if it's not too large
        if len(self.completed_with_metadata) <= 1000:
            report["completed_with_metadata"] = self.completed_with_metadata
        
        return report 