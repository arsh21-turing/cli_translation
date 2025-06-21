"""
Error handling module for batch processing.
"""
import os
import time
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ErrorHandler")

class ErrorHandler:
    """
    Manages errors during batch processing with retry capabilities
    and detailed error tracking.
    """
    
    # Define which errors are potentially recoverable
    RECOVERABLE_ERRORS = (
        TimeoutError,
        ConnectionError, 
        ConnectionRefusedError,
        ConnectionResetError,
        OSError,  # Some file access errors might be recoverable
    )
    
    def __init__(
        self, 
        log_file: Optional[str] = None, 
        max_retries: int = 3, 
        retry_interval: int = 5,
        error_dir: Optional[str] = None
    ):
        """
        Initialize the error handler.
        
        Args:
            log_file: Path to error log file
            max_retries: Maximum number of retry attempts for recoverable errors
            retry_interval: Base interval (in seconds) between retry attempts
            error_dir: Directory to save error details and failed items
        """
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_details: List[Dict[str, Any]] = []
        
        # Set up log file
        self.log_file = log_file
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Add file handler if log file specified
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        
        # Set up error directory
        if error_dir:
            self.error_dir = Path(error_dir)
            self.error_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.error_dir = Path("logs/batch_processing/errors")
            self.error_dir.mkdir(parents=True, exist_ok=True)
    
    def handle_error(self, error: Exception, context: Dict[str, Any], file_path: Optional[str] = None) -> None:
        """
        Handle an error with context information.
        
        Args:
            error: Exception that was raised
            context: Dictionary with contextual information about the error
            file_path: Optional path to the file that caused the error
        """
        error_type = self.categorize_error(error)
        self.error_counts[error_type] += 1
        
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Create detailed error entry
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "stack_trace": stack_trace,
            "context": context,
            "file_path": file_path
        }
        
        self.error_details.append(error_details)
        
        # Log the error
        self.log_error(error, context)
        
        # Save error details to file if file_path is provided
        if file_path:
            self._save_error_details(error_details, file_path)
    
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Log error with context information.
        
        Args:
            error: Exception that was raised
            context: Dictionary with contextual information
        """
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.error(f"Error: {type(error).__name__}: {str(error)} - Context: {context_str}")
    
    def should_retry(self, error_type: str, retry_count: int) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            error_type: Category of the error
            retry_count: Current retry attempt count
            
        Returns:
            True if operation should be retried, False otherwise
        """
        # Check if we haven't exceeded max retries
        if retry_count >= self.max_retries:
            return False
            
        # Check if error type is considered recoverable
        try:
            error_class = eval(error_type) if error_type in globals() else None
            if error_class and issubclass(error_class, self.RECOVERABLE_ERRORS):
                return True
        except:
            pass
            
        # Special types that we know are recoverable
        recoverable_types = [
            "TimeoutError", "ConnectionError", "ConnectionRefusedError",
            "ConnectionResetError", "OSError", "IOError", "RequestException"
        ]
        
        return any(rtype in error_type for rtype in recoverable_types)
    
    def retry_operation(
        self, 
        operation: Callable, 
        args: Optional[Tuple] = None, 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Function to retry
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (success, result/error)
        """
        args = args or ()
        kwargs = kwargs or {}
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Attempt operation
                result = operation(*args, **kwargs)
                
                # If successful, return the result
                if retry_count > 0:
                    logger.info(f"Operation succeeded on retry attempt {retry_count}")
                
                return True, result
                
            except Exception as e:
                last_error = e
                error_type = self.categorize_error(e)
                retry_count += 1
                
                # Log the error
                if retry_count <= self.max_retries:
                    # Calculate backoff time with exponential backoff & small jitter
                    backoff = self.retry_interval * (2 ** (retry_count - 1))
                    jitter = backoff * 0.1 * (1 - 2 * (time.time() % 1))  # 10% jitter
                    wait_time = backoff + jitter
                    
                    logger.warning(
                        f"Retry attempt {retry_count}/{self.max_retries} after error: "
                        f"{type(e).__name__}: {str(e)} - Waiting {wait_time:.1f}s"
                    )
                    
                    # Wait before retrying
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Operation failed after {self.max_retries} retries. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    )
        
        # All retries failed
        return False, last_error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors encountered.
        
        Returns:
            Dictionary with error summary
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "error_count_by_type": dict(self.error_counts),
            "recoverable_errors": sum(1 for error in self.error_details 
                                     if self.should_retry(error["error_type"], 0)),
            "unrecoverable_errors": sum(1 for error in self.error_details 
                                       if not self.should_retry(error["error_type"], 0)),
            "first_error": self.error_details[0] if self.error_details else None,
            "latest_error": self.error_details[-1] if self.error_details else None
        }
    
    def categorize_error(self, error: Exception) -> str:
        """
        Categorize error by type.
        
        Args:
            error: Exception to categorize
            
        Returns:
            String representation of error category
        """
        return type(error).__name__
    
    def save_failed_item(self, item: Any, error: Exception) -> None:
        """
        Save failed item for later processing.
        
        Args:
            item: The item that failed processing
            error: Exception that caused the failure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_type = self.categorize_error(error)
        
        # Create filename for the error
        error_file = self.error_dir / f"failed_item_{timestamp}_{error_type}.json"
        
        # Save error and item to file
        data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "item": item if not isinstance(item, (Path, str)) else str(item)
        }
        
        with open(error_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Failed item saved to {error_file}")
    
    def _save_error_details(self, error_details: Dict[str, Any], file_path: str) -> None:
        """
        Save detailed error information to a file.
        
        Args:
            error_details: Dictionary with error details
            file_path: Path to the file that caused the error
        """
        # Create a filename based on the original file and timestamp
        original_name = Path(file_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.error_dir / f"{original_name}.{timestamp}.error.json"
        
        # Save error details
        with open(error_file, 'w') as f:
            json.dump(error_details, f, indent=2, default=str)
        
        logger.info(f"Error details saved to {error_file}")
    
    def get_recoverable_items(self) -> List[Dict[str, Any]]:
        """
        Get list of potentially recoverable items.
        
        Returns:
            List of error details for recoverable errors
        """
        return [
            error for error in self.error_details
            if self.should_retry(error["error_type"], 0)
        ]
    
    def get_unrecoverable_items(self) -> List[Dict[str, Any]]:
        """
        Get list of unrecoverable items.
        
        Returns:
            List of error details for unrecoverable errors
        """
        return [
            error for error in self.error_details
            if not self.should_retry(error["error_type"], 0)
        ] 