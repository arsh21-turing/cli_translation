"""
Enhanced unified logging configuration for the translation evaluator project.
Features:
- Unified configuration across all modules
- Category-based log organization
- Asynchronous logging
- Structured JSON logging format
- Performance metrics tracking
- PII data sanitization
- Memory handlers for testing
- Runtime log level adjustment
"""
import os
import sys
import re
import time
import json
import uuid
import queue
import atexit
import logging
import threading
import traceback
import logging.config
import logging.handlers
from typing import Dict, Any, Optional, List, Union, Set, Pattern, Callable
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
import socket
import hashlib
import inspect
import platform

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Default logging levels
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG

# Default log filename format
DEFAULT_LOG_FILENAME_FORMAT = "%Y-%m-%d_%H-%M-%S.log"

# Async logging constants
ASYNC_QUEUE_SIZE = 10000
ASYNC_WORKER_INTERVAL = 0.1  # seconds

# Performance monitoring constants
SLOW_EXECUTION_THRESHOLD = 0.5  # seconds

# PII patterns to sanitize (email, phone, IP, credit card, etc.)
# Note: These are simplified patterns and might need improvement for production use
PII_PATTERNS = {
    'email': re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
    'phone': re.compile(r'(\+\d{1,3}[\s-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'),
    'ip': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    'credit_card': re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
    'ssn': re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
    'address': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)\b', re.IGNORECASE),
}

# Define log categories and their subdirectories
LOG_CATEGORIES = {
    "performance": "performance",
    "batch_processing": "batch_processing",
    "quality_learning": "quality_learning",
    "disagreements": "disagreements",
    "api": "api",
    "system": "system"
}

# Default formatter configuration
DEFAULT_FORMATTERS = {
    'detailed': {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    },
    'simple': {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    },
    'minimal': {
        'format': '%(levelname)s - %(message)s'
    }
}

# In-memory handler for tests
class MemoryHandler(logging.Handler):
    """Custom handler that keeps log records in memory for testing."""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))

    def get_records(self):
        return self.records

    def clear(self):
        self.records = []

# Global memory handler for tests
memory_handler = MemoryHandler()

class AsyncQueueHandler(logging.Handler):
    """
    Handler that places log records on a queue for processing by a separate thread.
    """
    def __init__(self, handlers: List[logging.Handler]):
        """
        Initialize with handlers that will process the queued records.
        
        Args:
            handlers: List of handlers to process the records
        """
        super().__init__()
        self.handlers = handlers
        self.queue = queue.Queue(ASYNC_QUEUE_SIZE)
        self.thread = None
        self.stop_event = threading.Event()
        self._start_worker()
        
        # Register shutdown handler to ensure logs are processed on exit
        atexit.register(self._shutdown)

    def emit(self, record):
        """
        Place record in the queue for asynchronous processing.
        
        Args:
            record: Log record to process
        """
        try:
            # Add current time to record for latency tracking
            record.enqueue_time = time.time()
            
            # Place in queue if not full
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                # If queue is full, log a warning and drop the record
                sys.stderr.write("Async logging queue is full, dropping log record\n")
                sys.stderr.flush()
        except Exception:
            self.handleError(record)

    def _consume_queue(self):
        """Process records from the queue in a separate thread."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Get record from queue, waiting up to ASYNC_WORKER_INTERVAL seconds
                try:
                    record = self.queue.get(timeout=ASYNC_WORKER_INTERVAL)
                except queue.Empty:
                    continue
                
                # Process record with each handler
                for handler in self.handlers:
                    if record.levelno >= handler.level:
                        # Add latency information to record
                        record.log_latency = time.time() - getattr(record, 'enqueue_time', time.time())
                        handler.handle(record)
                
                # Mark as done
                self.queue.task_done()
            except Exception:
                # Log any exceptions to stderr
                traceback.print_exc(file=sys.stderr)
                
        # Process any remaining records
        while not self.queue.empty():
            try:
                record = self.queue.get_nowait()
                for handler in self.handlers:
                    if record.levelno >= handler.level:
                        record.log_latency = time.time() - getattr(record, 'enqueue_time', time.time())
                        handler.handle(record)
                self.queue.task_done()
            except queue.Empty:
                break
            except Exception:
                traceback.print_exc(file=sys.stderr)

    def _start_worker(self):
        """Start the worker thread."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._consume_queue, daemon=True)
            self.thread.start()

    def _shutdown(self):
        """Ensure all logs are processed when application exits."""
        # Signal thread to stop
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            
            # Wait for queue to empty with a timeout
            self.thread.join(timeout=5)  # Wait for up to 5 seconds for the thread to finish

            # Process any remaining records
            while not self.queue.empty():
                try:
                    record = self.queue.get_nowait()
                    for handler in self.handlers:
                        if record.levelno >= handler.level:
                            record.log_latency = time.time() - getattr(record, 'enqueue_time', time.time())
                            if handler.stream and not handler.stream.closed:
                                handler.handle(record)
                    self.queue.task_done()
                except queue.Empty:
                    break
                except Exception:
                    traceback.print_exc(file=sys.stderr)

    def close(self):
        """Close handler and process any remaining logs."""
        self._shutdown()
        super().close()

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    """
    def __init__(self, include_extra_fields: bool = True, sanitize_pii: bool = True):
        """
        Initialize JSON formatter.
        
        Args:
            include_extra_fields: Whether to include extra fields added to LogRecord
            sanitize_pii: Whether to sanitize PII data in log messages
        """
        super().__init__()
        self.include_extra_fields = include_extra_fields
        self.sanitize_pii = sanitize_pii
        
        # System info for context
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
    
    def format(self, record):
        """
        Format the specified record as structured JSON.
        
        Args:
            record: LogRecord to format
            
        Returns:
            Formatted JSON string
        """
        # Create base log object
        log_object = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'path': record.pathname,
            'line': record.lineno,
            'function': record.funcName,
            'process': self.pid,
            'thread': record.thread,
            'thread_name': record.threadName,
            'hostname': self.hostname,
        }
        
        # Include exception info if present
        if record.exc_info:
            log_object['exception'] = {
                'type': str(record.exc_info[0].__name__),
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
        
        # Add log latency if available
        if hasattr(record, 'log_latency'):
            log_object['log_latency_ms'] = record.log_latency * 1000
            
        # Add performance metrics if available
        if hasattr(record, 'execution_time'):
            log_object['execution_time_ms'] = record.execution_time * 1000
            
        if hasattr(record, 'memory_usage'):
            log_object['memory_usage_mb'] = record.memory_usage / (1024 * 1024)  # Convert to MB
        
        # Add custom fields from record
        if self.include_extra_fields:
            for key, value in record.__dict__.items():
                if key not in log_object and not key.startswith('_') and isinstance(value, (str, int, float, bool, list, dict)):
                    if isinstance(value, (list, dict)):
                        # Convert complex structures to JSON-serializable form
                        try:
                            json.dumps(value)
                            log_object[key] = value
                        except (TypeError, OverflowError):
                            log_object[key] = str(value)
                    else:
                        log_object[key] = value
        
        # Sanitize PII if enabled
        if self.sanitize_pii:
            log_object = self._sanitize_pii(log_object)
            
        # Serialize to JSON
        return json.dumps(log_object)
    
    def _sanitize_pii(self, log_object: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize PII data in the log object.
        
        Args:
            log_object: Log data dictionary
            
        Returns:
            Sanitized log data dictionary
        """
        result = {}
        
        for key, value in log_object.items():
            if isinstance(value, str):
                # Sanitize strings
                result[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                # Recursively sanitize dictionaries
                result[key] = self._sanitize_pii(value)
            elif isinstance(value, list):
                # Sanitize list items
                result[key] = [
                    self._sanitize_pii(item) if isinstance(item, dict)
                    else self._sanitize_string(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                # Keep non-string values as is
                result[key] = value
                
        return result
    
    def _sanitize_string(self, text: str) -> str:
        """
        Sanitize a string by replacing PII patterns with hash values.
        
        Args:
            text: String to sanitize
            
        Returns:
            Sanitized string
        """
        if not text:
            return text
            
        sanitized = text
        # Skip sanitization for extremely large strings to avoid MemoryError
        if len(sanitized) > 10000:
            return sanitized
        try:
            for pattern_name, pattern in PII_PATTERNS.items():
                # Find all matches
                matches = pattern.findall(sanitized)
                # Replace each match with a hash
                for match in matches:
                    if isinstance(match, tuple):
                        match = ''.join(match)
                    hash_value = hashlib.sha256(match.encode()).hexdigest()[:8]
                    replacement = f"[{pattern_name.upper()}_REDACTED_{hash_value}]"
                    sanitized = sanitized.replace(match, replacement)
        except MemoryError:
            # If memory issues arise during sanitization, fall back to original text
            return text
        return sanitized

class PerformanceFilter(logging.Filter):
    """
    Filter that adds performance metrics to log records.
    """
    def __init__(self):
        super().__init__()
        
    def filter(self, record):
        """
        Add performance metrics to the log record.
        
        Args:
            record: LogRecord to enhance
            
        Returns:
            Always True (the record is always processed)
        """
        # Add memory usage if not already present
        if not hasattr(record, 'memory_usage'):
            try:
                import psutil
                process = psutil.Process(os.getpid())
                record.memory_usage = process.memory_info().rss
            except (ImportError, AttributeError):
                # If psutil is not available, don't add memory info
                pass
                
        return True

class LoggerConfig:
    """
    Enhanced unified logging configuration manager for the translation evaluator project.
    
    This class provides a centralized logging configuration that can be used
    across all modules in the project to ensure consistent logging behavior.
    
    Features:
    - Unified configuration across all modules
    - Category-based log organization
    - Asynchronous logging for performance
    - Structured JSON logging format
    - Performance metrics tracking
    - PII data sanitization
    - Memory handlers for testing
    - Runtime log level adjustment
    """
    
    _instance = None
    _initialized = False
    _context_stack = threading.local()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LoggerConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        log_dir: str = DEFAULT_LOG_DIR, 
        log_level: int = DEFAULT_LOG_LEVEL,
        console_level: int = DEFAULT_CONSOLE_LEVEL,
        file_level: int = DEFAULT_FILE_LEVEL,
        enable_console: bool = True,
        enable_json: bool = True,
        enable_async: bool = True,
        sanitize_pii: bool = True,
        performance_tracking: bool = True,
        config_file: Optional[str] = None
    ):
        """
        Initialize the enhanced logging configuration.
        
        Args:
            log_dir: Directory for log files (default: 'logs')
            log_level: Overall logging level (default: logging.INFO)
            console_level: Console logging level (default: logging.INFO)
            file_level: File logging level (default: logging.DEBUG)
            enable_console: Whether to enable console logging (default: True)
            enable_json: Whether to use JSON format for file logs (default: True)
            enable_async: Whether to use async logging (default: True)
            sanitize_pii: Whether to sanitize PII data (default: True)
            performance_tracking: Whether to track performance metrics (default: True)
            config_file: Optional path to a JSON logging config file
        """
        # Only initialize once (singleton)
        if LoggerConfig._initialized:
            return
            
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.console_level = console_level
        self.file_level = file_level
        self.enable_console = enable_console
        self.enable_json = enable_json
        self.enable_async = enable_async
        self.sanitize_pii = sanitize_pii
        self.performance_tracking = performance_tracking
        self.config_file = config_file
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories
        for category in LOG_CATEGORIES.values():
            category_dir = self.log_dir / category
            category_dir.mkdir(exist_ok=True)
            
        # Track custom handlers
        self.async_handler = None
        self.custom_filters = {}
        
        # Generate run ID for this logging session
        self.run_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Initialize context stack
        if not hasattr(self._context_stack, 'stack'):
            self._context_stack.stack = []
        
        # Load external config file if provided
        if config_file and os.path.exists(config_file):
            self._load_config_from_file(config_file)
        else:
            # Use default configuration
            self._configure_default_logging()
            
        LoggerConfig._initialized = True
        
        # Log startup information
        system_logger = self.get_logger("LoggerConfig", "system")
        system_logger.info(f"Logging initialized: run_id={self.run_id}, " 
                         f"session_start={self.start_time.isoformat()}, "
                         f"hostname={socket.gethostname()}, "
                         f"pid={os.getpid()}, "
                         f"python={platform.python_version()}")
    
    def _load_config_from_file(self, config_file: str) -> None:
        """
        Load logging configuration from a JSON file.
        
        Args:
            config_file: Path to the JSON config file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Process paths in config to ensure they exist
            if 'handlers' in config:
                for handler_name, handler_config in config['handlers'].items():
                    if 'filename' in handler_config:
                        # Make path absolute if it's not
                        if not os.path.isabs(handler_config['filename']):
                            handler_config['filename'] = os.path.join(
                                str(self.log_dir), handler_config['filename']
                            )
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(handler_config['filename']), exist_ok=True)
            
            # Apply the configuration
            logging.config.dictConfig(config)
            print(f"Loaded logging configuration from {config_file}")
            
            # Add async handler if enabled
            if self.enable_async:
                self._setup_async_handler()
                
        except Exception as e:
            print(f"Error loading logging config from {config_file}: {str(e)}")
            # Fall back to default configuration
            self._configure_default_logging()
    
    def _configure_default_logging(self) -> None:
        """Configure default enhanced logging setup."""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        formatters = {}
        for name, config in DEFAULT_FORMATTERS.items():
            formatters[name] = logging.Formatter(config['format'], config.get('datefmt'))
            
        # Create JSON formatter if enabled
        if self.enable_json:
            json_formatter = JsonFormatter(sanitize_pii=self.sanitize_pii)
        
        handlers = []
        
        # Add console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.CRITICAL)
            console_handler.setFormatter(formatters['simple'])
            handlers.append(console_handler)
        
        # Add default system log file handler
        system_log_dir = self.log_dir / LOG_CATEGORIES['system']
        timestamp = datetime.now().strftime(DEFAULT_LOG_FILENAME_FORMAT)
        system_log_file = system_log_dir / f"system_{timestamp}"
        
        file_handler = logging.FileHandler(system_log_file)
        file_handler.setLevel(self.file_level)
        
        # Use JSON formatter for file handler if enabled
        if self.enable_json:
            file_handler.setFormatter(json_formatter)
        else:
            file_handler.setFormatter(formatters['detailed'])
            
        handlers.append(file_handler)
        
        # Add performance filter if tracking enabled
        if self.performance_tracking:
            perf_filter = PerformanceFilter()
            for handler in handlers:
                handler.addFilter(perf_filter)
            self.custom_filters['performance'] = perf_filter
        
        # Add memory handler for tests
        if memory_handler not in handlers:
            memory_handler.setLevel(logging.DEBUG)
            memory_handler.setFormatter(formatters['detailed'])
            handlers.append(memory_handler)
        
        # Set up async handler if enabled
        if self.enable_async:
            self.async_handler = AsyncQueueHandler(handlers)
            root_logger.addHandler(self.async_handler)
        else:
            # Otherwise, add handlers directly
            for handler in handlers:
                root_logger.addHandler(handler)
        
        # Apply the configuration
        logging.captureWarnings(True)
    
    def _setup_async_handler(self) -> None:
        """Set up asynchronous logging handler."""
        # Get existing handlers
        root_logger = logging.getLogger()
        existing_handlers = root_logger.handlers[:]
        
        # Remove existing handlers
        for handler in existing_handlers:
            if handler is not self.async_handler:  # Don't remove existing async handler
                root_logger.removeHandler(handler)
        
        # Create async handler
        self.async_handler = AsyncQueueHandler(existing_handlers)
        root_logger.addHandler(self.async_handler)
    
    def get_logger(self, name: str, category: Optional[str] = None) -> logging.Logger:
        """
        Get a logger with the specified name and category.
        
        Args:
            name: Name of the logger (usually __name__)
            category: Optional category for specialized logging
            
        Returns:
            Configured Logger instance
        """
        logger = logging.getLogger(name)
        
        # Set minimum level
        if logger.level == 0:  # Default level is NOTSET (0)
            logger.setLevel(self.log_level)
        
        # Add run_id as an attribute to the logger
        logger.run_id = self.run_id
        
        # If a category is specified, add a file handler for that category
        if category and category in LOG_CATEGORIES:
            # Check if this logger already has a handler for this category
            if not any(getattr(h, '_category', None) == category for h in logger.handlers):
                # Create category-specific log file
                category_dir = self.log_dir / LOG_CATEGORIES[category]
                timestamp = datetime.now().strftime(DEFAULT_LOG_FILENAME_FORMAT)
                category_log_file = category_dir / f"{name.split('.')[-1]}_{timestamp}"
                
                # Create file handler
                file_handler = logging.FileHandler(category_log_file)
                file_handler.setLevel(self.file_level)
                
                # Use JSON formatter if enabled
                if self.enable_json:
                    file_handler.setFormatter(JsonFormatter(sanitize_pii=self.sanitize_pii))
                else:
                    file_handler.setFormatter(logging.Formatter(DEFAULT_FORMATTERS['detailed']['format']))
                
                # Mark this handler with its category
                file_handler._category = category
                
                # Add performance filter if enabled
                if self.performance_tracking and 'performance' in self.custom_filters:
                    file_handler.addFilter(self.custom_filters['performance'])
                
                # Add to logger or async handler
                if self.enable_async and self.async_handler:
                    # Add to async handler's list
                    self.async_handler.handlers.append(file_handler)
                else:
                    # Add directly to logger
                    logger.addHandler(file_handler)
        
        return logger
    
    def update_levels(
        self, 
        log_level: Optional[int] = None, 
        console_level: Optional[int] = None, 
        file_level: Optional[int] = None
    ) -> None:
        """
        Update logging levels for existing handlers.
        
        Args:
            log_level: New overall logging level (or None to leave unchanged)
            console_level: New console logging level (or None to leave unchanged)
            file_level: New file logging level (or None to leave unchanged)
        """
        # Update instance variables
        if log_level is not None:
            self.log_level = log_level
            logging.getLogger().setLevel(log_level)
            
        # Update console handlers
        if console_level is not None:
            self.console_level = console_level
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(console_level)
            
            # Update console handlers in async handler if enabled
            if self.async_handler:
                for handler in self.async_handler.handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        handler.setLevel(console_level)
        
        # Update file handlers
        if file_level is not None:
            self.file_level = file_level
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(file_level)
                    
            # Update file handlers in async handler if enabled
            if self.async_handler:
                for handler in self.async_handler.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.setLevel(file_level)
    
    def add_rotating_file_handler(
        self, 
        logger_name: str, 
        filename: str, 
        max_bytes: int = 10485760,  # 10 MB
        backup_count: int = 5,
        level: int = None,
        use_json: Optional[bool] = None
    ) -> None:
        """
        Add a rotating file handler to a specific logger.
        
        Args:
            logger_name: Name of the logger to add the handler to
            filename: Name of the log file
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            level: Logging level (uses file_level if None)
            use_json: Whether to use JSON formatter (uses global setting if None)
        """
        logger = logging.getLogger(logger_name)
        
        # Create full path for the log file
        if not os.path.isabs(filename):
            filename = os.path.join(str(self.log_dir), filename)
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create handler
        handler = logging.handlers.RotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        # Set level
        if level is None:
            level = self.file_level
        handler.setLevel(level)
        
        # Set formatter
        use_json = self.enable_json if use_json is None else use_json
        if use_json:
            handler.setFormatter(JsonFormatter(sanitize_pii=self.sanitize_pii))
        else:
            detailed_config = DEFAULT_FORMATTERS['detailed']
            handler.setFormatter(logging.Formatter(detailed_config['format'], detailed_config.get('datefmt')))
        
        # Add performance filter if enabled
        if self.performance_tracking and 'performance' in self.custom_filters:
            handler.addFilter(self.custom_filters['performance'])
            
        # Add handler
        if self.enable_async and self.async_handler:
            # Add to async handler's list
            self.async_handler.handlers.append(handler)
        else:
            # Add directly to logger
            logger.addHandler(handler)
    
    def clear_handlers(self, logger_name: Optional[str] = None) -> None:
        """
        Remove all handlers from a specific logger or all loggers.
        
        Args:
            logger_name: Name of the logger to clear (None for all loggers)
        """
        if logger_name:
            # Clear specific logger
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        else:
            # Clear root logger handlers
            root_logger = logging.getLogger()
            
            # Close and remove async handler if present
            if self.async_handler is not None:
                if self.async_handler in root_logger.handlers:
                    root_logger.removeHandler(self.async_handler)
                self.async_handler.close()
                self.async_handler = None
                
            # Remove all handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                handler.close()
                
            # Re-add basic handlers
            self._configure_default_logging()
    
    @staticmethod
    def get_available_log_files(category: Optional[str] = None) -> Dict[str, str]:
        """
        Get a list of available log files.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Dictionary mapping log file names to full paths
        """
        log_files = {}
        log_dir = Path(DEFAULT_LOG_DIR)
        
        if not log_dir.exists():
            return log_files
        
        if category and category in LOG_CATEGORIES:
            # Get logs from specific category
            category_dir = log_dir / LOG_CATEGORIES[category]
            if category_dir.exists():
                for file_path in category_dir.glob("*.log"):
                    log_files[file_path.name] = str(file_path)
        else:
            # Get all logs
            for category_name, category_dir_name in LOG_CATEGORIES.items():
                category_dir = log_dir / category_dir_name
                if category_dir.exists():
                    for file_path in category_dir.glob("*.log"):
                        log_files[f"{category_name}/{file_path.name}"] = str(file_path)
        
        return log_files
    
    def add_context(self, **kwargs) -> None:
        """
        Add context values that will be included in all log records.
        
        Args:
            **kwargs: Key-value pairs to add to the context
        """
        if not hasattr(self._context_stack, 'stack'):
            self._context_stack.stack = []
            
        # Add new context
        self._context_stack.stack.append(kwargs)
    
    def remove_context(self) -> None:
        """Remove the most recently added context from the stack."""
        if hasattr(self._context_stack, 'stack') and self._context_stack.stack:
            self._context_stack.stack.pop()
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context from the stack.
        
        Returns:
            Combined context from all stack entries
        """
        if not hasattr(self._context_stack, 'stack'):
            return {}
            
        # Combine all contexts in the stack (newer entries override older ones)
        context = {}
        for ctx in self._context_stack.stack:
            context.update(ctx)
            
        return context
    
    def performance_log(self, logger: logging.Logger, level: int = logging.INFO) -> Callable:
        """
        Decorator to log function performance metrics.
        
        Args:
            logger: Logger to use for logging
            level: Logging level for performance log
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Capture start time and memory
                start_time = time.time()
                
                try:
                    # Get initial memory usage if available
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_before = process.memory_info().rss
                    except (ImportError, AttributeError):
                        memory_before = 0
                        
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Calculate execution time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Get memory usage change if available
                    try:
                        if memory_before > 0:
                            memory_after = process.memory_info().rss
                            memory_change = memory_after - memory_before
                        else:
                            memory_change = 0
                    except (ImportError, AttributeError, NameError):
                        memory_change = 0
                    
                    # Add context to log message
                    context = {}
                    
                    # Try to get function signature
                    try:
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        
                        # Convert args to safe strings
                        safe_args = {}
                        for arg_name, arg_value in bound_args.arguments.items():
                            if isinstance(arg_value, (str, int, float, bool, type(None))):
                                safe_args[arg_name] = arg_value
                            else:
                                safe_args[arg_name] = type(arg_value).__name__
                                
                        context["function_args"] = safe_args
                    except (ValueError, TypeError):
                        # If we can't inspect the signature
                        pass
                    
                    # Set context for the log record
                    context.update({
                        "execution_time": execution_time,
                        "execution_time_ms": execution_time * 1000,
                        "memory_change_bytes": memory_change,
                        "memory_change_mb": memory_change / (1024 * 1024) if memory_change else 0,
                        "exceeds_threshold": execution_time > SLOW_EXECUTION_THRESHOLD,
                    })
                    
                    # Log with different levels based on execution time
                    log_level = level
                    if execution_time > SLOW_EXECUTION_THRESHOLD:
                        log_level = logging.WARNING
                        
                    # Create log message
                    logger.log(
                        log_level,
                        f"Performance: {func.__module__}.{func.__name__} "
                        f"executed in {execution_time:.4f}s "
                        f"({memory_change / (1024 * 1024):.2f} MB)" if memory_change else "",
                        extra=context
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log exception with performance context
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    logger.error(
                        f"Exception in {func.__module__}.{func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            "execution_time": execution_time,
                            "execution_time_ms": execution_time * 1000,
                            "exception_type": type(e).__name__,
                        }
                    )
                    
                    # Re-raise the exception
                    raise
                    
            return wrapper
        return decorator
    
    def attach_context_processor(self) -> None:
        """
        Attach a context processor to add context to all log records.
        This is called automatically during initialization.
        """
        # Define a filter to add context to all log records
        class ContextFilter(logging.Filter):
            def __init__(self, logger_config):
                super().__init__()
                self.logger_config = logger_config
                
            def filter(self, record):
                # Add context from stack to record
                context = self.logger_config.get_context()
                for key, value in context.items():
                    setattr(record, key, value)
                
                # Add run ID to record
                setattr(record, 'run_id', self.logger_config.run_id)
                
                # Always process the record
                return True
                
        # Create and add the filter
        context_filter = ContextFilter(self)
        self.custom_filters['context'] = context_filter
        
        # Add to root logger
        logging.getLogger().addFilter(context_filter)

# Create default logger configuration
default_logger_config = LoggerConfig()

def get_logger(name: str, category: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger with the enhanced configuration.
    
    Args:
        name: Name of the logger (usually __name__)
        category: Optional category for specialized logging
        
    Returns:
        Configured Logger instance
    """
    return default_logger_config.get_logger(name, category)

def log_performance(level: int = logging.INFO) -> Callable:
    """
    Decorator to log function performance metrics.
    
    Args:
        level: Logging level for performance log
        
    Returns:
        Decorator function
    """
    def decorator(func):
        logger = get_logger(func.__module__)
        return default_logger_config.performance_log(logger, level)(func)
    return decorator

class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        
    def __enter__(self):
        default_logger_config.add_context(**self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        default_logger_config.remove_context()

# Ensure context processor is attached
default_logger_config.attach_context_processor() 