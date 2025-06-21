#!/usr/bin/env python3
"""
Enhanced Async Translation Quality Analyzer
Main entry point for the command-line interface with comprehensive error handling,
asynchronous processing, rich CLI, performance monitoring, and smart caching.

Features:
- Asynchronous processing with asyncio for concurrent operations
- Rich CLI with interactive prompts and visualization
- Smart-cache statistics and analysis
- Performance monitoring with detailed reports
- Rich progress bars and tables
- Comprehensive error handling
- Backward-compatible CLI entry point
"""

import sys
import os
import argparse
import logging
import traceback
import time
import html as _html_mod
import datetime
import asyncio
import functools
import atexit
import signal
import contextlib
import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple, Set, Callable
import textwrap
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.box import Box
from rich.progress import (
    Progress, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeRemainingColumn,
    SpinnerColumn
)
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout

# Check for async libraries
try:
    import aiohttp
    import aiofiles
    HAS_ASYNC_LIBS = True
except ImportError:
    HAS_ASYNC_LIBS = False

# Enhanced logger imports
from logger_config import get_logger, log_performance, LogContext

# Add performance monitoring imports
try:
    from performance_monitor import PerformanceMonitor
    performance_monitoring_available = True
except ImportError:
    performance_monitoring_available = False

# Add smart cache imports for statistics
try:
    from smart_cache import SmartCache, smart_cache
    smart_cache_available = True
except ImportError:
    smart_cache_available = False
    SmartCache = None

# Import async components if available
if HAS_ASYNC_LIBS:
    try:
        from async_groq_client import AsyncGroqClient
        from async_embedding_generator import AsyncEmbeddingGenerator
        from async_groq_evaluator import AsyncGroqEvaluator
        from async_translation_quality_analyzer import AsyncTranslationQualityAnalyzer
        ASYNC_COMPONENTS_AVAILABLE = True
    except ImportError:
        ASYNC_COMPONENTS_AVAILABLE = False
else:
    ASYNC_COMPONENTS_AVAILABLE = False

# Import our components
from config_manager import ConfigManager
from model_loader import ModelLoader, ModelType, InferenceMode, MultilingualModelManager
from text_processor import TextProcessor
from embedding_generator import MultilingualEmbeddingGenerator
from similarity_calculator import SimilarityCalculator
from language_utils import LanguageDetector, EmbeddingBasedLanguageDetector, get_supported_languages
from analyzer import TranslationQualityAnalyzer

# Import sync fallback components
from groq_client import GroqClient
from groq_evaluator import GroqEvaluator
from translation_quality_analyzer import TranslationQualityAnalyzer as SyncTranslationQualityAnalyzer, EmbeddingGenerator
from embedding_cache import EmbeddingCache

# Import additional components
try:
    from batch_processor import BatchProcessor
except ImportError:
    BatchProcessor = None

# Candidate ranking relies on additional utilities
# Import optional dependencies lazily to avoid overhead when not used.
from translation_ranker import calculate_translation_confidence

# Extra serialisation libs (optional)
import yaml  # Added for YAML output support

# -----------------------------------------------------------------------------
# Simple HTML output helper exposed for external import (tests)
# -----------------------------------------------------------------------------


def output_html(
    data: Dict[str, Any],
    title: str = "Translation Report",
    include_diagnostics: bool = False,
) -> str:
    """Return a basic HTML document string for *data*.

    The structure is intentionally simple so it renders in any browser and so
    that automated tests can easily assert on key strings.
    """

    def esc(s: str) -> str:
        import html as _h
        return _h.escape(str(s))

    html_parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>" + esc(title) + "</title></head><body>",
        f"<h1>{esc(title)}</h1>",
    ]

    if "source_text" in data:
        html_parts.append("<h2>Source Text</h2><p>" + esc(data["source_text"]) + "</p>")

    # Ranked translations table
    if data.get("ranked_translations"):
        html_parts.append("<h2>Ranked Translations</h2>")
        html_parts.append("<table border='1' cellpadding='4' cellspacing='0'>")
        html_parts.append("<tr><th>Rank</th><th>Quality</th><th>Similarity</th><th>Confidence</th><th>Translation</th></tr>")
        for idx, item in enumerate(data["ranked_translations"]):
            html_parts.append(
                f"<tr><td>{idx+1}</td><td>{item.get('quality_score',0):.4f}</td>"
                f"<td>{item.get('similarity',0):.4f}</td><td>{item.get('confidence',0):.4f}</td>"
                f"<td>{esc(item.get('text') or item.get('translation',''))}</td></tr>"
            )
        html_parts.append("</table>")

    if include_diagnostics and data.get("diagnostics"):
        diag = data["diagnostics"]
        html_parts.append("<h2>Clustering Diagnostics</h2><ul>")
        for k, v in diag.items():
            if isinstance(v, float):
                v = f"{v:.4f}"
            html_parts.append(f"<li>{esc(k)}: {esc(v)}</li>")
        html_parts.append("</ul>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


class AsyncTranslationEvaluator:
    """
    Enhanced async translation evaluator that provides backward compatibility
    with the original main.py functionality while adding async processing capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, interactive: bool = False):
        """Initialize the async translation evaluator."""
        self.config = ConfigManager(config_path) if config_path else ConfigManager()
        self.interactive = interactive
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if performance_monitoring_available else None
        self.perf_data = {}
        
        # Note: PerformanceMonitor doesn't have register_callback method
        # We'll track performance manually in the perf_data dictionary
        
        # Async settings
        self.max_concurrent_embeddings = self.config.get("async.max_concurrent_embeddings", default=10)
        self.max_concurrent_api_calls = self.config.get("async.max_concurrent_api_calls", default=5)
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.get("async.thread_workers", default=4))
        
        # Semaphores
        self.embedding_semaphore = asyncio.Semaphore(self.max_concurrent_embeddings)
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_api_calls)
        
        # Caching
        self.cache_dir = self.config.get("cache.directory", default="cache")
        self.embedding_cache = EmbeddingCache(
            max_size=self.config.get("cache.embedding_size", default=10000)
        )
        
        # Smart cache if available
        if SmartCache:
            self.smart_cache = SmartCache(
                cache_dir=self.cache_dir,
                ttl=self.config.get("cache.ttl", default=86400),
                max_size=self.config.get("cache.max_size", default=1000)
            )
        else:
            self.smart_cache = None
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
        
        # Components (initialized later)
        self.model_loader = None
        self.embedding_generator = None
        self.groq_client = None
        self.groq_evaluator = None
        self.quality_analyzer = None
        self.analyzer = None  # For backward compatibility
        
        self.initialized = False
        self.use_async = HAS_ASYNC_LIBS and ASYNC_COMPONENTS_AVAILABLE
        
        logger.info(f"{'Async' if self.use_async else 'Sync'} Translation Evaluator initialized")
        
        # Register shutdown handlers
        atexit.register(self._sync_shutdown_handler)
    
    def _sync_shutdown_handler(self):
        """Synchronous shutdown handler for atexit."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._shutdown())
        finally:
            loop.close()
    
    async def _shutdown(self):
        """Perform graceful shutdown operations."""
        logger.info("Performing graceful shutdown...")
        self.shutdown_event.set()
        await self.close()
    
    def _update_performance_data(self, operation: str, duration: float, metadata: Dict[str, Any]):
        """Callback for performance monitor to update performance data."""
        if operation not in self.perf_data:
            self.perf_data[operation] = {
                "count": 0, "total_time": 0.0, "min_time": float("inf"), 
                "max_time": 0.0, "avg_time": 0.0
            }
        
        data = self.perf_data[operation]
        data["count"] += 1
        data["total_time"] += duration
        data["min_time"] = min(data["min_time"], duration)
        data["max_time"] = max(data["max_time"], duration)
        data["avg_time"] = data["total_time"] / data["count"]
    
    async def initialize(self) -> None:
        """Initialize async components and resources."""
        if self.initialized:
            return
        
        with LogContext(operation="initialization"):
            logger.info("Initializing components...")
            
            # Initialize model loader
            self.model_loader = ModelLoader(config=self.config)
            # Model will be loaded lazily when needed
            
            # Initialize components based on availability
            if self.use_async:
                # Async components (fallback to sync for now)
                self.embedding_generator = EmbeddingGenerator(
                    use_cache=True
                )
                
                self.groq_client = GroqClient(
                    api_key=self.config.get_api_key("groq")
                )
                
                self.groq_evaluator = GroqEvaluator(self.groq_client)
                
                self.quality_analyzer = SyncTranslationQualityAnalyzer(
                    embedding_generator=self.embedding_generator, 
                    groq_evaluator=self.groq_evaluator,
                    config_manager=self.config
                )
            else:
                # Sync components  
                self.embedding_generator = EmbeddingGenerator(
                    use_cache=True
                )
                
                self.groq_client = GroqClient(
                    api_key=self.config.get_api_key("groq")
                )
                
                self.groq_evaluator = GroqEvaluator(self.groq_client)
                
                self.quality_analyzer = SyncTranslationQualityAnalyzer(
                    self.embedding_generator, self.groq_evaluator, self.config
                )
            
            # For backward compatibility
            self.analyzer = self.quality_analyzer
            
            self.initialized = True
            logger.info("Components initialized successfully")
    
    async def close(self) -> None:
        """Close resources and connections."""
        if hasattr(self, 'groq_client') and self.groq_client:
            if hasattr(self.groq_client, 'session') and hasattr(self.groq_client.session, 'closed'):
                if not self.groq_client.session.closed:
                    await self.groq_client.session.close()
        
        self.thread_pool.shutdown(wait=False)
        
        if self.embedding_cache:
            # EmbeddingCache is already in memory, no need to save to disk
            self.embedding_cache.clear()
        
        if self.smart_cache:
            self.smart_cache.save()
        
        logger.info("Resources closed successfully")
    
    async def analyze_translation(
        self, 
        source_text: str, 
        translated_text: str,
        source_lang: str = 'en',
        target_lang: str = 'es',
        include_error_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a translation with backward compatibility for the original interface.
        """
        if not self.initialized:
            await self.initialize()
        
        with LogContext(source_lang=source_lang, target_lang=target_lang):
            # Use sync analyzer with async wrapper (since async components don't exist yet)
            def analyze_wrapper():
                return self.quality_analyzer.analyze_pair(
                    source_text, translated_text, use_groq=True, detailed=False
                )
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, analyze_wrapper
            )
            
            # Add backward compatibility fields if needed
            if 'embedding_similarity' not in result and 'similarity' in result:
                result['embedding_similarity'] = result['similarity']
            
            return result
    
    async def demo_translation_evaluation(self):
        """Interactive demo for translation evaluation."""
        if not console:
            logger.warning("Rich console not available for demo")
            return
            
        console.print(Panel.fit(
            "[bold green]Translation Evaluation Demo[/bold green]\n"
            "This demo will evaluate the quality of a translation using multiple methods."
        ))
        
        # Sample translation pairs for demo
        samples = [
            {
                "source": "The quick brown fox jumps over the lazy dog.",
                "target": "El zorro marrón rápido salta sobre el perro perezoso.",
                "source_lang": "en",
                "target_lang": "es",
                "description": "Classic pangram"
            },
            {
                "source": "It was the best of times, it was the worst of times.",
                "target": "Era el mejor de los tiempos, era el peor de los tiempos.",
                "source_lang": "en",
                "target_lang": "es",
                "description": "Literature (A Tale of Two Cities)"
            },
            {
                "source": "Machine learning is a subset of artificial intelligence.",
                "target": "El aprendizaje automático es un subconjunto de la inteligencia artificial.",
                "source_lang": "en",
                "target_lang": "es",
                "description": "Technical content"
            }
        ]
        
        # Ask user to select a sample or enter custom text
        console.print("[bold]Choose an option:[/bold]")
        console.print("1. Use a sample translation")
        console.print("2. Enter your own translation")
        
        choice = Prompt.ask("Enter your choice", choices=["1", "2"], default="1")
        
        if choice == "1":
            # Display sample options
            console.print("\n[bold]Select a sample translation:[/bold]")
            for i, sample in enumerate(samples):
                console.print(f"{i+1}. [cyan]{sample['description']}[/cyan]")
                console.print(f"   Source ({sample['source_lang']}): {sample['source']}")
                console.print(f"   Target ({sample['target_lang']}): {sample['target']}")
                console.print()
                
            sample_choice = Prompt.ask(
                "Select a sample", 
                choices=[str(i+1) for i in range(len(samples))],
                default="1"
            )
            
            selected = samples[int(sample_choice) - 1]
            source_text = selected["source"]
            translated_text = selected["target"]
            source_lang = selected["source_lang"]
            target_lang = selected["target_lang"]
            
        else:
            # Get custom input
            console.print("\n[bold]Enter your translation:[/bold]")
            source_text = Prompt.ask("Source text")
            source_lang = Prompt.ask("Source language code", default="en")
            translated_text = Prompt.ask("Translated text")
            target_lang = Prompt.ask("Target language code", default="es")
        
        # Display processing status
        with console.status("[bold green]Evaluating translation...[/bold green]"):
            # Run the evaluation
            result = await self.analyze_translation(
                source_text, translated_text, source_lang, target_lang
            )
        
        # Display results
        self._display_evaluation_results(result)
        
        # Ask if user wants to try another translation
        if Confirm.ask("\nWould you like to evaluate another translation?", default=False):
            await self.demo_translation_evaluation()
        
        return result
    
    def _display_evaluation_results(self, result: Dict[str, Any]) -> None:
        """Display evaluation results using rich."""
        if not console:
            return
            
        # Create results panel
        console.print("\n[bold]Evaluation Results:[/bold]")
        
        # Display source and translation
        source = result.get('source_text', '')
        translation = result.get('translated_text', '')
        
        console.print(Panel(
            f"[bold]Source[/bold] ({result.get('source_language', 'unknown')}):\n{source}",
            border_style="blue"
        ))
        console.print(Panel(
            f"[bold]Translation[/bold] ({result.get('target_language', 'unknown')}):\n{translation}",
            border_style="green"
        ))
        
        # Display scores
        embedding_score = result.get('embedding_similarity', 0)
        groq_score = result.get('groq_quality_score', 0)
        combined_score = result.get('combined_score', 0)
        
        # Color-code scores
        def color_score(score, max_val=1.0):
            normalized = score / max_val
            if normalized >= 0.8:
                return f"[green]{score:.2f}[/green]"
            elif normalized >= 0.6:
                return f"[yellow]{score:.2f}[/yellow]"
            else:
                return f"[red]{score:.2f}[/red]"
        
        # Create scores table
        scores_table = Table(show_header=True, header_style="bold")
        scores_table.add_column("Metric")
        scores_table.add_column("Score")
        
        scores_table.add_row("Embedding Similarity", color_score(embedding_score))
        scores_table.add_row("Groq Quality Score", color_score(groq_score, 10.0))
        scores_table.add_row("Combined Score", color_score(combined_score))
        
        console.print(scores_table)
        
        # Display Groq analysis if available
        strengths = result.get('groq_strengths', [])
        weaknesses = result.get('groq_weaknesses', [])
        analysis = result.get('groq_analysis', '')
        
        if analysis or strengths or weaknesses:
            console.print("\n[bold]Groq Analysis:[/bold]")
            
            if strengths:
                console.print("[bold green]Strengths:[/bold green]")
                for strength in strengths:
                    console.print(f"  ✓ {strength}")
            
            if weaknesses:
                console.print("[bold red]Weaknesses:[/bold red]")
                for weakness in weaknesses:
                    console.print(f"  ✗ {weakness}")
            
            if analysis:
                console.print(Panel(analysis, title="Detailed Analysis"))
    
    async def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single file containing translations."""
        if not self.initialized:
            await self.initialize()
            
        file_path = Path(file_path)
        with LogContext(operation="file_processing", file=str(file_path)):
            logger.info(f"Processing file: {file_path}")
            
            try:
                # Read file asynchronously if available
                if HAS_ASYNC_LIBS:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                else:
                    # Fallback to sync reading in thread pool
                    content = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        lambda: file_path.read_text(encoding='utf-8')
                    )
                
                # Parse content (JSON format)
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Try as line-delimited JSON
                    data = []
                    for line in content.splitlines():
                        if line.strip():
                            try:
                                item = json.loads(line)
                                data.append(item)
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
                
                if isinstance(data, dict):
                    data = [data]
                
                # Process each item
                results = []
                for item in data:
                    try:
                        source_text = item.get('source_text', item.get('source', ''))
                        translated_text = item.get('translated_text', item.get('translation', item.get('target', '')))
                        source_lang = item.get('source_lang', item.get('source_language', 'en'))
                        target_lang = item.get('target_lang', item.get('target_language', 'es'))
                        
                        result = await self.analyze_translation(
                            source_text, translated_text, source_lang, target_lang
                        )
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error processing item: {str(e)}")
                        results.append({
                            "error": str(e),
                            "source_text": item.get('source_text', ''),
                            "translated_text": item.get('translated_text', ''),
                            "status": "error"
                        })
                
                # Prepare output path
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{file_path.stem}_results.json"
                
                # Write results
                if HAS_ASYNC_LIBS:
                    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(results, indent=2))
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        lambda: output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
                    )
                
                logger.info(f"Results saved to {output_path}")
                
                result = {
                    "file_path": str(file_path),
                    "output_path": str(output_path),
                    "items_processed": len(results),
                    "successful_items": sum(1 for r in results if "error" not in r),
                    "status": "success",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                return result
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
                return {
                    "file_path": str(file_path),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }
    
    async def process_directory(self, directory_path: Union[str, Path], pattern: str = "*.json") -> List[Dict[str, Any]]:
        """Process all matching files in a directory."""
        if not self.initialized:
            await self.initialize()
            
        directory_path = Path(directory_path)
        with LogContext(operation="directory_processing", directory=str(directory_path)):
            logger.info(f"Processing directory: {directory_path} (pattern: {pattern})")
            
            # Find all matching files
            files = list(directory_path.glob(pattern))
            logger.info(f"Found {len(files)} files matching pattern")
            
            # Process files sequentially to avoid overwhelming the system
            results = []
            for file_path in files:
                try:
                    result = await self.process_file(file_path)
                    results.append(result)
                    logger.info(f"Processed {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    results.append({
                        "file_path": str(file_path),
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
            successful = sum(1 for r in results if r.get("status") == "success")
            logger.info(f"Directory processing completed: {successful}/{len(results)} files successful")
            
            return results


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('translation_evaluator.log')
    ]
)

# Get enhanced logger for main module
logger = get_logger(__name__, "system")

# Create a global performance monitor object
performance_monitor = None

# Global console for rich output
console = Console()

@contextmanager
def monitor_operation(operation_name):
    """Context manager for timing operations and tracking performance."""
    global performance_monitor
    start_time = time.time()
    success = True
    
    try:
        if performance_monitor:
            performance_monitor.start_timer(operation_name)
        yield
    except Exception as e:
        success = False
        raise e
    finally:
        elapsed = time.time() - start_time
        if performance_monitor:
            if operation_name.startswith("api_"):
                # For API operations, we record both the timer and API stats
                parts = operation_name.split("_", 2)
                if len(parts) == 3:  # api_name_endpoint
                    api_name, endpoint = parts[1], parts[2]
                    error_message = None if success else str(sys.exc_info()[1])
                    performance_monitor.record_api_response(
                        api_name, endpoint, elapsed, success, error_message=error_message
                    )
                performance_monitor.stop_timer(operation_name)
            elif operation_name.startswith("component_"):
                # For component operations
                component_name = operation_name[10:]  # Remove "component_" prefix
                performance_monitor.record_component_execution(component_name, success, elapsed)
                performance_monitor.stop_timer(operation_name)
            else:
                # For regular operations
                performance_monitor.stop_timer(operation_name)

def setup_components():
    """Set up the required components for translation evaluation with error handling and performance monitoring."""
    global performance_monitor
    
    components = {
        "analyzer": None,
        "dual_system": None,
        "config_manager": None,
        "weight_config_tool": None,
        "errors": []
    }
    
    try:
        # Initialize performance monitor first if available
        if performance_monitoring_available:
            try:
                performance_monitor = PerformanceMonitor(log_dir="./logs/performance")
                logging.info("Performance monitoring initialized")
            except Exception as e:
                logging.error(f"Failed to initialize performance monitoring: {e}")
                performance_monitor = None
        
        # Import all required modules with error handling and performance tracking
        with monitor_operation("component_config_manager"):
            try:
                from config_manager import ConfigManager
                components["config_manager"] = ConfigManager()
                # Update performance monitor with config manager if needed
                if performance_monitor and components["config_manager"]:
                    performance_monitor = PerformanceMonitor(
                        config_manager=components["config_manager"],
                        log_dir=components["config_manager"].get("performance_log_dir", "./logs/performance")
                    )
                logging.info("Configuration manager initialized successfully")
            except ImportError as e:
                logging.warning(f"Could not import ConfigManager: {e}")
                components["errors"].append(f"No configuration manager available: {str(e)}")
            except Exception as e:
                logging.error(f"Error initializing ConfigManager: {e}")
                components["errors"].append(f"Configuration manager initialization failed: {str(e)}")
        
        # Set up embedding components with performance monitoring
        embedding_generator = None
        with monitor_operation("component_embedding"):
            try:
                from model_loader import ModelLoader
                from translation_quality_analyzer import EmbeddingGenerator
                from embedding_cache import EmbeddingCache
                
                cache_dir = components["config_manager"].get("cache_dir", "./cache") if components["config_manager"] else "./cache"
                model_name = components["config_manager"].get("model", "all-MiniLM-L6-v2") if components["config_manager"] else "all-MiniLM-L6-v2"
                
                # Initialize embedding cache
                embedding_cache = EmbeddingCache(cache_dir=cache_dir)
                logging.info("Embedding cache initialized")
                
                # Initialize model loader
                model_loader = ModelLoader(model_name=model_name, cache_dir=cache_dir)
                logging.info(f"Model loader initialized with model: {model_name}")
                
                # Initialize embedding generator
                embedding_generator = EmbeddingGenerator(model_loader, embedding_cache)
                logging.info("Embedding generator initialized successfully")
                
            except ImportError as e:
                logging.error(f"Could not import embedding modules: {e}")
                components["errors"].append(f"Embedding functionality not available: {str(e)}")
            except Exception as e:
                logging.error(f"Error initializing embedding components: {e}")
                components["errors"].append(f"Embedding initialization failed: {str(e)}")
        
        # Initialize text processor with performance monitoring
        text_processor = None
        with monitor_operation("component_text_processor"):
            try:
                from text_processor import TextProcessor
                text_processor = TextProcessor()
                logging.info("Text processor initialized")
            except ImportError as e:
                logging.warning(f"Could not import TextProcessor: {e}")
                components["errors"].append(f"Text processing not available: {str(e)}")
            except Exception as e:
                logging.error(f"Error initializing TextProcessor: {e}")
                components["errors"].append(f"Text processor initialization failed: {str(e)}")
        
        # Initialize segment alignment analyzer with performance monitoring
        segment_alignment_analyzer = None
        with monitor_operation("component_segment_alignment"):
            try:
                import segment_alignment
                segment_alignment_analyzer = segment_alignment.SegmentAlignmentAnalyzer(
                    embedding_generator, text_processor
                )
                logging.info("Segment alignment analyzer initialized")
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not import SegmentAlignmentAnalyzer: {e}")
                components["errors"].append("Segment alignment not available")
            except Exception as e:
                logging.error(f"Error initializing SegmentAlignmentAnalyzer: {e}")
                components["errors"].append(f"Segment alignment initialization failed: {str(e)}")
        
        # Initialize Groq components with performance monitoring
        groq_client = None
        groq_evaluator = None
        with monitor_operation("component_groq"):
            try:
                from groq_client import GroqClient
                from groq_evaluator import GroqTranslationEvaluator
                
                groq_api_key = None
                if components["config_manager"]:
                    groq_api_key = components["config_manager"].get_api_key("groq")
                
                if not groq_api_key:
                    groq_api_key = os.environ.get("GROQ_API_KEY")
                
                if groq_api_key:
                    try:
                        groq_client = GroqClient(api_key=groq_api_key)
                        # Wrap API methods of groq_client with performance monitoring
                        if performance_monitor:
                            original_evaluate_translation = groq_client.evaluate_translation
                            original_generate_completion = groq_client.generate_completion
                            original_generate_chat_completion = groq_client.generate_chat_completion
                            
                            def wrapped_evaluate_translation(*args, **kwargs):
                                with monitor_operation("api_groq_evaluate_translation"):
                                    return original_evaluate_translation(*args, **kwargs)
                            
                            def wrapped_generate_completion(*args, **kwargs):
                                with monitor_operation("api_groq_generate_completion"):
                                    return original_generate_completion(*args, **kwargs)
                            
                            def wrapped_generate_chat_completion(*args, **kwargs):
                                with monitor_operation("api_groq_generate_chat_completion"):
                                    return original_generate_chat_completion(*args, **kwargs)
                            
                            groq_client.evaluate_translation = wrapped_evaluate_translation
                            groq_client.generate_completion = wrapped_generate_completion
                            groq_client.generate_chat_completion = wrapped_generate_chat_completion
                        
                        groq_evaluator = GroqTranslationEvaluator(groq_client)
                        logging.info("Groq evaluation services initialized successfully")
                    except Exception as e:
                        logging.warning(f"Failed to initialize Groq services with API key: {e}")
                        components["errors"].append(f"Groq API initialization failed: {str(e)}")
                else:
                    logging.warning("Groq API key not found. Linguistic assessment will not be available.")
                    components["errors"].append("Groq API key not found. Linguistic assessment will be limited.")
            except ImportError as e:
                logging.warning(f"Could not import Groq modules: {e}")
                components["errors"].append(f"Groq functionality not available: {str(e)}")
            except Exception as e:
                logging.error(f"Error initializing Groq components: {e}")
                components["errors"].append(f"Groq initialization failed: {str(e)}")
        
        # Initialize weight config tool with performance monitoring
        weight_config_tool = None
        with monitor_operation("component_weight_config"):
            try:
                from weight_config_tool import WeightConfigTool
                weight_config_tool = WeightConfigTool(components["config_manager"])
                components["weight_config_tool"] = weight_config_tool
                logging.info("Weight configuration tool initialized")
            except ImportError as e:
                logging.warning(f"Could not import WeightConfigTool: {e}")
                components["errors"].append("Weight configuration tool not available")
            except Exception as e:
                logging.error(f"Error initializing WeightConfigTool: {e}")
                components["errors"].append(f"Weight configuration tool initialization failed: {str(e)}")
        
        # Initialize translation quality analyzer with performance monitoring
        with monitor_operation("component_translation_quality_analyzer"):
            try:
                from translation_quality_analyzer import TranslationQualityAnalyzer
                analyzer = TranslationQualityAnalyzer(
                    embedding_generator=embedding_generator,
                    groq_evaluator=groq_evaluator,
                    config_manager=components["config_manager"],
                    text_processor=text_processor,
                    segment_alignment=segment_alignment_analyzer
                )
                components["analyzer"] = analyzer
                logging.info("Translation quality analyzer initialized")
            except ImportError as e:
                logging.error(f"Could not import TranslationQualityAnalyzer: {e}")
                components["errors"].append(f"Translation quality analyzer not available: {str(e)}")
            except Exception as e:
                logging.error(f"Error initializing TranslationQualityAnalyzer: {e}")
                components["errors"].append(f"Translation quality analyzer initialization failed: {str(e)}")
        
        # Initialize dual analysis system with performance monitoring
        with monitor_operation("component_dual_analysis_system"):
            try:
                from dual_analysis_system import DualAnalysisSystem
                dual_system = DualAnalysisSystem(
                    embedding_generator=embedding_generator,
                    groq_evaluator=groq_evaluator,
                    config_manager=components["config_manager"],
                    text_processor=text_processor
                )
                components["dual_system"] = dual_system
                logging.info("Dual analysis system initialized")
            except ImportError as e:
                logging.warning(f"Could not import DualAnalysisSystem: {e}")
                components["errors"].append(f"Dual analysis system not available: {str(e)}")
            except Exception as e:
                logging.error(f"Error initializing DualAnalysisSystem: {e}")
                components["errors"].append(f"Dual analysis system initialization failed: {str(e)}")
        
        # Log the number of errors encountered during setup
        components_status = {
            "analyzer": components["analyzer"] is not None,
            "dual_system": components["dual_system"] is not None,
            "config_manager": components["config_manager"] is not None,
            "weight_config_tool": components["weight_config_tool"] is not None,
            "embedding": embedding_generator is not None,
            "groq": groq_evaluator is not None,
            "text_processor": text_processor is not None,
            "performance_monitor": performance_monitor is not None
        }
        
        logging.info(f"Components initialized with status: {components_status}")
        if components["errors"]:
            logging.warning(f"Setup completed with {len(components['errors'])} warnings/errors")
        else:
            logging.info("Setup completed successfully with no errors")
            
        # Add performance monitor to components
        components["performance_monitor"] = performance_monitor
            
        return components
            
    except Exception as e:
        logging.critical(f"Critical error during component setup: {e}\n{traceback.format_exc()}")
        components["errors"].append(f"Critical setup error: {str(e)}")
        return components

def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Translation Quality Analyzer - Evaluate the quality of translations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--source-file', '-sf', type=str,
        help='Path to source text file'
    )
    input_group.add_argument(
        '--target-file', '-tf', type=str,
        help='Path to translated text file'
    )
    input_group.add_argument(
        '--source-text', '-st', type=str,
        help='Source text provided directly via command line'
    )
    input_group.add_argument(
        '--target-text', '-tt', type=str,
        help='Translated text provided directly via command line'
    )
    input_group.add_argument(
        '--interactive', '-i', action='store_true',
        help='Enter source and target text interactively'
    )
    
    # Translation operations
    translation_group = parser.add_argument_group('Translation Operations')
    translation_group.add_argument(
        '--translate', action='store_true',
        help='Translate source text to target language'
    )
    translation_group.add_argument(
        '--rate-translation', action='store_true',
        help='Rate the quality of a translation without analyzing it'
    )
    translation_group.add_argument(
        '--translation-model', type=str,
        help='Specify model to use for translation'
    )
    translation_group.add_argument(
        '--translate-batch', action='store_true',
        help='Process file as a batch of translations (one per line)'
    )
    translation_group.add_argument(
        '--batch-output', type=str,
        help='Output file for batch translation results'
    )
    
    # Language options
    language_group = parser.add_argument_group('Language Options')
    language_group.add_argument(
        '--source-lang', type=str,
        help='Source language code (ISO 639-1, e.g., "en", "fr", "zh")'
    )
    language_group.add_argument(
        '--target-lang', type=str,
        help='Target language code (ISO 639-1)'
    )
    language_group.add_argument(
        '--list-languages', action='store_true',
        help='List supported languages and exit'
    )
    language_group.add_argument(
        '--detect-language', action='store_true',
        help='Detect language of input text'
    )
    language_group.add_argument(
        '--analyze-composition', action='store_true',
        help='Analyze linguistic composition of text'
    )
    language_group.add_argument(
        '--fast-detection', action='store_true',
        help='Use faster but potentially less accurate language detection'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--similarity', action='store_true',
        help='Perform semantic similarity analysis'
    )
    analysis_group.add_argument(
        '--metric', type=str,
        choices=['cosine', 'euclidean', 'dot', 'manhattan', 'angular'],
        default='cosine',
        help='Similarity metric to use for comparison'
    )
    analysis_group.add_argument(
        '--classify', action='store_true',
        help='Include semantic match classification in results'
    )
    analysis_group.add_argument(
        '--segmented', action='store_true',
        help='Treat input texts as individual segments rather than documents'
    )
    analysis_group.add_argument(
        '--cross-lingual', action='store_true',
        help='Enable cross-lingual comparison mode'
    )
    analysis_group.add_argument(
        '--similarity-threshold', type=float, default=0.75,
        help='Threshold for considering texts similar (0.0-1.0)'
    )
    analysis_group.add_argument(
        '--preprocessing', type=str,
        choices=['minimal', 'standard', 'aggressive'], default='standard',
        help='Level of text preprocessing to apply'
    )
    analysis_group.add_argument(
        '--detailed-report', action='store_true',
        help='Generate a detailed analysis report'
    )
    
    # Groq integration options
    groq_group = parser.add_argument_group('Groq Options')
    groq_flags = groq_group.add_mutually_exclusive_group()
    groq_flags.add_argument(
        '--use-groq', dest='use_groq', action='store_true', default=None,
        help='Use Groq LLM for enhanced quality evaluation'
    )
    groq_flags.add_argument(
        '--no-groq', dest='use_groq', action='store_false',
        help='Disable Groq LLM and rely on embedding-based metrics only'
    )

    # Alignment analysis options
    alignment_group = parser.add_argument_group('Alignment Analysis')
    alignment_group.add_argument(
        '--weak-alignment', dest='weak_alignment', action='store_true',
        help='Detect and report weak segment alignments in addition to overall analysis'
    )
    alignment_group.add_argument(
        '--segment-type', dest='segment_type', choices=['sentence', 'paragraph'], default='sentence',
        help='Segmentation granularity used for alignment analysis'
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument(
        '--config', type=str,
        help='Path to configuration file'
    )
    config_group.add_argument(
        '--api-key', type=str,
        help='API key for remote services'
    )
    config_group.add_argument(
        '--cache-dir', type=str,
        help='Directory for caching models and embeddings'
    )
    config_group.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching of embeddings and models'
    )
    config_group.add_argument(
        '--inference-mode', type=str,
        choices=['local', 'api', 'hybrid'], default='local',
        help='Mode for model inference (local, api, or hybrid)'
    )
    config_group.add_argument(
        '--embedding-model', type=str,
        help='Name or path of embedding model to use'
    )
    config_group.add_argument(
        '--multilingual-model', type=str,
        choices=['default', 'high_quality', 'efficient'],
        help='Type of multilingual model to use'
    )
    
    # Candidate Ranking options
    ranking_group = parser.add_argument_group('Candidate Ranking')
    ranking_group.add_argument('--rank-candidates', action='store_true',
                               help='Rank multiple candidate translations')
    ranking_group.add_argument('--candidates', type=str,
                               help='Comma-separated candidate translations')
    ranking_group.add_argument('--candidates-file', type=str,
                               help='Path to file with candidate translations (one per line)')
    ranking_group.add_argument('--confidence-method', type=str,
                               choices=['distribution', 'gap', 'range'], default='distribution',
                               help='Confidence scoring method')
    ranking_group.add_argument('--output-format', type=str,
                               choices=['table', 'json', 'csv', 'yaml', 'html'], default='table',
                               help='Output format for ranking results')
    ranking_group.add_argument('--include-diagnostics', action='store_true',
                               help='Include clustering diagnostics in the output')
    ranking_group.add_argument('--output-file', type=str,
                               help='Save results to a file instead of printing to stdout')
    ranking_group.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                               help='Embedding model to use for ranking')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output', '-o', type=str,
        help='Output file for analysis results (default: stdout)'
    )
    output_group.add_argument(
        '--format', type=str,
        choices=['text', 'json', 'html', 'markdown'], default='text',
        help='Output format'
    )
    output_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    output_group.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress all non-error output'
    )
    output_group.add_argument(
        '--color', type=str, choices=['auto', 'always', 'never'], default='auto',
        help='Control colored output'
    )
    
    # Misc options
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--version', action='store_true',
        help='Show version information and exit'
    )
    misc_group.add_argument(
        '--debug', action='store_true',
        help='Enable debug output'
    )
    misc_group.add_argument(
        '--clear-cache', action='store_true',
        help='Clear all cached data and exit'
    )
    misc_group.add_argument(
        '--run-alignment-demo', action='store_true',
        help='Run a demo of segment alignment analysis'
    )
    misc_group.add_argument(
        '--run-weight-demo', action='store_true',
        help='Run a demo of custom quality score weights'
    )
    
    # Add performance monitoring options
    perf_group = parser.add_argument_group('Performance monitoring')
    perf_group.add_argument('--monitor', action='store_true', 
                          help='Enable detailed performance monitoring')
    perf_group.add_argument('--performance-report', action='store_true',
                          help='Generate performance report after execution')
    perf_group.add_argument('--performance-output', 
                          help='Output file for performance report')
    perf_group.add_argument('--performance-format', 
                          choices=['markdown', 'text', 'json'], default='markdown',
                          help='Format for performance report')
    
    # Add cache statistics and management options  
    cache_group = parser.add_argument_group('Cache management')
    cache_group.add_argument('--cache-stats', action='store_true',
                          help='Display detailed cache statistics')
    cache_group.add_argument('--cache-stats-format', 
                          choices=['text', 'json', 'markdown', 'html'], 
                          default='text',
                          help='Format for cache statistics output')
    cache_group.add_argument('--cache-stats-output', 
                          help='Save cache statistics to file')
    cache_group.add_argument('--cache-analyze', action='store_true',
                          help='Analyze cache performance and provide recommendations')
    cache_group.add_argument('--disable-cache', action='store_true',
                          help='Disable the cache for this execution')
    
    # Add async processing options
    async_group = parser.add_argument_group('Async Processing')
    async_group.add_argument('--async-mode', action='store_true',
                           help='Enable asynchronous processing mode')
    async_group.add_argument('--max-concurrent-embeddings', type=int, default=10,
                           help='Maximum concurrent embedding operations')
    async_group.add_argument('--max-concurrent-api-calls', type=int, default=5,
                           help='Maximum concurrent API calls')
    async_group.add_argument('--thread-workers', type=int, default=4,
                           help='Number of thread pool workers for CPU-bound operations')
    async_group.add_argument('--batch-file', type=str,
                           help='Process a batch file of translations (JSON format)')
    async_group.add_argument('--batch-directory', type=str,
                           help='Process all files in a directory')
    async_group.add_argument('--file-pattern', type=str, default='*.json',
                           help='File pattern for batch directory processing')
    async_group.add_argument('--demo', action='store_true',
                           help='Run interactive demonstration')
    async_group.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmark')
    async_group.add_argument('--benchmark-iterations', type=int, default=100,
                           help='Number of iterations for benchmark')
    
    return parser

def read_file_text(file_path: str) -> str:
    """
    Read text from a file, handling encoding issues.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File contents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file can't be decoded
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Try UTF-8 first (most common)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try other common encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
                
        # If all else fails, try binary mode with errors='replace'
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            logger.warning(f"File {file_path} contains characters that couldn't be decoded properly")
            return text

def get_interactive_input(prompt: str, multiline: bool = True) -> str:
    """
    Get text input from the user interactively.
    
    Args:
        prompt: Text to display as prompt
        multiline: Whether to allow multiline input
        
    Returns:
        str: User input
    """
    console = Console()
    console.print(f"\n[bold cyan]{prompt}[/bold cyan]")
    
    if multiline:
        console.print("[dim](Enter text, then press CTRL+D (or CTRL+Z on Windows) to finish)[/dim]")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            # User pressed CTRL+D (or CTRL+Z on Windows)
            pass
        return "\n".join(lines)
    else:
        return input()

def show_version():
    """Display version information."""
    version = "1.0.0"  # Should be imported from package
    console = Console()
    console.print(Panel(
        f"[bold]Translation Quality Analyzer[/bold] version {version}\n"
        f"Python {sys.version}",
        title="Version Information",
        border_style="cyan"
    ))

def list_supported_languages():
    """List supported languages."""
    # Initialize minimal components to list languages
    config = ConfigManager()
    model_loader = ModelLoader(config)
    
    # Now properly initialize MultilingualModelManager
    multilingual_manager = MultilingualModelManager(config, model_loader)
    
    # Get languages from the multilingual manager instead
    langs = multilingual_manager.get_supported_languages()
    console = Console()
    
    # Create a formatted table
    table = Table(title="Supported Languages", box=Box.ROUNDED)
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Language", style="green")
    table.add_column("Family", style="yellow")
    
    # Add rows sorted alphabetically by language name
    for lang in sorted(langs, key=lambda x: x['name']):
        family = lang['family'] if lang['family'] else "Other"
        table.add_row(lang['code'], lang['name'], family)
    
    console.print(table)

def clear_all_cache(config: ConfigManager):
    """Clear all cached data."""
    console = Console()
    
    cache_dir = Path(os.path.expanduser(config.get("cache.directory", "~/.tqa/cache")))
    models_dir = Path(os.path.expanduser(config.get("models.embedding.cache_dir", "~/.tqa/models")))
    embedding_cache = Path(os.path.expanduser(config.get("models.embedding.cache_dir", "~/.tqa/embedding_cache")))
    language_cache = Path(os.path.expanduser(config.get("language.cache_dir", "~/.tqa/language_cache")))
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Clearing cache...", total=4)
        
        # Clear cache directory
        if cache_dir.exists():
            for file in cache_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil
                        shutil.rmtree(file)
                except Exception as e:
                    logger.error(f"Error clearing cache file {file}: {e}")
        progress.update(task, advance=1)
        
        # Clear embedding cache
        if embedding_cache.exists():
            for file in embedding_cache.glob("*.npy"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error clearing embedding cache file {file}: {e}")
        progress.update(task, advance=1)
        
        # Clear language cache
        if language_cache.exists():
            for file in language_cache.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error clearing language cache file {file}: {e}")
        progress.update(task, advance=1)
        
        # Note: We don't clear model files by default as they're large downloads
        # Just inform the user
        model_count = sum(1 for _ in models_dir.glob("*")) if models_dir.exists() else 0
        progress.update(task, advance=1)
    
    console.print(f"[green]✓[/green] Cache directories cleared")
    if model_count > 0:
        console.print(f"[yellow]![/yellow] {model_count} model files remain in {models_dir}")
        console.print(f"[dim]To clear models as well, manually delete: {models_dir}[/dim]")

def write_to_file(file_path: str, content: str):
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e}")
        return False

def format_cache_stats(stats, format='text'):
    """
    Format cache statistics for display.
    
    Args:
        stats: Cache statistics dictionary
        format: Output format ('text', 'json', 'markdown', or 'html')
        
    Returns:
        Formatted statistics string
    """
    import json
    import math
    
    if format == 'json':
        return json.dumps(stats, indent=2)
        
    # For non-JSON formats, prepare the data
    memory_size = stats.get('memory_size', 0)
    memory_max = stats.get('memory_max_size', 0)
    disk_size = stats.get('disk_size', 0)
    disk_max = stats.get('disk_max_size', 0)
    
    memory_usage_pct = (memory_size / memory_max * 100) if memory_max > 0 else 0
    disk_usage_pct = (disk_size / disk_max * 100) if disk_max > 0 else 0
    
    memory_hits = stats.get('memory_hits', 0)
    memory_misses = stats.get('memory_misses', 0)
    disk_hits = stats.get('disk_hits', 0)
    disk_misses = stats.get('disk_misses', 0)
    
    total_requests = memory_hits + memory_misses
    memory_hit_rate = (memory_hits / total_requests * 100) if total_requests > 0 else 0
    
    total_disk_requests = disk_hits + disk_misses
    disk_hit_rate = (disk_hits / total_disk_requests * 100) if total_disk_requests > 0 else 0
    
    overall_hits = memory_hits + disk_hits
    overall_requests = total_requests + total_disk_requests - disk_hits  # Avoid double counting
    overall_hit_rate = (overall_hits / overall_requests * 100) if overall_requests > 0 else 0
    
    memory_evictions = stats.get('memory_evictions', 0)
    disk_evictions = stats.get('disk_evictions', 0)
    
    api_calls_saved = stats.get('api_calls_saved', 0)
    computation_time_saved = stats.get('computation_time_saved', 0)
    bytes_saved = stats.get('bytes_saved', 0)
    
    # Calculate derived metrics
    if computation_time_saved > 0:
        time_saved_str = format_time_duration(computation_time_saved)
    else:
        time_saved_str = "0 seconds"
        
    if bytes_saved > 0:
        data_saved_str = format_file_size(bytes_saved)
    else:
        data_saved_str = "0 bytes"
    
    # Format based on output type
    if format == 'markdown':
        return f"""# Smart Cache Statistics

## Cache Utilization
- **Memory Cache:** {memory_size:,} / {memory_max:,} items ({memory_usage_pct:.1f}%)
- **Disk Cache:** {disk_size:,} / {disk_max:,} items ({disk_usage_pct:.1f}%)

## Hit Rates
- **Memory Hit Rate:** {memory_hit_rate:.2f}% ({memory_hits:,} hits, {memory_misses:,} misses)
- **Disk Hit Rate:** {disk_hit_rate:.2f}% ({disk_hits:,} hits, {disk_misses:,} misses)
- **Overall Hit Rate:** {overall_hit_rate:.2f}% ({overall_hits:,} total hits)

## Cache Operations
- **Memory Evictions:** {memory_evictions:,} items
- **Disk Evictions:** {disk_evictions:,} items

## Resource Savings
- **API Calls Saved:** {api_calls_saved:,} calls
- **Computation Time Saved:** {time_saved_str}
- **Data Transfer Saved:** {data_saved_str}

## Efficiency
- **Cache Efficiency Score:** {calculate_cache_efficiency(stats):.1f}/10
"""
    elif format == 'html':
        def hit_rate_class(rate):
            if rate >= 80:
                return "good"
            elif rate >= 50:
                return "average"
            else:
                return "poor"
        
        def efficiency_class(score):
            if score >= 7:
                return "good"
            elif score >= 4:
                return "average"
            else:
                return "poor"
        
        return f"""<!DOCTYPE html>
<html>
<head>
<title>Smart Cache Statistics</title>
<style>
    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
    h1, h2 {{ color: #333; }}
    .stat-group {{ margin-bottom: 25px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
    .stat-row {{ display: flex; margin-bottom: 8px; }}
    .stat-label {{ flex: 1; font-weight: bold; }}
    .stat-value {{ flex: 1; text-align: right; }}
    .good {{ color: green; }}
    .average {{ color: orange; }}
    .poor {{ color: red; }}
    .meter {{ height: 10px; background-color: #eee; border-radius: 5px; margin-top: 5px; }}
    .meter-fill {{ height: 100%; background-color: #4CAF50; border-radius: 5px; }}
    .meter-fill.average {{ background-color: orange; }}
    .meter-fill.poor {{ background-color: red; }}
</style>
</head>
<body>
<h1>Smart Cache Statistics</h1>
<div class="stat-group">
<h2>Cache Utilization</h2>
    <div class="stat-row">
        <div class="stat-label">Memory Cache:</div>
        <div class="stat-value">{memory_size:,} / {memory_max:,} items ({memory_usage_pct:.1f}%)</div>
    </div>
    <div class="meter">
        <div class="meter-fill" style="width: {min(memory_usage_pct, 100)}%;"></div>
    </div>
    <div class="stat-row">
        <div class="stat-label">Disk Cache:</div>
        <div class="stat-value">{disk_size:,} / {disk_max:,} items ({disk_usage_pct:.1f}%)</div>
    </div>
    <div class="meter">
        <div class="meter-fill" style="width: {min(disk_usage_pct, 100)}%;"></div>
    </div>
</div>
<div class="stat-group">
<h2>Hit Rates</h2>
    <div class="stat-row">
        <div class="stat-label">Memory Hit Rate:</div>
        <div class="stat-value {hit_rate_class(memory_hit_rate)}">{memory_hit_rate:.2f}%</div>
    </div>
    <div class="meter">
        <div class="meter-fill {hit_rate_class(memory_hit_rate)}" style="width: {memory_hit_rate}%;"></div>
    </div>
    <div class="stat-row">
        <div class="stat-label">Overall Hit Rate:</div>
        <div class="stat-value {hit_rate_class(overall_hit_rate)}">{overall_hit_rate:.2f}%</div>
    </div>
    <div class="meter">
        <div class="meter-fill {hit_rate_class(overall_hit_rate)}" style="width: {overall_hit_rate}%;"></div>
    </div>
</div>
<div class="stat-group">
<h2>Resource Savings</h2>
    <div class="stat-row">
        <div class="stat-label">API Calls Saved:</div>
        <div class="stat-value">{api_calls_saved:,} calls</div>
    </div>
    <div class="stat-row">
        <div class="stat-label">Computation Time Saved:</div>
        <div class="stat-value">{time_saved_str}</div>
    </div>
    <div class="stat-row">
        <div class="stat-label">Data Transfer Saved:</div>
        <div class="stat-value">{data_saved_str}</div>
    </div>
</div>
<div class="stat-group">
<h2>Efficiency</h2>
    <div class="stat-row">
        <div class="stat-label">Cache Efficiency Score:</div>
        <div class="stat-value {efficiency_class(calculate_cache_efficiency(stats))}">{calculate_cache_efficiency(stats):.1f}/10</div>
    </div>
    <div class="meter">
        <div class="meter-fill {efficiency_class(calculate_cache_efficiency(stats))}" 
             style="width: {calculate_cache_efficiency(stats)*10}%;"></div>
    </div>
</div>
<p><small>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
</body>
</html>
"""
    else:  # Default to text format
        # Create a nicely formatted text report with boxes
        width = 80

        # Header
        text = []
        text.append("┌" + "─" * (width - 2) + "┐")
        text.append("│" + "SMART CACHE STATISTICS".center(width - 2) + "│")
        text.append("├" + "─" * (width - 2) + "┤")
        
        # Cache Utilization
        text.append("│" + " Cache Utilization ".center(width - 2, "─") + "│")
        text.append("│ Memory Cache: " + 
                   f"{memory_size:,} / {memory_max:,} items ({memory_usage_pct:.1f}%)".ljust(width - 16) + "│")
        text.append("│ Disk Cache:   " + 
                   f"{disk_size:,} / {disk_max:,} items ({disk_usage_pct:.1f}%)".ljust(width - 16) + "│")
        
        # Hit Rates
        text.append("│" + " Hit Rates ".center(width - 2, "─") + "│")
        text.append("│ Memory Hit Rate: " + 
                   f"{memory_hit_rate:.2f}% ({memory_hits:,} hits, {memory_misses:,} misses)".ljust(width - 20) + "│")
        text.append("│ Disk Hit Rate:   " + 
                   f"{disk_hit_rate:.2f}% ({disk_hits:,} hits, {disk_misses:,} misses)".ljust(width - 20) + "│")
        text.append("│ Overall Hit Rate: " + 
                   f"{overall_hit_rate:.2f}% ({overall_hits:,} total hits)".ljust(width - 21) + "│")
        
        # Cache Operations
        text.append("│" + " Cache Operations ".center(width - 2, "─") + "│")
        text.append("│ Memory Evictions: " + 
                   f"{memory_evictions:,} items".ljust(width - 21) + "│")
        text.append("│ Disk Evictions:   " + 
                   f"{disk_evictions:,} items".ljust(width - 21) + "│")
        
        # Resource Savings
        text.append("│" + " Resource Savings ".center(width - 2, "─") + "│")
        text.append("│ API Calls Saved:       " + 
                   f"{api_calls_saved:,} calls".ljust(width - 27) + "│")
        text.append("│ Computation Time Saved: " + 
                   f"{time_saved_str}".ljust(width - 28) + "│")
        text.append("│ Data Transfer Saved:    " + 
                   f"{data_saved_str}".ljust(width - 28) + "│")
        
        # Efficiency
        text.append("│" + " Efficiency ".center(width - 2, "─") + "│")
        text.append("│ Cache Efficiency Score: " + 
                   f"{calculate_cache_efficiency(stats):.1f}/10".ljust(width - 28) + "│")
        
        # Footer
        text.append("└" + "─" * (width - 2) + "┘")
        
        # Add interpretation of statistics
        text.append("")
        text.append(interpret_cache_stats(stats))
        
        return "\n".join(text)

def calculate_cache_efficiency(stats):
    """Calculate an overall cache efficiency score (0-10)."""
    # Get hit rates
    memory_hits = stats.get('memory_hits', 0)
    memory_misses = stats.get('memory_misses', 0)
    disk_hits = stats.get('disk_hits', 0)
    disk_misses = stats.get('disk_misses', 0)

    total_requests = memory_hits + memory_misses
    memory_hit_rate = (memory_hits / total_requests) if total_requests > 0 else 0

    overall_hits = memory_hits + disk_hits
    overall_requests = total_requests + disk_misses  # Avoid double counting
    overall_hit_rate = (overall_hits / overall_requests) if overall_requests > 0 else 0

    # Get eviction rates
    memory_evictions = stats.get('memory_evictions', 0)
    eviction_rate = (memory_evictions / (memory_hits + 1)) if memory_hits > 0 else 1

    # Calculate time and API call savings
    api_calls_saved = stats.get('api_calls_saved', 0)
    computation_time_saved = stats.get('computation_time_saved', 0)

    # Compute the score (weighted combination)
    hit_rate_score = overall_hit_rate * 6  # 0-6 points from hit rate
    eviction_score = max(0, 2 - (eviction_rate * 2))  # 0-2 points from low eviction rate

    # Additional points for API savings
    api_score = min(2, api_calls_saved / 50)  # Up to 2 points for API savings

    total_score = hit_rate_score + eviction_score + api_score
    return min(10, total_score)  # Cap at 10

def interpret_cache_stats(stats):
    """Provide a human-readable interpretation of cache statistics."""
    memory_hits = stats.get('memory_hits', 0)
    memory_misses = stats.get('memory_misses', 0)
    disk_hits = stats.get('disk_hits', 0)
    disk_misses = stats.get('disk_misses', 0)

    total_requests = memory_hits + memory_misses
    memory_hit_rate = (memory_hits / total_requests) if total_requests > 0 else 0

    overall_hits = memory_hits + disk_hits
    overall_requests = total_requests + disk_misses  # Avoid double counting
    overall_hit_rate = (overall_hits / overall_requests) if overall_requests > 0 else 0

    api_calls_saved = stats.get('api_calls_saved', 0)
    computation_time_saved = stats.get('computation_time_saved', 0)

    # Generate interpretation
    interpretation = "INTERPRETATION:\n"

    if overall_hit_rate >= 0.8:
        interpretation += "- The cache is performing extremely well with a high hit rate.\n"
    elif overall_hit_rate >= 0.5:
        interpretation += "- The cache is performing adequately but could be improved.\n"
    else:
        interpretation += "- The cache hit rate is low, indicating potential optimization opportunities.\n"

    if memory_hit_rate > 0.8:
        interpretation += "- Memory cache utilization is excellent, providing fast responses.\n"

    if api_calls_saved > 0:
        interpretation += f"- The cache has successfully saved {api_calls_saved} API calls, reducing costs and latency.\n"

    if computation_time_saved > 1:
        time_saved_str = format_time_duration(computation_time_saved)
        interpretation += f"- {time_saved_str} of computation time has been saved through caching.\n"

    # Add recommendations
    interpretation += "\nRECOMMENDATIONS:\n"

    if overall_hit_rate < 0.5:
        interpretation += "- Consider increasing cache TTL values to improve hit rates.\n"
        interpretation += "- Review cache key generation to ensure consistency.\n"

    if memory_hit_rate < 0.3:
        interpretation += "- Consider increasing memory cache size for better performance.\n"

    memory_size = stats.get('memory_size', 0)
    memory_max = stats.get('memory_max_size', 1)
    memory_usage_pct = (memory_size / memory_max) if memory_max > 0 else 0

    if memory_usage_pct > 0.9:
        interpretation += "- Memory cache is nearly full. Consider increasing memory_size to reduce evictions.\n"

    disk_size = stats.get('disk_size', 0)
    disk_max = stats.get('disk_max_size', 1)
    disk_usage_pct = (disk_size / disk_max) if disk_max > 0 else 0

    if disk_usage_pct > 0.9:
        interpretation += "- Disk cache is nearly full. Consider increasing disk_size or clearing infrequently accessed items.\n"

    return interpretation

def format_time_duration(seconds):
    """Format time duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def format_file_size(size_bytes):
    """Format file size in a human-readable way."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        kb = size_bytes / 1024
        return f"{kb:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        mb = size_bytes / (1024 * 1024)
        return f"{mb:.2f} MB"
    else:
        gb = size_bytes / (1024 * 1024 * 1024)
        return f"{gb:.2f} GB"

def analyze_cache_performance(output_file=None):
    """
    Analyze cache performance and provide detailed recommendations.

    Args:
        output_file: Optional file to save analysis report
    """
    if not smart_cache_available:
        print("Error: Smart cache is not available")
        return
        
    try:
        # Get cache statistics
        stats = smart_cache.get_stats()
        
        # Analyze cache performance
        memory_size = stats.get('memory_size', 0)
        memory_max = stats.get('memory_max_size', 1)
        memory_usage_pct = (memory_size / memory_max * 100) if memory_max > 0 else 0
        
        disk_size = stats.get('disk_size', 0)
        disk_max = stats.get('disk_max_size', 1)
        disk_usage_pct = (disk_size / disk_max * 100) if disk_max > 0 else 0
        
        memory_hits = stats.get('memory_hits', 0)
        memory_misses = stats.get('memory_misses', 0)
        disk_hits = stats.get('disk_hits', 0)
        disk_misses = stats.get('disk_misses', 0)
        
        total_requests = memory_hits + memory_misses
        memory_hit_rate = (memory_hits / total_requests * 100) if total_requests > 0 else 0
        
        overall_hits = memory_hits + disk_hits
        overall_requests = total_requests + disk_misses  # Avoid double counting
        overall_hit_rate = (overall_hits / overall_requests * 100) if overall_requests > 0 else 0
        
        memory_evictions = stats.get('memory_evictions', 0)
        disk_evictions = stats.get('disk_evictions', 0)
        
        api_calls_saved = stats.get('api_calls_saved', 0)
        computation_time_saved = stats.get('computation_time_saved', 0)
        
        # Generate analysis
        analysis = []
        analysis.append("## Cache Performance Analysis")
        analysis.append(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        analysis.append("### Current Performance Metrics")
        analysis.append(f"- Memory Cache Usage: {memory_usage_pct:.1f}% ({memory_size:,}/{memory_max:,} items)")
        analysis.append(f"- Disk Cache Usage: {disk_usage_pct:.1f}% ({disk_size:,}/{disk_max:,} items)")
        analysis.append(f"- Overall Hit Rate: {overall_hit_rate:.2f}% ({overall_hits:,} hits from {overall_requests:,} requests)")
        analysis.append(f"- Memory Hit Rate: {memory_hit_rate:.2f}% ({memory_hits:,} hits from {total_requests:,} memory requests)")
        analysis.append(f"- API Calls Saved: {api_calls_saved:,}")
        analysis.append(f"- Computation Time Saved: {format_time_duration(computation_time_saved)}")
        
        # Performance assessment
        analysis.append("\n### Performance Assessment")
        
        if overall_hit_rate >= 80:
            analysis.append("- **Excellent Hit Rate**: The cache is performing very well with a hit rate above 80%.")
        elif overall_hit_rate >= 60:
            analysis.append("- **Good Hit Rate**: The cache is performing well with a hit rate above 60%.")
        elif overall_hit_rate >= 40:
            analysis.append("- **Average Hit Rate**: The cache has a moderate hit rate that could be improved.")
        else:
            analysis.append("- **Poor Hit Rate**: The cache has a low hit rate, indicating potential issues.")
        
        if memory_hit_rate >= 70:
            analysis.append("- **Efficient Memory Usage**: Memory cache is effectively serving requests.")
        else:
            analysis.append("- **Suboptimal Memory Usage**: Memory cache hit rate is lower than expected.")
        
        if memory_usage_pct >= 90:
            analysis.append("- **Memory Cache Pressure**: Memory cache is nearly full, which may lead to increased evictions.")
        elif memory_usage_pct <= 20:
            analysis.append("- **Underutilized Memory Cache**: Memory cache has significant unused capacity.")
        
        if disk_usage_pct >= 90:
            analysis.append("- **Disk Cache Pressure**: Disk cache is nearly full, which may lead to increased evictions.")
        
        if memory_evictions > memory_hits * 0.2:
            analysis.append("- **High Eviction Rate**: Many items are being evicted before they can be reused.")
        
        # Efficiency estimation
        efficiency_score = calculate_cache_efficiency(stats)
        analysis.append(f"\n### Overall Efficiency Score: {efficiency_score:.1f}/10")
        
        if efficiency_score >= 8:
            analysis.append("- **Excellent**: Cache is performing exceptionally well.")
        elif efficiency_score >= 6:
            analysis.append("- **Good**: Cache is performing well but has room for improvement.")
        elif efficiency_score >= 4:
            analysis.append("- **Average**: Cache is providing some benefit but could be significantly improved.")
        else:
            analysis.append("- **Poor**: Cache effectiveness is limited and needs optimization.")
        
        # Recommendations
        analysis.append("\n### Recommendations")
        
        if overall_hit_rate < 50:
            analysis.append("1. **Increase TTL Values**: Consider increasing the time-to-live for cached items to improve hit rates.")
            analysis.append("   - Current hit rate suggests items may be expiring before they can be reused.")
            analysis.append("   - Try increasing the TTL for embedding and API caches by 2-3x.")
        
        if memory_hit_rate < 40:
            analysis.append("2. **Increase Memory Cache Size**: Consider increasing the memory cache size to keep more items in fast memory.")
            analysis.append(f"   - Current memory cache limited to {memory_max:,} items, which may be insufficient.")
            analysis.append(f"   - Recommended size: {int(memory_max * 1.5):,} items.")
        
        if memory_evictions > memory_hits * 0.2:
            analysis.append("3. **Review Cache Key Generation**: High eviction rates suggest potential issues with cache key generation.")
            analysis.append("   - Ensure consistent key generation for the same inputs.")
            analysis.append("   - Check for unnecessary variation in cache keys.")
        
        if disk_usage_pct > 90:
            analysis.append("4. **Increase Disk Cache Size or Clear Old Entries**: Disk cache is nearly full.")
            analysis.append(f"   - Current disk cache limited to {disk_max:,} items.")
            analysis.append(f"   - Recommended size: {int(disk_max * 1.5):,} items.")
            analysis.append("   - Alternatively, run a cache cleanup to remove old entries.")
        
        if api_calls_saved < 10 and overall_requests > 100:
            analysis.append("5. **Review API Caching Strategy**: Few API calls are being saved.")
            analysis.append("   - Ensure API functions are properly decorated with @cached_api.")
            analysis.append("   - Consider increasing API cache TTL for stable endpoints.")
        
        # Additional general recommendations
        analysis.append("\n### General Optimization Tips")
        analysis.append("- **Batch Similar Requests**: Group similar requests to improve cache efficiency.")
        analysis.append("- **Normalize Inputs**: Standardize inputs before caching to increase hit rates.")
        analysis.append("- **Preemptive Caching**: For predictable workflows, consider preloading the cache.")
        analysis.append("- **Monitor Cache Growth**: Regularly check cache size and performance metrics.")
        analysis.append("- **Adjust TTL Strategically**: Use longer TTL for stable data and shorter TTL for volatile data.")
        
        # Join analysis
        report = "\n".join(analysis)
        
        # Output report
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Cache performance analysis saved to {output_file}")
        else:
            print(report)
            
    except Exception as e:
        print(f"Error analyzing cache performance: {e}")
        import traceback
        print(traceback.format_exc())

def analyze_translation_with_alignment(source_text, translation, use_groq=True, detailed=True, custom_weights=None, config_path=None, segment_type='sentence'):
    """
    Analyze translation with segment alignment detection.
    
    Args:
        source_text: Source text
        translation: Translation to analyze
        use_groq: Whether to use Groq for enhanced evaluation
        detailed: Whether to get detailed analysis
        custom_weights: Custom weights for the analysis
        config_path: Path to a configuration file
        segment_type: Segmentation granularity used for alignment analysis
        
    Returns:
        Analysis results with alignment information
    """
    from translation_quality_analyzer import TranslationQualityAnalyzer
    
    groq_evaluator = None
    if use_groq:
        try:
            from groq_client import GroqClient
            from groq_evaluator import GroqTranslationEvaluator
            groq_client = GroqClient()
            groq_evaluator = GroqTranslationEvaluator(client=groq_client)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not initialize Groq for demo: {e}")
            use_groq = False
    
    # Initialize analyzer
    analyzer = TranslationQualityAnalyzer(
        groq_evaluator=groq_evaluator
    )
    
    # Analyze with alignment detection
    results = analyzer.analyze_pair(
        source_text=source_text,
        translation=translation,
        use_groq=use_groq,
        detailed=detailed,
        detect_weak_alignments=True,
        segment_type=segment_type,
        custom_weights=custom_weights,
        config_path=config_path
    )
    
    return results

def compare_translations_with_alignment(source_text, translations, use_groq=True):
    """
    Compare multiple translations with segment alignment detection.
    
    Args:
        source_text: Source text
        translations: List of translations to compare
        use_groq: Whether to use Groq for enhanced evaluation
        
    Returns:
        Comparison results with alignment information
    """
    from translation_quality_analyzer import TranslationQualityAnalyzer

    groq_evaluator = None
    if use_groq:
        try:
            from groq_client import GroqClient
            from groq_evaluator import GroqTranslationEvaluator
            groq_client = GroqClient()
            groq_evaluator = GroqTranslationEvaluator(client=groq_client)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not initialize Groq for demo: {e}")
            use_groq = False
    
    # Initialize analyzer
    analyzer = TranslationQualityAnalyzer(
        groq_evaluator=groq_evaluator
    )
    
    # Rank translations with alignment detection
    results = analyzer.rank_candidates(
        source_text=source_text,
        candidates=translations,
        use_groq=use_groq,
        detect_weak_alignments=True
    )
    
    return results

def run_alignment_demo(console):
    """Runs a self-contained demo of the segment alignment functionality."""
    from segment_alignment import generate_alignment_report
    
    console.print(Panel("[bold cyan]Running Segment Alignment Analysis Demo[/bold cyan]"))

    source_text = """
    The new regulations on carbon emissions will take effect next quarter. 
    Companies must comply with these standards or face substantial penalties.
    Small businesses with fewer than 50 employees may apply for a temporary exemption.
    """
    
    translation = """
    Las nuevas regulaciones sobre emisiones de carbono entrarán en vigor el próximo trimestre.
    Las empresas deben cumplir con estos estándares o enfrentar sanciones sustanciales.
    Las pequeñas empresas con menos de 50 trabajadores pueden solicitar una excepción temporal.
    """
    
    bad_translation = """
    Las nuevas reglas sobre emisiones comenzarán pronto.
    Empresas deben cumplir o pagar multas.
    Negocios pequeños pueden pedir tiempo extra.
    """
    
    console.print("\n[bold]1. Analyzing a single good translation...[/bold]")
    # Analyze a single translation with alignment detection
    results = analyze_translation_with_alignment(
        source_text=source_text,
        translation=translation,
        use_groq=True,
        detailed=True
    )
    
    # Generate report
    report = generate_alignment_report(results)
    console.print(report)
    
    console.print("\n[bold]2. Comparing multiple translations (one good, one weak)...[/bold]")
    # Compare multiple translations
    comparison_results = compare_translations_with_alignment(
        source_text=source_text,
        translations=[translation, bad_translation],
        use_groq=True
    )
    
    # Print summary of comparison
    console.print("\n[bold]TRANSLATION COMPARISON RESULTS:[/bold]")
    console.print("=" * 50)
    
    if "ranked_translations" in comparison_results:
        for i, trans in enumerate(comparison_results["ranked_translations"]):
            console.print(f"Rank {i+1}: Composite Score {trans.get('quality_score', 0):.2f}")
            if "metrics" in trans and "alignment_analysis" in trans["metrics"]:
                alignment = trans["metrics"]["alignment_analysis"]
                summary = alignment.get("enhanced_summary") or alignment.get("weak_alignment_summary", {})
                
                console.print(f"  Alignment: {summary.get('severity_level', 'N/A').upper()}")
                console.print(f"  Main Finding: {summary.get('main_finding', 'N/A')}")
            console.print("")
    return 0

def run_weight_demo(console):
    """Runs a demo of using custom weights for analysis."""
    console.print(Panel("[bold cyan]Running Custom Weights Analysis Demo[/bold cyan]"))

    source_text = """
    The new regulations on carbon emissions will take effect next quarter. 
    Companies must comply with these standards or face substantial penalties.
    Small businesses with fewer than 50 employees may apply for a temporary exemption.
    """
    
    translation = """
    Las nuevas regulaciones sobre emisiones de carbono entrarán en vigor el próximo trimestre.
    Las empresas deben cumplir con estos estándares o enfrentar sanciones sustanciales.
    Las pequeñas empresas con menos de 50 trabajadores pueden solicitar una excepción temporal.
    """
    
    # Analyze with default weights
    console.print("\n[bold]1. Analyzing with default weights...[/bold]")
    default_results = analyze_translation_with_alignment(
        source_text,
        translation,
        use_groq=True,
        detailed=True
    )
    console.print(f"Composite Score (default weights): [bold green]{default_results.get('composite_score', 0):.4f}[/bold green]")
    
    # Define custom weights
    custom_weights = {
        "embedding_similarity": 0.8,      # Emphasize embedding similarity
        "alignment_score": 0.1,           # De-emphasize alignment score
        "groq_score": 0.1,                # De-emphasize Groq's simple score
        "embedding_metrics_weight": 1.5,  # Boost the embedding metric group
        "alignment_metrics_weight": 0.5,  # Lower the alignment metric group
        "groq_simple_metrics_weight": 0.5 # Lower the Groq metric group
    }
    
    console.print("\n[bold]2. Analyzing with custom weights...[/bold]")
    console.print("[dim]Weights used:[/dim]")
    for k, v in custom_weights.items():
        console.print(f"[dim]  - {k}: {v}[/dim]")
        
    custom_results = analyze_translation_with_alignment(
        source_text,
        translation,
        use_groq=True,
        detailed=True,
        custom_weights=custom_weights
    )
    console.print(f"Composite Score (custom weights): [bold yellow]{custom_results.get('composite_score', 0):.4f}[/bold yellow]")
    
    return 0

async def async_main():
    """Enhanced async main function for the translation evaluator."""
    global performance_monitor
    
    start_time = time.time()
    
    # Parse arguments
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Initialize async evaluator
    evaluator = AsyncTranslationEvaluator(
        config_path=getattr(args, 'config', None),
        interactive=getattr(args, 'interactive', False)
    )
    
    try:
        # Handle special commands first
        if args.version:
            show_version()
            return 0
        
        if args.list_languages:
            list_supported_languages()
            return 0
        
        if args.clear_cache:
            clear_all_cache(evaluator.config)
            return 0
        
        # Handle async-specific commands
        if args.demo:
            await evaluator.initialize()
            await evaluator.demo_translation_evaluation()
            return 0
        
        if args.benchmark:
            await evaluator.initialize()
            return await run_benchmark(evaluator, args.benchmark_iterations, args.interactive)
        
        if args.batch_file:
            await evaluator.initialize()
            result = await evaluator.process_file(args.batch_file)
            if result.get("status") == "error":
                logger.error(f"Batch processing failed: {result.get('error')}")
                return 1
            console.print(f"Processed {result.get('items_processed')} items")
            console.print(f"Results written to {result.get('output_path')}")
            return 0
        
        if args.batch_directory:
            await evaluator.initialize()
            results = await evaluator.process_directory(args.batch_directory, args.file_pattern)
            successful = sum(1 for r in results if r.get("status") == "success")
            total = len(results)
            console.print(f"Directory processing completed: {successful}/{total} files successful")
            return 0 if successful == total else 1
        
        # Handle cache statistics (existing functionality)
        if args.cache_stats and not any([
            args.source_text, args.source_file, args.target_text, 
            args.target_file, args.cache_analyze
        ]):
            return handle_cache_stats(args)
        
        if args.cache_analyze:
            return handle_cache_analysis(args)
        
        # Check if required arguments are provided for translation analysis
        if not (args.source_text or args.source_file):
            if args.interactive:
                source_text = get_interactive_input("Enter source text:")
            else:
                console.print("[bold red]Error:[/bold red] No source text provided. Use --source-text, --source-file, or --interactive.")
                return 1
        else:
            source_text = await read_source_text(args)
        
        if not (args.target_text or args.target_file):
            if args.interactive:
                target_text = get_interactive_input("Enter target text:")
            else:
                console.print("[bold red]Error:[/bold red] No target text provided. Use --target-text, --target-file, or --interactive.")
                return 1
        else:
            target_text = await read_target_text(args)
        
        # Get language codes
        source_lang = args.source_lang or 'en'
        target_lang = args.target_lang or 'es'
        
        if args.interactive and (not args.source_lang or not args.target_lang):
            if not args.source_lang:
                source_lang = Prompt.ask("Enter source language code", default="en")
            if not args.target_lang:
                target_lang = Prompt.ask("Enter target language code", default="es")
        
        # Initialize evaluator and run analysis
        await evaluator.initialize()
        
        with console.status("[bold green]Running translation analysis..."):
            results = await evaluator.analyze_translation(
                source_text,
                target_text,
                source_lang,
                target_lang,
                include_error_analysis=getattr(args, 'detailed_report', False)
            )
        
        # Generate and output report
        report = generate_report(results, args.format)
        
        if args.output:
            await write_output_file(args.output, report)
            if not args.quiet:
                console.print(f"[green]Analysis report saved to {args.output}[/green]")
        else:
            console.print(report)
        
        # Show performance summary if monitoring enabled
        if args.monitor or args.performance_report:
            show_performance_summary(evaluator)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if args.verbose:
            console.print(Panel(traceback.format_exc(), title="Error Details", border_style="red"))
        return 1
    finally:
        await evaluator.close()


async def read_source_text(args) -> str:
    """Read source text from file or argument."""
    if args.source_text:
        return args.source_text
    
    if HAS_ASYNC_LIBS:
        async with aiofiles.open(args.source_file, 'r', encoding='utf-8') as f:
            return await f.read()
    else:
        return Path(args.source_file).read_text(encoding='utf-8')


async def read_target_text(args) -> str:
    """Read target text from file or argument."""
    if args.target_text:
        return args.target_text
    
    if HAS_ASYNC_LIBS:
        async with aiofiles.open(args.target_file, 'r', encoding='utf-8') as f:
            return await f.read()
    else:
        return Path(args.target_file).read_text(encoding='utf-8')


async def write_output_file(file_path: str, content: str) -> None:
    """Write content to output file."""
    if HAS_ASYNC_LIBS:
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
    else:
        Path(file_path).write_text(content, encoding='utf-8')


def generate_report(results: Dict[str, Any], format_type: str) -> str:
    """Generate a report from analysis results."""
    if format_type == 'json':
        return json.dumps(results, indent=2, default=str)
    elif format_type == 'html':
        return output_html(results, "Translation Analysis Report")
    elif format_type == 'markdown':
        return generate_markdown_report(results)
    else:  # text format
        return generate_text_report(results)


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate a markdown report from results."""
    report = "# Translation Analysis Report\n\n"
    
    if 'source_text' in results:
        report += f"**Source Text:** {results['source_text']}\n\n"
    if 'translated_text' in results:
        report += f"**Translation:** {results['translated_text']}\n\n"
    
    report += "## Scores\n\n"
    if 'embedding_similarity' in results:
        report += f"- **Embedding Similarity:** {results['embedding_similarity']:.4f}\n"
    if 'groq_quality_score' in results:
        report += f"- **Groq Quality Score:** {results['groq_quality_score']:.2f}/10\n"
    if 'combined_score' in results:
        report += f"- **Combined Score:** {results['combined_score']:.4f}\n"
    
    # Add Groq analysis if available
    if 'groq_analysis' in results:
        analysis = results['groq_analysis']
        report += "\n## Detailed Analysis\n\n"
        if analysis.get('analysis'):
            report += f"{analysis['analysis']}\n\n"
        
        if analysis.get('strengths'):
            report += "### Strengths\n"
            for strength in analysis['strengths']:
                report += f"- {strength}\n"
            report += "\n"
        
        if analysis.get('weaknesses'):
            report += "### Areas for Improvement\n"
            for weakness in analysis['weaknesses']:
                report += f"- {weakness}\n"
            report += "\n"
    
    return report


def generate_text_report(results: Dict[str, Any]) -> str:
    """Generate a text report from results."""
    report = "Translation Analysis Report\n"
    report += "=" * 30 + "\n\n"
    
    if 'embedding_similarity' in results:
        report += f"Embedding Similarity: {results['embedding_similarity']:.4f}\n"
    if 'groq_quality_score' in results:
        report += f"Groq Quality Score: {results['groq_quality_score']:.2f}/10\n"
    if 'combined_score' in results:
        report += f"Combined Score: {results['combined_score']:.4f}\n"
    
    return report


async def run_benchmark(evaluator, iterations: int, interactive: bool) -> int:
    """Run performance benchmark."""
    sample_data = [
        {
            "source_text": "The quick brown fox jumps over the lazy dog.",
            "translated_text": "El zorro marrón rápido salta sobre el perro perezoso.",
            "source_lang": "en",
            "target_lang": "es"
        }
    ] * iterations
    
    if interactive:
        console.print(f"[bold]Running benchmark with {iterations} iterations...[/bold]")
    
    start_time = time.time()
    
    # Run benchmark
    results = []
    for item in sample_data:
        result = await evaluator.analyze_translation(
            item["source_text"],
            item["translated_text"], 
            item["source_lang"],
            item["target_lang"]
        )
        results.append(result)
    
    end_time = time.time()
    duration = end_time - start_time
    successful = sum(1 for r in results if "error" not in r)
    
    # Display results
    benchmark_results = {
        "total_items": len(sample_data),
        "successful": successful,
        "failed": len(results) - successful,
        "total_time_seconds": duration,
        "items_per_second": len(sample_data) / duration,
        "avg_time_per_item_seconds": duration / len(sample_data)
    }
    
    if interactive:
        bench_table = Table(show_header=True, header_style="bold")
        bench_table.add_column("Metric")
        bench_table.add_column("Value")
        
        bench_table.add_row("Total Items", str(benchmark_results["total_items"]))
        bench_table.add_row("Successful", str(benchmark_results["successful"]))
        bench_table.add_row("Failed", str(benchmark_results["failed"]))
        bench_table.add_row("Total Time", f"{benchmark_results['total_time_seconds']:.3f} seconds")
        bench_table.add_row("Items/Second", f"{benchmark_results['items_per_second']:.2f}")
        bench_table.add_row("Avg Time/Item", f"{benchmark_results['avg_time_per_item_seconds']:.4f} seconds")
        
        console.print("\n[bold]Benchmark Results:[/bold]")
        console.print(bench_table)
    else:
        print(json.dumps(benchmark_results, indent=2))
    
    return 0


def handle_cache_stats(args) -> int:
    """Handle cache statistics display."""
    if not smart_cache_available:
        console.print("[bold red]Error:[/bold red] Smart cache is not available in this installation.")
        return 1
    
    try:
        cache_stats = smart_cache.get_stats()
        formatted_stats = format_cache_stats(cache_stats, args.cache_stats_format)
        
        if args.cache_stats_output:
            with open(args.cache_stats_output, 'w', encoding='utf-8') as f:
                f.write(formatted_stats)
            console.print(f"Cache statistics saved to {args.cache_stats_output}")
        else:
            console.print(formatted_stats)
        
        return 0
    except Exception as e:
        console.print(f"[bold red]Error getting cache statistics:[/bold red] {e}")
        return 1


def handle_cache_analysis(args) -> int:
    """Handle cache performance analysis."""
    if not smart_cache_available:
        console.print("[bold red]Error:[/bold red] Smart cache is not available in this installation.")
        return 1
    
    try:
        analyze_cache_performance(args.cache_stats_output)
        return 0
    except Exception as e:
        console.print(f"[bold red]Error analyzing cache performance:[/bold red] {e}")
        return 1


def show_performance_summary(evaluator):
    """Show performance summary if available."""
    if evaluator.perf_data:
        console.print("\n[bold]Performance Summary:[/bold]")
        for operation, data in evaluator.perf_data.items():
            console.print(f"- {operation}: {data['count']} calls, avg {data['avg_time']:.3f}s")


def main():
    """Entry point that sets up and runs the async event loop."""
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Setup signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(
                shutdown(loop, sig)
            ))
        
        # Run async_main and get exit code
        exit_code = loop.run_until_complete(async_main())
        
        # Exit with the returned code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)


async def shutdown(loop, signal=None):
    """Cleanup tasks tied to the service's shutdown."""
    if signal:
        logger.info(f"Received exit signal {signal.name}")
    
    logger.info("Cleaning up resources...")
    
    # Wait 1 second to give tasks a chance to complete
    await asyncio.sleep(1)
    
    # Cancel all tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for task in tasks:
        task.cancel()
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    loop.stop()

# -----------------------------------------------------------------------------
# Candidate ranking CLI helper (lightweight – used by unit tests)
# -----------------------------------------------------------------------------

def rank_translations_cli(args) -> int:
    """Light-weight wrapper around ``calculate_translation_confidence`` used by the
    unit tests in *tests/test_cli.py*.

    The function expects an *args* object (e.g. an ``argparse.Namespace``) to
    expose the following attributes (defaults mimic the test-fixture):
        source_text            – direct source string, or
        source_file            – path to text file with the source string
        candidates             – comma-separated candidate translations, or
        candidates_file        – path to a file containing one candidate per line
        model                  – name of sentence-embedding model (optional)
        confidence_method      – distribution | gap | range
        output_format          – json | yaml | csv | table | html
        include_diagnostics    – boolean flag
        output_file            – optional path to write the results

    On success the function prints the requested representation (unless writing
    to *output_file*) and returns **0**. Any error is printed to *stderr* and the
    function returns **1** so that the calling test can handle a non-zero exit
    code if desired.
    """
    import json
    import sys
    from pathlib import Path
    # Optional dependency only required for YAML output
    try:
        import yaml  # type: ignore
        _HAS_YAML = True
    except Exception:
        _HAS_YAML = False

    # ------------------------------------------------------------------
    # Resolve source text
    # ------------------------------------------------------------------
    try:
        if getattr(args, "source_text", None):
            source_text = str(args.source_text).strip()
        elif getattr(args, "source_file", None):
            source_path = Path(args.source_file)
            if not source_path.exists():
                print(f"Error: source file not found – {source_path}", file=sys.stderr)
                return 1
            source_text = source_path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            print("Error: either --source-text or --source-file must be provided.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Error reading source text: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Resolve candidate translations
    # ------------------------------------------------------------------
    candidates: list[str] = []
    try:
        if getattr(args, "candidates_file", None):
            cand_path = Path(args.candidates_file)
            if not cand_path.exists():
                print(f"Error: candidates file not found – {cand_path}", file=sys.stderr)
                return 1
            with cand_path.open("r", encoding="utf-8", errors="replace") as fh:
                candidates = [ln.strip() for ln in fh if ln.strip()]
        elif getattr(args, "candidates", None):
            candidates = [c.strip() for c in str(args.candidates).split(",") if c.strip()]
        if not candidates:
            print("Error: at least one candidate translation must be supplied.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Error reading candidate translations: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Run ranking/ confidence calculation
    # ------------------------------------------------------------------
    try:
        results = calculate_translation_confidence(
            source_text=source_text,
            candidates=candidates,
            model_name=getattr(args, "model", "all-MiniLM-L6-v2"),
            confidence_method=getattr(args, "confidence_method", "distribution"),
            include_diagnostics=bool(getattr(args, "include_diagnostics", False)),
        )
    except Exception as exc:
        print(f"Error during ranking: {exc}", file=sys.stderr)
        return 1

    payload: dict[str, object] = {"source_text": source_text, **results}

    # ------------------------------------------------------------------
    # Serialise output in the requested format
    # ------------------------------------------------------------------
    fmt = str(getattr(args, "output_format", "json")).lower()
    output_str: str = ""

    if fmt == "json":
        output_str = json.dumps(payload, ensure_ascii=False)
    elif fmt == "yaml":
        if not _HAS_YAML:
            print("Error: PyYAML not available but --output-format yaml requested.", file=sys.stderr)
            return 1
        output_str = yaml.safe_dump(payload, allow_unicode=True)
    elif fmt == "csv":
        import csv
        import io
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["translation", "similarity", "confidence"])
        writer.writeheader()
        for item in payload.get("ranked_translations", []):
            writer.writerow({
                "translation": item.get("translation", ""),
                "similarity": item.get("similarity", 0),
                "confidence": item.get("confidence", 0),
            })
        output_str = buffer.getvalue()
    elif fmt == "table":
        # Rich table to stdout (no capture in tests for this format)
        try:
            from rich.table import Table
            from rich.console import Console
            _console = Console()
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", justify="right")
            table.add_column("Similarity", justify="right")
            table.add_column("Confidence", justify="right")
            table.add_column("Translation")
            for idx, item in enumerate(payload.get("ranked_translations", []), start=1):
                table.add_row(
                    str(idx),
                    f"{item.get('similarity', 0):.4f}",
                    f"{item.get('confidence', 0):.4f}",
                    item.get("translation", ""),
                )
            _console.print(table)
            output_str = ""  # Already printed via rich
        except Exception as exc:
            print(f"Error generating table output: {exc}", file=sys.stderr)
            return 1
    elif fmt == "html":
        output_str = output_html(payload, title="Candidate Ranking", include_diagnostics=bool(getattr(args, "include_diagnostics", False)))
    else:
        print(f"Error: unknown output format '{fmt}'.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Persist/ print results
    # ------------------------------------------------------------------
    try:
        if getattr(args, "output_file", None):
            out_path = Path(args.output_file)
            out_path.write_text(output_str, encoding="utf-8")
        else:
            if output_str:
                print(output_str)
        return 0
    except Exception as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if performance_monitor:
            # Try to save performance data even if there's an error
            try:
                performance_monitor.save_statistics("error_stats.json")
                performance_monitor.cleanup()
            except:
                pass
        
        if smart_cache_available:
            # Ensure cache is properly shutdown
            try:
                smart_cache.shutdown()
            except:
                pass
                
        logging.critical(f"Unhandled exception in main: {e}\n{traceback.format_exc()}")
        print(f"Critical error: {str(e)}")
        sys.exit(1)