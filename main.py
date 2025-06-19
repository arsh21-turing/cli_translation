#!/usr/bin/env python3
"""
Smart CLI Translation Quality Analyzer
Main entry point for the command-line interface
"""

import sys
import os
import argparse
import logging
import time
import html as _html_mod
import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.box import Box
from rich.progress import Progress

# Import our components
from config_manager import ConfigManager
from model_loader import ModelLoader, ModelType, InferenceMode, MultilingualModelManager
from text_processor import TextProcessor
from embedding_generator import MultilingualEmbeddingGenerator
from similarity_calculator import SimilarityCalculator
from language_utils import LanguageDetector, EmbeddingBasedLanguageDetector, get_supported_languages
from analyzer import TranslationQualityAnalyzer

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger('tqa')

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
            from groq_evaluator import GroqEvaluator
            groq_client = GroqClient()
            groq_evaluator = GroqEvaluator(client=groq_client)
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
            from groq_evaluator import GroqEvaluator
            groq_client = GroqClient()
            groq_evaluator = GroqEvaluator(client=groq_client)
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

def main():
    """Main entry point for the CLI tool."""
    # Set up argument parser
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Create console for rich output early
    console = Console(color_system="auto" if args.color == "auto" else 
                      (True if args.color == "always" else False))

    if args.run_alignment_demo:
        return run_alignment_demo(console)

    if args.run_weight_demo:
        return run_weight_demo(console)

    # Handle candidate-ranking shortcut early
    if getattr(args, 'rank_candidates', False):
        return rank_translations_cli(args)
    
    # Configure logging level
    if args.debug:
        logging.getLogger('tqa').setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger('tqa').setLevel(logging.ERROR)
    
    # Handle version display
    if args.version:
        show_version()
        return 0
    
    # Initialize configuration
    config = ConfigManager(config_path=args.config)
    
    # Override config with command line options
    if args.api_key:
        config.set_api_key("default", args.api_key)
    if args.cache_dir:
        config.set("cache.directory", args.cache_dir)
    if args.no_cache:
        config.set("models.embedding.use_cache", False)
    if args.inference_mode:
        config.set("inference_mode", args.inference_mode)
    if args.embedding_model:
        config.set("models.embedding.default", args.embedding_model)
    if args.multilingual_model:
        config.set("models.multilingual.type", args.multilingual_model)
    
    # Handle cache clearing
    if args.clear_cache:
        clear_all_cache(config)
        return 0
    
    # Initialize base components
    model_loader = ModelLoader(
        config=config,
        inference_mode=args.inference_mode or config.get("inference_mode", "hybrid")
    )
    
    # Initialize multilingual model manager
    multilingual_model_manager = MultilingualModelManager(config, model_loader)
    
    # Initialize the text processor (required by the analyzer)
    text_processor = TextProcessor()
    
    # Handle language listing - now uses MultilingualModelManager 
    if args.list_languages:
        # We've already defined a function for this
        list_supported_languages()
        return 0
    
    # Initialize the analyzer with all components explicitly
    analyzer = TranslationQualityAnalyzer(
        config=config,
        model_loader=multilingual_model_manager,
        text_processor=text_processor
    )
    
    # Get input text
    source_text = None
    target_text = None
    
    # Interactive mode has the highest precedence
    if args.interactive:
        console.print(Panel(
            "Enter the source text and translated text interactively.",
            title="Interactive Mode",
            border_style="cyan"
        ))
        
        if args.translate:
            # For translation, we only need source text and target language
            source_text = get_interactive_input("Enter text to translate:")
            if not args.target_lang:
                args.target_lang = get_interactive_input("Enter target language code (e.g., es for Spanish):", multiline=False)
        else:
            source_text = get_interactive_input("Enter source text:")
            if args.rate_translation:
                # For rating, we need both texts
                target_text = get_interactive_input("Enter translated text:")
            elif not args.translate:
                # Default mode needs both texts
                target_text = get_interactive_input("Enter translated text:")
    else:
        # Command-line text has precedence over files
        if args.source_text:
            source_text = args.source_text
        elif args.source_file:
            try:
                source_text = read_file_text(args.source_file)
            except Exception as e:
                console.print(f"[bold red]Error reading source file:[/bold red] {e}")
                return 1
        
        if args.target_text:
            target_text = args.target_text
        elif args.target_file:
            try:
                target_text = read_file_text(args.target_file)
            except Exception as e:
                console.print(f"[bold red]Error reading target file:[/bold red] {e}")
                return 1
    
    # Handle translation
    if args.translate:
        # Check if we have source text
        if not source_text:
            console.print("[bold red]Error:[/bold red] Source text is required for translation.")
            console.print("[dim]Use --source-text or --source-file to provide input.[/dim]")
            return 1
            
        # Check if we have target language
        if not args.target_lang:
            console.print("[bold red]Error:[/bold red] Target language is required for translation.")
            console.print("[dim]Use --target-lang to specify the language to translate into.[/dim]")
            return 1
            
        # Auto-detect source language if not provided
        if not args.source_lang:
            detection_result = analyzer.detect_language_advanced(source_text)
            if isinstance(detection_result, dict):
                args.source_lang = detection_result['language']
                console.print(f"[yellow]Auto-detected source language:[/yellow] {args.source_lang}")
            else:
                args.source_lang = detection_result
                console.print(f"[yellow]Auto-detected source language:[/yellow] {args.source_lang}")
        
        # Show translation status
        console.print(Panel(
            f"Translating from {args.source_lang} to {args.target_lang}...",
            title="Translation In Progress",
            border_style="yellow"
        ))
        
        # Handle batch translation
        if args.translate_batch:
            # Split into lines and translate each
            lines = source_text.strip().split('\n')
            translations = []
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Translating...", total=len(lines))
                
                for line in lines:
                    if line.strip():
                        # Use the multilingual model manager to get the appropriate model
                        translation = analyzer.translate_text(
                            line.strip(),
                            source_lang=args.source_lang,
                            target_lang=args.target_lang,
                            model_name=args.translation_model
                        )
                        translations.append(translation)
                    else:
                        translations.append("")
                    progress.update(task, advance=1)
            
            # Join translations
            result = "\n".join(translations)
            
            # Save to file if requested
            if args.batch_output:
                if write_to_file(args.batch_output, result):
                    console.print(f"[green]Translations saved to:[/green] {args.batch_output}")
                else:
                    console.print(f"[red]Failed to save translations to:[/red] {args.batch_output}")
            
            # Display first few translations
            table = Table(title=f"Batch Translation Results ({len(lines)} lines)")
            table.add_column("Source", style="cyan", no_wrap=False)
            table.add_column("Translation", style="green", no_wrap=False)
            
            # Show at most 5 examples
            for i, (src, tgt) in enumerate(zip(lines[:5], translations[:5])):
                # Truncate very long segments
                src_display = (src[:80] + "...") if len(src) > 80 else src
                tgt_display = (tgt[:80] + "...") if len(tgt) > 80 else tgt
                table.add_row(src_display, tgt_display)
                
            console.print(table)
            
            if len(lines) > 5:
                console.print(f"[dim]...and {len(lines) - 5} more lines[/dim]")
        else:
            # Single translation - use multilingual model manager to get the appropriate model
            translation = analyzer.translate_text(
                source_text,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                model_name=args.translation_model
            )
            
            # Display translation
            panel = Panel(
                translation,
                title=f"Translation from {args.source_lang} to {args.target_lang}",
                border_style="green"
            )
            console.print(panel)
            
            # Save to file if requested
            if args.output:
                if write_to_file(args.output, translation):
                    console.print(f"[green]Translation saved to:[/green] {args.output}")
                else:
                    console.print(f"[red]Failed to save translation to:[/red] {args.output}")
            
            # Set as target text for possible rating
            target_text = translation
        
        # After translation, if rate-translation is also specified, continue to rating
        if not args.rate_translation:
            return 0
    
    # Handle language detection
    if args.detect_language:
        # Check if we have input text
        if not source_text and not args.source_text and not args.source_file:
            # Prompt for text if none provided
            source_text = get_interactive_input("Enter text to detect language:")
        
        # Use source text for language detection if we have it
        text_to_analyze = source_text if source_text else ""
        
        # Perform language detection
        detection_result = analyzer.detect_language_advanced(
            text_to_analyze,
            fast_mode=args.fast_detection,
            detailed=args.verbose
        )
        
        # Create rich console for output
        if isinstance(detection_result, dict):
            # Detailed result
            lang_code = detection_result['language']
            confidence = detection_result['confidence'] * 100  # Convert to percentage
            lang_name = detection_result.get('language_name', 'Unknown')
            
            # Color based on confidence
            if confidence >= 85:
                conf_color = "green"
            elif confidence >= 60:
                conf_color = "yellow"
            else:
                conf_color = "red"
            
            # Basic panel with language info
            panel_content = [
                f"[bold]Detected Language:[/bold] {lang_name} ({lang_code})",
                f"[bold]Confidence:[/bold] [{conf_color}]{confidence:.1f}%[/{conf_color}]",
                f"[bold]Detection Method:[/bold] {detection_result['method'].capitalize()}"
            ]
            
            # Add multilingual info if available
            if detection_result.get('is_multilingual'):
                panel_content.append("\n[bold]Multilingual Content Detected[/bold]")
                
                # Add composition details
                if 'language_composition' in detection_result:
                    composition = detection_result['language_composition']
                    top_languages = sorted(composition.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    panel_content.append("[bold]Language Composition:[/bold]")
                    for lang, percentage in top_languages:
                        try:
                            import pycountry
                            lang_obj = pycountry.languages.get(alpha_2=lang)
                            lang_name = lang_obj.name if lang_obj else lang
                        except (AttributeError, KeyError, ImportError):
                            lang_name = lang
                            
                        panel_content.append(f"  {lang_name}: {percentage*100:.1f}%")
            
            # Create and display the panel
            panel = Panel(
                "\n".join(panel_content),
                title="Language Detection Results",
                border_style=conf_color
            )
            console.print(panel)
            
            # Show detailed script information for verbose mode
            if args.verbose and 'scripts' in detection_result:
                scripts = detection_result['scripts']
                
                if scripts:
                    # Create a table for scripts
                    table = Table(title="Script Composition")
                    table.add_column("Script", style="cyan")
                    table.add_column("Proportion", style="magenta")
                    
                    # Add rows for each script (sorted by proportion)
                    for script, prop in sorted(scripts.items(), key=lambda x: x[1], reverse=True):
                        if prop > 0.01:  # Only show scripts with at least 1%
                            table.add_row(
                                script,
                                f"{prop*100:.1f}%"
                            )
                    
                    console.print(table)
            
            # Show all candidate languages for verbose mode
            if args.verbose and 'all_scores' in detection_result:
                scores = detection_result['all_scores']
                
                if len(scores) > 1:
                    # Create a table for candidate languages
                    table = Table(title="Language Candidates")
                    table.add_column("Language", style="cyan")
                    table.add_column("Score", style="magenta")
                    
                    # Add rows for each language (sorted by score)
                    top_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    for lang, score in top_langs:
                        try:
                            import pycountry
                            lang_obj = pycountry.languages.get(alpha_2=lang)
                            lang_name = lang_obj.name if lang_obj else lang
                        except (AttributeError, KeyError, ImportError):
                            lang_name = lang
                            
                        table.add_row(
                            f"{lang_name} ({lang})",
                            f"{score*100:.1f}%"
                        )
                    
                    console.print(table)
        else:
            # Simple result (just language code)
            try:
                import pycountry
                lang_obj = pycountry.languages.get(alpha_2=detection_result)
                lang_name = lang_obj.name if lang_obj else "Unknown"
            except (AttributeError, KeyError, ImportError):
                lang_name = "Unknown"
                
            console.print(f"Detected language: {lang_name} ({detection_result})")
        
        return 0
        
    # Handle text composition analysis
    elif args.analyze_composition:
        # Check if we have input text
        if not source_text and not args.source_text and not args.source_file:
            # Prompt for text if none provided
            source_text = get_interactive_input("Enter text to analyze composition:")
        
        # Use source text for composition analysis
        text_to_analyze = source_text if source_text else ""
        
        # Perform composition analysis
        analysis = analyzer.analyze_text_composition(text_to_analyze)
        
        # Display main results
        panel_content = [
            f"[bold]Primary Language:[/bold] {analysis['primary_language']}",
            f"[bold]Number of Languages:[/bold] {analysis['language_count']}",
            f"[bold]Is Multilingual:[/bold] {'Yes' if analysis['is_multilingual'] else 'No'}"
        ]
        
        panel = Panel(
            "\n".join(panel_content),
            title="Text Composition Analysis",
            border_style="cyan"
        )
        console.print(panel)
        
        # Display language composition
        if analysis['composition']:
            # Create table for language composition
            table = Table(title="Language Composition")
            table.add_column("Language", style="cyan")
            table.add_column("Percentage", style="magenta")
            
            # Add rows for each language
            for lang, percentage in sorted(
                analysis['composition'].items(), key=lambda x: x[1], reverse=True
            ):
                try:
                    import pycountry
                    lang_obj = pycountry.languages.get(alpha_2=lang)
                    lang_name = lang_obj.name if lang_obj else lang
                except (AttributeError, KeyError, ImportError):
                    lang_name = lang
                    
                table.add_row(
                    f"{lang_name} ({lang})",
                    f"{percentage*100:.1f}%"
                )
            
            console.print(table)
            
        # Display segments for verbose mode
        if args.verbose and analysis['segments']:
            console.print("\n[bold]Text Segments by Language:[/bold]\n")
            
            for i, segment in enumerate(analysis['segments']):
                lang = segment['language']
                try:
                    import pycountry
                    lang_obj = pycountry.languages.get(alpha_2=lang)
                    lang_name = f"{lang_obj.name} ({lang})" if lang_obj else lang
                except (AttributeError, KeyError, ImportError):
                    lang_name = lang
                    
                # Color based on confidence
                confidence = segment['confidence'] * 100
                if confidence >= 85:
                    color = "green"
                elif confidence >= 60:
                    color = "yellow"
                else:
                    color = "red"
                    
                # Display segment with its language
                console.print(f"[{color}]Segment {i+1} - {lang_name} ({confidence:.1f}%):[/{color}]")
                # Limit segment display to avoid overwhelming the console
                text = segment['text']
                if len(text) > 100:
                    text = text[:97] + "..."
                console.print(text)
                console.print("")  # Empty line
                
        return 0
    
    # Handle translation rating
    if args.rate_translation:
        # Check if we have both texts
        if not source_text or not target_text:
            console.print("[bold red]Error:[/bold red] Both source and target texts are required for rating.")
            return 1
            
        # Auto-detect languages if not provided
        if not args.source_lang:
            detection = analyzer.detect_language_advanced(source_text)
            if isinstance(detection, dict):
                args.source_lang = detection['language']
            else:
                args.source_lang = detection
            console.print(f"[yellow]Auto-detected source language:[/yellow] {args.source_lang}")
            
        if not args.target_lang:
            detection = analyzer.detect_language_advanced(target_text)
            if isinstance(detection, dict):
                args.target_lang = detection['language']
            else:
                args.target_lang = detection
            console.print(f"[yellow]Auto-detected target language:[/yellow] {args.target_lang}")
        
        # Rate the translation
        console.print(Panel(
            f"Rating translation quality from {args.source_lang} to {args.target_lang}...",
            title="Translation Rating",
            border_style="yellow"
        ))
        
        # Perform quick rating
        # Use multilingual_model_manager for language-specific models
        rating = analyzer.analyze(
            source_text,
            target_text,
            source_lang=args.source_lang,
            target_lang=args.target_lang
        )
        
        # Display rating results
        score = rating.quality_score * 100  # Convert to percentage
        
        # Color based on score
        if score >= 80:
            color = "green"
        elif score >= 60:
            color = "yellow"
        else:
            color = "red"
            
        # Create rating panel
        panel_content = [
            f"[bold]Overall Rating:[/bold] [bold {color}]{score:.1f}%[/bold {color}]",
        ]
        
        # Add subscores if available
        if rating.fluency_score:
            fluency = rating.fluency_score * 100
            fluency_color = "green" if fluency >= 80 else "yellow" if fluency >= 60 else "red"
            panel_content.append(f"[bold]Fluency:[/bold] [{fluency_color}]{fluency:.1f}%[/{fluency_color}]")
            
        if rating.accuracy_score:
            accuracy = rating.accuracy_score * 100
            accuracy_color = "green" if accuracy >= 80 else "yellow" if accuracy >= 60 else "red"
            panel_content.append(f"[bold]Accuracy:[/bold] [{accuracy_color}]{accuracy:.1f}%[/{accuracy_color}]")
            
        if rating.detailed_feedback:
            panel_content.append(f"\n[bold]Feedback:[/bold] {rating.detailed_feedback}")
        
        panel = Panel(
            "\n".join(panel_content),
            title="Translation Rating Results",
            border_style=color
        )
        
        console.print(panel)
        
        return 0
    
    # Check if we have required input for analysis
    if not source_text or not target_text:
        # If either text is missing and we're not in a special mode, show error
        if source_text and not target_text:
            console.print("[bold yellow]Warning:[/bold yellow] Target text is missing. Please provide translated text.")
        elif not source_text and target_text:
            console.print("[bold yellow]Warning:[/bold yellow] Source text is missing. Please provide source text.")
        else:
            # Both missing, but we might be in a special mode
            if not args.detect_language and not args.analyze_composition:
                console.print("[bold red]Error:[/bold red] Both source and target texts are required.")
                console.print("[dim]Use --source-text/--target-text or --source-file/--target-file to provide input.[/dim]")
                console.print("[dim]Or use --interactive for guided input.[/dim]")
        
        # Show command line help
        console.print("\n[bold cyan]Command-line arguments:[/bold cyan]")
        parser.print_help()
        return 1
    
    # Perform analysis based on selected mode
    if args.similarity:
        # Perform semantic similarity analysis
        # Use MultilingualModelManager for language-specific models
        results = analyzer.analyze_semantic_similarity(
            source_text,
            target_text,
            metric=args.metric,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            segmented=args.segmented,
            classification=args.classify
        )
        
        # Overall similarity panel
        avg_similarity = results["average_similarity"] * 100
        source_lang = results["source_language"]
        target_lang = results["target_language"]
        
        # Color code based on similarity
        if avg_similarity >= 80:
            color = "green"
        elif avg_similarity >= 60:
            color = "yellow"
        else:
            color = "red"
        
        # Include classification if requested
        classification_info = ""
        if args.classify and "overall_match_class" in results:
            match_class = results["overall_match_class"]
            classification_info = f"\nMatch class: [bold]{match_class.upper()}[/bold]"
        
        panel = Panel(
            f"[bold]Semantic Similarity:[/bold] [bold {color}]{avg_similarity:.1f}%[/bold {color}]\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Metric: {results['metric']}"
            f"{classification_info}",
            title="Similarity Analysis Results",
            border_style=color
        )
        
        console.print(panel)
        
        # Display segment analysis if verbose or multi-segment
        if (args.verbose or len(results["segment_analysis"]) > 1) and not args.segmented:
            # Create table for segment comparison
            table = Table(title="Segment-by-Segment Comparison", box=Box.ROUNDED)
            table.add_column("Source", style="cyan", no_wrap=False)
            table.add_column("Target", style="green", no_wrap=False)
            table.add_column("Similarity", style="magenta")
            
            if args.classify:
                table.add_column("Match", style="yellow")
                
            # Add rows for segments (limit to top 10 if there are many)
            segments_to_show = results["segment_analysis"]
            if len(segments_to_show) > 10 and not args.verbose:
                console.print(f"[yellow]Showing 10 of {len(segments_to_show)} segments. Use --verbose to see all.[/yellow]")
                segments_to_show = segments_to_show[:10]
                
            for segment in segments_to_show:
                # Truncate very long segments for display
                source_display = (segment["source"][:80] + "...") if len(segment["source"]) > 80 else segment["source"]
                target_display = (segment["target"][:80] + "...") if len(segment["target"]) > 80 else segment["target"]
                
                sim_value = segment["similarity"] * 100
                sim_color = "green" if sim_value >= 80 else "yellow" if sim_value >= 60 else "red"
                similarity_cell = f"[{sim_color}]{sim_value:.1f}%[/{sim_color}]"
                
                if args.classify and "match_class" in segment:
                    table.add_row(
                        source_display, 
                        target_display,
                        similarity_cell,
                        segment["match_class"].upper()
                    )
                else:
                    table.add_row(
                        source_display, 
                        target_display,
                        similarity_cell
                    )
            
            console.print(table)
            
        # Display performance info
        if "analysis_time" in results:
            console.print(f"[dim]Analysis completed in {results['analysis_time']:.2f} seconds[/dim]")
    
    elif args.cross_lingual:
        # Perform cross-lingual analysis
        # Use MultilingualModelManager for language-specific models
        results = analyzer.analyze_cross_lingual_similarity(
            source_text,
            target_text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            similarity_metric=args.metric,
            preprocessing_level=args.preprocessing,
            detailed=args.verbose
        )
        
        # Basic results panel
        avg_similarity = results["average_similarity"] * 100
        source_lang = results["source_language"]
        target_lang = results["target_language"]
        
        # Color code based on similarity
        if avg_similarity >= 80:
            color = "green"
        elif avg_similarity >= 60:
            color = "yellow"
        else:
            color = "red"
        
        panel = Panel(
            f"[bold]Cross-lingual Similarity:[/bold] [bold {color}]{avg_similarity:.1f}%[/bold {color}]\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Metric: {results['metric']}",
            title="Translation Analysis Results",
            border_style=color
        )
        
        console.print(panel)
        
        # Detailed results if verbose
        if args.verbose and "alignment_confidence" in results:
            confidence = results["alignment_confidence"] * 100
            
            detail_panel = Panel(
                f"[bold]Alignment confidence:[/bold] {confidence:.1f}%\n"
                f"Number of mutual best matches: {len(results['mutual_best_matches'])}\n"
                f"Total segments analyzed: {len(results['similarity_scores'])}",
                title="Detailed Analysis",
                border_style="cyan"
            )
            
            console.print(detail_panel)
            
            # Display a heatmap visualization of the similarity matrix if not too large
            if len(results["similarity_scores"]) <= 20 and "similarity_matrix" in results:
                console.print("[bold]Similarity Matrix:[/bold]")
                matrix = results["similarity_matrix"]
                
                # Create a simple ASCII heatmap
                for row in matrix:
                    cells = []
                    for val in row:
                        # Convert similarity to a color intensity
                        intensity = min(int(val * 9), 8)  # 0-8 scale
                        cells.append(f"[color(231)][[/color(231)][color({232+intensity*3})]{'█' * intensity}[/color({232+intensity*3})][color(231)]][/color(231)]")
                    console.print(" ".join(cells))
    
    else:
        if hasattr(args, 'weak_alignment') and args.weak_alignment:
            # Perform composite analysis with optional weak alignment detection
            use_groq_flag = (True if args.use_groq is None else args.use_groq)
            results = analyze_translation_with_alignment(
                source_text=source_text,
                translation=target_text,
                use_groq=use_groq_flag,
                detailed=args.detailed_report,
                segment_type=args.segment_type,
                config_path=args.config
            )
            composite = results.get('composite_score', 0) * 100
            # Colour based on composite score
            if composite >= 80:
                color = "green"
            elif composite >= 60:
                color = "yellow"
            else:
                color = "red"
            panel_lines = [
                f"[bold]Composite Quality Score:[/bold] [bold {color}]{composite:.1f}%[/bold {color}]"
            ]
            # If alignment summary available, show its headline
            alignment = results.get('alignment_analysis', {})
            summary = alignment.get('enhanced_summary') or alignment.get('weak_alignment_summary') or {}
            if summary:
                sev = summary.get('severity_level', 'n/a').upper()
                finding = summary.get('main_finding', 'No summary')
                panel_lines.append(f"[bold]Alignment Severity:[/bold] {sev}")
                panel_lines.append(f"[bold]Key Finding:[/bold] {finding}")
            panel = Panel("\n".join(panel_lines), title="Translation Quality & Alignment", border_style=color)
            console.print(panel)
        else:
            # Default to standard translation quality analysis
            # Use MultilingualModelManager for language-specific models
            results = analyzer.analyze(
                source_text,
                target_text,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                detailed=args.detailed_report
            )
            
            # Display results
            quality_score = results.quality_score * 100  # Convert to percentage
            
            # Color based on quality score
            if quality_score >= 80:
                color = "green"
            elif quality_score >= 60:
                color = "yellow"
            else:
                color = "red"
            
            panel = Panel(
                f"[bold]Translation Quality Score:[/bold] [bold {color}]{quality_score:.1f}%[/bold {color}]\n"
                f"Source language: {results.source_lang}\n"
                f"Target language: {results.target_lang}\n"
                f"Fluency score: {results.fluency_score*100:.1f}%\n"
                f"Adequacy score: {results.accuracy_score*100:.1f}%",
                title="Translation Quality Analysis",
                border_style=color
            )
            
            console.print(panel)
            
            # Show detailed report if requested
            if args.detailed_report and hasattr(results, "segment_scores"):
                # Create table for segment scores
                table = Table(title="Segment Quality Scores")
                table.add_column("Source", style="cyan", no_wrap=False)
                table.add_column("Translation", style="green", no_wrap=False)
                table.add_column("Quality", style="magenta")
                
                # Add rows for segments (limit to 10 if there are many)
                segments = results.segment_scores
                if len(segments) > 10 and not args.verbose:
                    console.print(f"[yellow]Showing 10 of {len(segments)} segments. Use --verbose to see all.[/yellow]")
                    segments = segments[:10]
                    
                for segment in segments:
                    # Truncate very long segments for display
                    source = (segment.source[:80] + "...") if len(segment.source) > 80 else segment.source
                    target = (segment.target[:80] + "...") if len(segment.target) > 80 else segment.target
                    score = segment.score * 100
                    
                    # Color code based on score
                    score_color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
                    
                    table.add_row(
                        source,
                        target,
                        f"[{score_color}]{score:.1f}%[/{score_color}]"
                    )
                    
                console.print(table)
    
    return 0

# -----------------------------------------------------------------------------
# Candidate ranking helpers (stand-alone to keep existing logic untouched)
# -----------------------------------------------------------------------------


def _read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as fh:
        return fh.read().strip()


def _read_candidates_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as fh:
        return [line.strip() for line in fh if line.strip()]


def _print_table(ranked: List[Dict[str, Any]], source: Optional[str] = None, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    from tabulate import tabulate  # local import to keep fast startup

    headers = ["Rank", "Similarity", "Confidence", "Translation"]
    rows = [
        [i + 1, f"{item['similarity']:.4f}", f"{item['confidence']:.4f}", item['translation' if 'translation' in item else 'text']]  # type: ignore
        for i, item in enumerate(ranked)
    ]
    if source:
        print(f"Source: {source}\n")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Diagnostics summary
    if diagnostics:
        print("\n=== Clustering Diagnostics ===")
        print(f"Optimal clusters: {diagnostics.get('optimal_clusters')}")
        print(f"Cluster sizes: {diagnostics.get('cluster_sizes')}")
        print(f"Within-cluster cohesion: {diagnostics.get('cluster_cohesion'):.4f}")
        print(f"Between-cluster separation: {diagnostics.get('cluster_separation'):.4f}")
        print(f"Variance explained: {diagnostics.get('variance_explained'):.4f}")

def _output_json(result: Dict[str, Any], source: Optional[str]) -> str:
    import json

    payload = {"source_text": source or "", **result}
    return json.dumps(payload, indent=2, ensure_ascii=False)

def _output_csv(result: Dict[str, Any], source: Optional[str]) -> str:
    import csv
    from io import StringIO

    ranked = result.get("ranked_translations", [])
    diagnostics = result.get("diagnostics")

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Rank", "Similarity", "Confidence", "Translation"])
    if source:
        writer.writerow(["# Source", source])
        writer.writerow([])
    for idx, item in enumerate(ranked):
        writer.writerow([idx + 1, f"{item['similarity']:.4f}", f"{item['confidence']:.4f}", item['translation' if 'translation' in item else 'text']])  # type: ignore

    if diagnostics:
        writer.writerow([])
        writer.writerow(["# Clustering Diagnostics"])
        writer.writerow(["Optimal clusters", diagnostics.get('optimal_clusters')])
        writer.writerow(["Cluster sizes", diagnostics.get('cluster_sizes')])
        writer.writerow(["Within-cluster cohesion", f"{diagnostics.get('cluster_cohesion'):.4f}"])
        writer.writerow(["Between-cluster separation", f"{diagnostics.get('cluster_separation'):.4f}"])
        writer.writerow(["Variance explained", f"{diagnostics.get('variance_explained'):.4f}"])

    return buf.getvalue()

def _output_yaml(result: Dict[str, Any], source: Optional[str]) -> str:
    payload = {"source_text": source or "", **result}
    return yaml.dump(payload, sort_keys=False, allow_unicode=True)

def _output_html(result: Dict[str, Any], source: Optional[str], include_diag: bool) -> str:
    payload = {"source_text": source or "", **result}
    return output_html(payload, include_diagnostics=include_diag)

def rank_translations_cli(args) -> int:
    """Standalone CLI handler for --rank-candidates."""
    # Source text
    if args.source_file:
        try:
            source_text = _read_text_file(args.source_file)
        except FileNotFoundError:
            print(f"[ERROR] Source file not found: {args.source_file}")
            return 1
    elif args.source_text:
        source_text = args.source_text
    else:
        print("[ERROR] Source text is required (use --source-text or --source-file).")
        return 1

    # Candidate list
    candidates: List[str] = []
    if args.candidates_file:
        try:
            candidates.extend(_read_candidates_file(args.candidates_file))
        except FileNotFoundError:
            print(f"[ERROR] Candidates file not found: {args.candidates_file}")
            return 1
    if args.candidates:
        candidates.extend([c.strip() for c in args.candidates.split(',') if c.strip()])

    if not candidates:
        print("[ERROR] No candidate translations provided (use --candidates or --candidates-file).")
        return 1

    result = calculate_translation_confidence(
        source_text,
        candidates,
        model_name=args.model,
        confidence_method=args.confidence_method,
        include_diagnostics=args.include_diagnostics,
    )

    ranked = result["ranked_translations"]

    fmt = args.output_format.lower()
    if fmt == "json":
        content = _output_json(result, source_text)
    elif fmt == "yaml":
        content = _output_yaml(result, source_text)
    elif fmt == "csv":
        content = _output_csv(result, source_text)
    elif fmt == "html":
        content = _output_html(result, source_text, args.include_diagnostics)
    else:
        _print_table(ranked, source_text, result.get("diagnostics") if args.include_diagnostics else None)
        content = None

    # Save or print
    if content is not None:
        if args.output_file:
            try:
                with open(args.output_file, "w", encoding="utf-8") as fh:
                    fh.write(content)
                print(f"Results written to {args.output_file}")
            except Exception as exc:
                print(f"[ERROR] Could not write to {args.output_file}: {exc}")
                return 1
        else:
            print(content)

    return 0

if __name__ == "__main__":
    sys.exit(main())