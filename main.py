#!/usr/bin/env python3
"""
Smart CLI Translation Quality Analyzer
A command-line tool to analyze and evaluate translation quality.
"""

import argparse
import sys
import os
from pathlib import Path

from config_manager import ConfigManager

def main():
    """Main entry point for the CLI application."""
    # Load configuration
    config = ConfigManager()
    
    parser = argparse.ArgumentParser(
        description="Smart CLI Translation Quality Analyzer - Evaluate translation quality between languages"
    )
    
    # Required arguments
    parser.add_argument("--source", "-s", type=str, required=True, 
                        help="Source text to analyze")
    parser.add_argument("--translation", "-t", type=str, required=True, 
                        help="Translated text to evaluate")
    
    # Optional arguments
    parser.add_argument("--source-lang", type=str, 
                        help="Source language code (auto-detect if not specified)")
    parser.add_argument("--target-lang", type=str, 
                        help="Target language code (auto-detect if not specified)")
    parser.add_argument("--model", "-m", type=str,
                        help=f"Embedding model to use (default: {config.get_model_path()})")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose output")
    parser.add_argument("--output", "-o", type=str, 
                        help="Output path for analysis results (default: stdout)")
    parser.add_argument("--config", "-c", type=str,
                        help="Path to custom configuration file")
    parser.add_argument("--set-api-key", type=str, nargs=2, metavar=("SERVICE", "KEY"),
                        help="Set API key for specified service (groq or huggingface)")
    
    args = parser.parse_args()
    
    # Handle config operations
    if args.config:
        config = ConfigManager(args.config)
    
    if args.set_api_key:
        service, key = args.set_api_key
        if service.lower() in ["groq", "huggingface"]:
            config.set(f"api.{service.lower()}.api_key", key)
            print(f"API key for {service} has been updated.")
            sys.exit(0)
        else:
            print(f"Unknown service: {service}. Supported services: groq, huggingface")
            sys.exit(1)
    
    # Basic validation
    if not args.source or not args.translation:
        parser.print_help()
        sys.exit(1)
    
    # Will add actual analysis functionality in the future
    print("Source text:", args.source)
    print("Translation:", args.translation)
    print("Analysis functionality will be implemented in future updates.")
    print(f"Using configuration from: {config.config_path}")

if __name__ == "__main__":
    main() 