"""
Configuration Manager for the Smart CLI Translation Quality Analyzer
Handles settings for model paths, cache, API endpoints, and credentials
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manages application configuration including model paths, API credentials,
    cache directories and other settings through JSON configuration files.
    """
    # Default configuration values
    DEFAULT_CONFIG = {
        "models": {
            "embedding": {
                "default": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "alternatives": [
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "sentence-transformers/distiluse-base-multilingual-cased-v1"
                ],
                "cache_dir": "~/.tqa/models"
            },
            "translator": {
                "default": "Helsinki-NLP/opus-mt-en-ROMANCE",
                "alternatives": [
                    "facebook/m2m100_418M",
                    "facebook/nllb-200-distilled-600M"
                ]
            }
        },
        "api": {
            "groq": {
                "endpoint": "https://api.groq.com/v1",
                "model": "mixtral-8x7b-32768",
                "api_key": "",
                "timeout": 30
            },
            "huggingface": {
                "endpoint": "https://api-inference.huggingface.co/models",
                "api_key": "",
                "timeout": 15
            }
        },
        "cache": {
            "enabled": True,
            "directory": "~/.tqa/cache",
            "max_size_mb": 1000,
            "ttl_days": 30
        },
        "analysis": {
            "similarity_threshold": 0.75,
            "min_quality_score": 0.6,
            "detailed_reports": True
        },
        "ui": {
            "colors": True,
            "progress_bars": True,
            "verbose_logging": False
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager with optional custom config path.
        
        Args:
            config_path: Path to custom config file (defaults to ~/.tqa/config.json)
        """
        self.logger = logging.getLogger("tqa.config")
        
        # Set up default paths
        self.user_home = Path.home()
        self.config_dir = Path(os.path.expanduser("~/.tqa"))
        
        # Create config directory if it doesn't exist
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created configuration directory: {self.config_dir}")
        
        # Determine config file path
        self.config_path = Path(config_path) if config_path else self.config_dir / "config.json"
                
        # Load configuration
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from file, creating default if not exists."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    user_config = json.load(file)
                    # Update default config with user settings
                    self._deep_update(self.config, user_config)
                    self.logger.info(f"Configuration loaded from {self.config_path}")
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error loading configuration: {e}")
                self.logger.info("Using default configuration")
        else:
            self.save_config()
            self.logger.info(f"Created default configuration at {self.config_path}")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Create parent directories if they don't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as file:
                json.dump(self.config, file, indent=2)
            
            # Set restrictive permissions (only user can read/write)
            if os.name == 'posix':
                os.chmod(self.config_path, 0o600)
                
            self.logger.info(f"Configuration saved to {self.config_path}")
        except IOError as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'api.groq.endpoint')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any, save: bool = True) -> None:
        """
        Set configuration value using dot notation path.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'api.groq.api_key')
            value: Value to set
            save: Whether to save config to file after update
        """
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the appropriate nested dictionary
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
        
        # Save if requested
        if save:
            self.save_config()
    
    def get_model_path(self, model_type: str = "embedding") -> str:
        """
        Get the path for the specified model type.
        
        Args:
            model_type: Type of model (embedding, translator, etc.)
            
        Returns:
            Model path or identifier
        """
        return self.get(f"models.{model_type}.default")
    
    def get_cache_dir(self) -> str:
        """Get the expanded cache directory path."""
        cache_dir = self.get("cache.directory")
        return os.path.expanduser(cache_dir)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for specified service.
        
        Args:
            service: Service name (groq, huggingface)
            
        Returns:
            API key or None if not set
        """
        # Try to get from config
        api_key = self.get(f"api.{service}.api_key")
        
        # If not in config, try environment variable
        if not api_key:
            env_var = f"{service.upper()}_API_KEY"
            api_key = os.environ.get(env_var)
            
            # If found in environment, update config but don't save to file
            if api_key:
                self.set(f"api.{service}.api_key", api_key, save=False)
                
        return api_key
    
    def is_api_configured(self, service: str) -> bool:
        """Check if the specified API service is properly configured."""
        return bool(self.get_api_key(service))
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Recursively update nested dictionaries.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value 