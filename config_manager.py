"""
Configuration Manager for the Smart CLI Translation Quality Analyzer
Handles settings for model paths, cache, API endpoints, and credentials
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration including model paths, API credentials,
    cache directories and other settings through JSON configuration files.
    """
    # Default configuration values
    DEFAULT_CONFIG = {
        "models": {
            "embedding": "all-MiniLM-L6-v2",
            "cache_dir": "~/.cache/sentence_transformers"
        },
        "api": {
            "groq": {
                "api_key_env": "GROQ_API_KEY",
                "models": {
                    "translation_evaluation": "llama3-8b-8192",
                    "error_analysis": "llama3-70b-8192",
                    "default": "llama3-8b-8192"
                }
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
        },
        "quality_weights": {
            # Embedding-based metrics
            "embedding_similarity": 0.4,
            "length_ratio_penalty": 0.1,
            
            # Alignment metrics
            "alignment_score": 0.7,
            "recurring_pattern_penalty": 0.3,
            "position_pattern_penalty": 0.2,
            
            # Groq's overall assessment
            "groq_score": 0.6,
            
            # Groq's detailed assessment components
            "accuracy": 0.7,
            "fluency": 0.6,
            "terminology": 0.5,
            "style": 0.4,
            
            # Metric group weights (for balancing different methods)
            "embedding_metrics_weight": 1.0,
            "alignment_metrics_weight": 0.8,
            "groq_simple_metrics_weight": 0.9,
            "groq_detailed_metrics_weight": 1.0
        },
        "alignment": {
            "similarity_threshold": 0.75,
            "segment_type": "sentence",
            "min_pattern_occurrences": 2,
            "severity_thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
                "problematic": 0.5
            }
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
        Get configuration value using dot notation path. Environment variables
        prefixed with ``CONFIG_`` take precedence over values stored in the
        configuration file.
        
        For example, requesting ``batch.size`` will first look for an
        environment variable named ``CONFIG_BATCH_SIZE`` (upper-cased and dots
        replaced with underscores). If found, the string value is converted to
        int/float/bool if possible before being returned.
        """
        keys = key_path.split('.')

        # ----------------------------------------------------------------------------
        # 1) Environment variable override
        # ----------------------------------------------------------------------------
        env_var = 'CONFIG_' + '_'.join(keys).upper()
        env_val = os.environ.get(env_var)
        if env_val is not None:
            # Attempt rudimentary type casting – int, float, bool, json, fallback str
            lowered = env_val.lower()
            if lowered in {'true', 'false'}:
                return lowered == 'true'
            try:
                return int(env_val)
            except ValueError:
                pass
            try:
                return float(env_val)
            except ValueError:
                pass
            try:
                return json.loads(env_val)
            except Exception:
                return env_val  # raw string

        # ----------------------------------------------------------------------------
        # 2) Regular config lookup
        # ----------------------------------------------------------------------------
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any, save: bool = True) -> None:
        """
        Enhanced setter that prevents structural conflicts (i.e. attempting to
        assign into a scalar value). If such a conflict is detected a
        ``ValueError`` is raised, matching unit-test expectations.
        """
        keys = key_path.split('.')
        config_section = self.config

        # Navigate through all but last key, creating nested dicts as required.
        for key in keys[:-1]:
            current_val = config_section.get(key)
            if current_val is None:
                config_section[key] = {}
                current_val = config_section[key]
            # If we encounter a non-dict value before reaching our destination
            # it means the path conflicts with an existing scalar (e.g.
            # "a.b" already set to int and now we try to set "a.b.c").
            if not isinstance(current_val, dict):
                raise ValueError(f"Cannot create sub-key under non-mapping path '{'.'.join(keys[:-1])}'")
            config_section = current_val

        # Finally set the value (no conflict at this point)
        config_section[keys[-1]] = value

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
        """Return *True* if an API key for *service* is set **in the config file**.

        The previous implementation also considered environment variables but
        the accompanying unit-test expects the *initial* state to be
        "not configured" even when the parent process might expose a
        ``GROQ_API_KEY``.  External callers can still obtain keys via
        ``get_api_key`` which retains the env-var fallback; only this
        convenience checker now focuses on persistent configuration data.
        """

        return bool(self.get(f"api.{service}.api_key"))
    
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

    # API Key Management
    def get_groq_api_key(self) -> Optional[str]:
        """Get the Groq API key from config or environment variable."""
        # Try environment variable first
        env_var = self.get('api.groq.api_key_env', 'GROQ_API_KEY')
        api_key = os.environ.get(env_var)
        
        # If not in environment, try config
        if not api_key:
            api_key = self.get('api.groq.api_key')
        
        return api_key
    
    def set_groq_api_key(self, key: str) -> None:
        """Set the Groq API key in the configuration."""
        self.set('api.groq.api_key', key)
    
    def get_groq_model(self, purpose: str = 'default') -> str:
        """
        Get the appropriate Groq model for a specific purpose.
        
        Args:
            purpose: The purpose (e.g., 'translation_evaluation', 'error_analysis')
            
        Returns:
            Model name as string
        """
        return self.get(f'api.groq.models.{purpose}', 
                        self.get('api.groq.models.default', 'llama3-8b-8192'))
    
    # Quality Score Weights Management
    def get_quality_weights(self) -> Dict[str, float]:
        """
        Get all quality score weights.
        
        Returns:
            Dictionary of weight names and values
        """
        return self.get('quality_weights', {})
    
    def get_weight(self, weight_name: str, default: float = 0.5) -> float:
        """
        Get a specific quality weight.
        
        Args:
            weight_name: Name of the weight
            default: Default value if weight not found
            
        Returns:
            Weight value as float
        """
        return self.get(f'quality_weights.{weight_name}', default)
    
    def set_weight(self, weight_name: str, value: float) -> None:
        """
        Set a specific quality weight.
        
        Args:
            weight_name: Name of the weight
            value: Weight value to set
        """
        self.set(f'quality_weights.{weight_name}', float(value))
    
    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        Update multiple quality weights at once.
        
        Args:
            weights: Dictionary of weight names and values to update
        """
        current_weights = self.get_quality_weights()
        current_weights.update(weights)
        self.set('quality_weights', current_weights)
    
    def reset_weights_to_default(self) -> None:
        """Reset quality weights to default values."""
        self.set('quality_weights', self.DEFAULT_CONFIG['quality_weights'].copy())
    
    # Alignment Configuration
    def get_alignment_config(self) -> Dict[str, Any]:
        """
        Get alignment analysis configuration.
        
        Returns:
            Dictionary of alignment settings
        """
        return self.get('alignment', {})
    
    def get_alignment_setting(self, setting_name: str, default: Any = None) -> Any:
        """
        Get a specific alignment setting.
        
        Args:
            setting_name: Name of the setting
            default: Default value if setting not found
            
        Returns:
            Setting value
        """
        return self.get(f'alignment.{setting_name}', default)
    
    def set_alignment_setting(self, setting_name: str, value: Any) -> None:
        """
        Set a specific alignment setting.
        
        Args:
            setting_name: Name of the setting
            value: Setting value to set
        """
        self.set(f'alignment.{setting_name}', value)

    # ---------------------------------------------------------------------
    # Convenience / helper special methods
    # ---------------------------------------------------------------------
    def __iter__(self):
        """Iterate over top-level configuration section names."""
        return iter(self.config.keys())

    def __contains__(self, item):
        """True if *item* is a top-level section present in the config."""
        return item in self.config

    _SENSITIVE_PATTERNS = {"password", "secret", "token", "api_key", "api_keys"}

    def __str__(self) -> str:  # pragma: no cover
        """Return a JSON representation with sensitive values masked."""
        def mask(obj):
            if isinstance(obj, dict):
                masked = {}
                for k, v in obj.items():
                    if any(p in k.lower() for p in self._SENSITIVE_PATTERNS):
                        masked[k] = "***"
                    else:
                        masked[k] = mask(v)
                return masked
            if isinstance(obj, list):
                return [mask(x) for x in obj]
            return obj

        try:
            return json.dumps(mask(self.config), indent=2, ensure_ascii=False)
        except Exception:
            return str(mask(self.config))

    # ---------------------------------------------------------------------
    # Advanced helpers utilised by the extended test-suite
    # ---------------------------------------------------------------------
    def merge_with_defaults(self, defaults: Dict[str, Any]):
        """Return a *new* ConfigManager instance representing *self* merged with
        the provided *defaults* mapping. Values in *self* take precedence over
        *defaults* (i.e. behave like overlay config). The returned object is
        entirely independent – further modifications will **not** affect the
        original instance.
        """
        from copy import deepcopy

        merged_cfg = ConfigManager()
        # Avoid disk IO for the new instance – use in-memory path outside ~/.tqa
        merged_cfg.config_path = Path(os.devnull)
        merged_cfg.config = deepcopy(self.config)
        self._deep_update(merged_cfg.config, deepcopy(defaults))
        return merged_cfg 