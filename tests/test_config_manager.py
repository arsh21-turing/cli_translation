"""
Unit tests for ConfigManager class
Tests configuration loading, saving, access, and API key management
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

# Import ConfigManager
from config_manager import ConfigManager

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # Cleanup after tests
    shutil.rmtree(tmp_dir)

@pytest.fixture
def default_config():
    """Return a copy of the default configuration."""
    return {
        "models": {
            "embedding": {
                "default": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "alternatives": [
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                ],
                "cache_dir": "~/.tqa/models"
            }
        },
        "api": {
            "groq": {
                "endpoint": "https://api.groq.com/v1",
                "api_key": ""
            },
            "huggingface": {
                "endpoint": "https://api-inference.huggingface.co/models",
                "api_key": ""
            }
        },
        "cache": {
            "enabled": True,
            "directory": "~/.tqa/cache",
            "max_size_mb": 1000
        }
    }

@pytest.fixture
def custom_config():
    """Return a custom configuration for testing."""
    return {
        "models": {
            "embedding": {
                "default": "custom-model",
                "cache_dir": "/custom/path"
            }
        },
        "api": {
            "groq": {
                "endpoint": "https://custom-endpoint.com",
                "api_key": "test-api-key"
            }
        },
        "cache": {
            "enabled": False
        }
    }

@pytest.fixture
def config_manager_with_default(temp_dir, default_config):
    """Create a ConfigManager with modified DEFAULT_CONFIG for testing."""
    # Patch the DEFAULT_CONFIG in ConfigManager
    with patch.object(ConfigManager, 'DEFAULT_CONFIG', default_config):
        # Create a config file path in the temp directory
        config_path = os.path.join(temp_dir, "config.json")
        config = ConfigManager(config_path)
        yield config

@pytest.fixture
def config_manager_with_custom_file(temp_dir, default_config, custom_config):
    """Create a ConfigManager with a pre-existing custom config file."""
    # Patch the DEFAULT_CONFIG in ConfigManager
    with patch.object(ConfigManager, 'DEFAULT_CONFIG', default_config):
        # Create a config file with custom settings
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f)
        
        # Create ConfigManager that should load this file
        config = ConfigManager(config_path)
        yield config

class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_default_config_initialization(self, config_manager_with_default, default_config):
        """Test that ConfigManager initializes with default configuration."""
        # Check that config contains default values
        assert config_manager_with_default.get("models.embedding.default") == default_config["models"]["embedding"]["default"]
        assert config_manager_with_default.get("api.groq.endpoint") == default_config["api"]["groq"]["endpoint"]
        assert config_manager_with_default.get("cache.enabled") == default_config["cache"]["enabled"]
        
        # Check that config file was created
        assert os.path.exists(config_manager_with_default.config_path)

    def test_custom_config_loading(self, config_manager_with_custom_file, custom_config):
        """Test loading configuration from existing file."""
        # Check that custom values were loaded
        assert config_manager_with_custom_file.get("models.embedding.default") == custom_config["models"]["embedding"]["default"]
        assert config_manager_with_custom_file.get("api.groq.endpoint") == custom_config["api"]["groq"]["endpoint"]
        assert config_manager_with_custom_file.get("cache.enabled") == custom_config["cache"]["enabled"]
        
        # Check merged config (custom overriding default)
        assert config_manager_with_custom_file.get("api.huggingface.endpoint") is not None

    def test_config_saving(self, config_manager_with_default):
        """Test saving configuration changes."""
        # Modify config
        config_manager_with_default.set("models.new_model", "test-model")
        config_manager_with_default.set("api.new_service.url", "https://test.com")
        
        # Create a new ConfigManager with the same path to verify loading saved changes
        new_config = ConfigManager(config_manager_with_default.config_path)
        
        # Check that changes were saved and loaded
        assert new_config.get("models.new_model") == "test-model"
        assert new_config.get("api.new_service.url") == "https://test.com"

    def test_dot_notation_get(self, config_manager_with_default, default_config):
        """Test getting values using dot notation."""
        # Test getting various nested values
        assert config_manager_with_default.get("models.embedding.default") == default_config["models"]["embedding"]["default"]
        assert config_manager_with_default.get("api.groq.endpoint") == default_config["api"]["groq"]["endpoint"]
        assert config_manager_with_default.get("cache.max_size_mb") == default_config["cache"]["max_size_mb"]
        
        # Test default value for non-existent key
        assert config_manager_with_default.get("nonexistent.key", "default") == "default"
        assert config_manager_with_default.get("models.nonexistent", {}) == {}

    def test_dot_notation_set(self, config_manager_with_default):
        """Test setting values using dot notation."""
        # Set values at different nesting levels
        config_manager_with_default.set("models.embedding.default", "new-model", save=False)
        config_manager_with_default.set("api.new_key", "new-value", save=False)
        config_manager_with_default.set("new_section.nested.deeply.key", "value", save=False)
        
        # Check that values were set correctly
        assert config_manager_with_default.get("models.embedding.default") == "new-model"
        assert config_manager_with_default.get("api.new_key") == "new-value"
        assert config_manager_with_default.get("new_section.nested.deeply.key") == "value"
        
        # Verify nested structure was created correctly
        assert "new_section" in config_manager_with_default.config
        assert "nested" in config_manager_with_default.config["new_section"]
        assert "deeply" in config_manager_with_default.config["new_section"]["nested"]
        assert "key" in config_manager_with_default.config["new_section"]["nested"]["deeply"]

    def test_deep_update(self, config_manager_with_default):
        """Test _deep_update method for merging configurations."""
        # Create a deep update source
        update_source = {
            "models": {
                "embedding": {
                    "default": "updated-model",
                    "new_option": "value"
                },
                "new_model_type": {
                    "default": "new-default"
                }
            },
            "new_top_level": "top-value"
        }
        
        # Apply the update
        config_manager_with_default._deep_update(config_manager_with_default.config, update_source)
        
        # Check updates at various levels
        assert config_manager_with_default.get("models.embedding.default") == "updated-model"
        assert config_manager_with_default.get("models.embedding.new_option") == "value"
        assert config_manager_with_default.get("models.new_model_type.default") == "new-default"
        assert config_manager_with_default.get("new_top_level") == "top-value"
        
        # Check that non-updated values remain
        assert config_manager_with_default.get("api.groq.endpoint") is not None
        assert config_manager_with_default.get("cache.enabled") is not None

    def test_model_path_getter(self, config_manager_with_default, default_config):
        """Test get_model_path method."""
        # Test default model path
        assert config_manager_with_default.get_model_path() == default_config["models"]["embedding"]["default"]
        
        # Set a custom model path and test
        custom_model = "custom-test-model"
        config_manager_with_default.set("models.embedding.default", custom_model, save=False)
        assert config_manager_with_default.get_model_path() == custom_model
        
        # Test with non-default model type (should return None or empty string as it doesn't exist)
        nonexistent_model = config_manager_with_default.get_model_path("nonexistent")
        assert nonexistent_model is None or nonexistent_model == ""

    def test_cache_dir_expansion(self, config_manager_with_default):
        """Test get_cache_dir expands user directory."""
        # Set a path with tilde
        tilde_path = "~/test/cache"
        config_manager_with_default.set("cache.directory", tilde_path, save=False)
        
        # Get expanded path
        expanded_path = config_manager_with_default.get_cache_dir()
        
        # Check that tilde was expanded
        assert "~" not in expanded_path
        assert str(Path.home()) in expanded_path
        assert expanded_path.endswith("test/cache")

    @patch.dict(os.environ, {"GROQ_API_KEY": "env-api-key"})
    def test_api_key_from_env(self, config_manager_with_default):
        """Test getting API key from environment variable."""
        # Check that environment variable is used
        api_key = config_manager_with_default.get_api_key("groq")
        assert api_key == "env-api-key"
        
        # Check that key was set in config but not saved to file
        assert config_manager_with_default.get("api.groq.api_key") == "env-api-key"

    def test_api_key_from_config(self, config_manager_with_custom_file, custom_config):
        """Test getting API key from config file."""
        # The custom config fixture has an API key set
        api_key = config_manager_with_custom_file.get_api_key("groq")
        assert api_key == custom_config["api"]["groq"]["api_key"]

    def test_api_configured_check(self, config_manager_with_default):
        """Test is_api_configured method."""
        # Initially API is not configured
        assert not config_manager_with_default.is_api_configured("groq")
        
        # Set an API key
        config_manager_with_default.set("api.groq.api_key", "test-key", save=False)
        
        # Now API should be configured
        assert config_manager_with_default.is_api_configured("groq")

    def test_config_file_permissions(self, temp_dir, default_config):
        """Test that saved config file has appropriate permissions."""
        # Skip on non-POSIX platforms
        if os.name != 'posix':
            pytest.skip("File permission test only runs on POSIX systems")
            
        # Create config and ensure it gets saved
        with patch.object(ConfigManager, 'DEFAULT_CONFIG', default_config):
            config_path = os.path.join(temp_dir, "config_perms.json")
            config = ConfigManager(config_path)
            config.save_config()
            
            # Check file permissions (only user should have read/write)
            file_mode = os.stat(config_path).st_mode & 0o777  # Get permission bits
            assert file_mode & 0o600 == 0o600  # User has read/write
            assert file_mode & 0o077 == 0  # Group/others have no permissions

    def test_invalid_json_config(self, temp_dir, default_config):
        """Test handling of invalid JSON in config file."""
        # Create an invalid config file
        config_path = os.path.join(temp_dir, "invalid_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("{invalid json")
            
        # Try to load the invalid config
        with patch.object(ConfigManager, 'DEFAULT_CONFIG', default_config):
            config = ConfigManager(config_path)
            
            # Should use default config values despite invalid file
            assert config.get("models.embedding.default") == default_config["models"]["embedding"]["default"]
            assert config.get("api.groq.endpoint") == default_config["api"]["groq"]["endpoint"]

    def test_empty_config_file(self, temp_dir, default_config):
        """Test handling of empty config file."""
        # Create an empty config file
        config_path = os.path.join(temp_dir, "empty_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("")
            
        # Try to load the empty config
        with patch.object(ConfigManager, 'DEFAULT_CONFIG', default_config):
            config = ConfigManager(config_path)
            
            # Should use default config values for empty file
            assert config.get("models.embedding.default") == default_config["models"]["embedding"]["default"]
            assert config.get("api.groq.endpoint") == default_config["api"]["groq"]["endpoint"]

    def test_custom_config_path(self, temp_dir, default_config):
        """Test using a custom config path."""
        custom_path = os.path.join(temp_dir, "custom", "nested", "config.json")
        
        # Path shouldn't exist yet
        assert not os.path.exists(custom_path)
        
        # Create config with custom path
        with patch.object(ConfigManager, 'DEFAULT_CONFIG', default_config):
            config = ConfigManager(custom_path)
            
            # Directory should be created
            assert os.path.exists(os.path.dirname(custom_path))
            assert os.path.exists(custom_path)
            
            # Check config was loaded with defaults
            assert config.get("models.embedding.default") == default_config["models"]["embedding"]["default"]