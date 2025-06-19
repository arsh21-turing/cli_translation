# tests/test_smart_cache_comprehensive.py

import unittest
import tempfile
import time
import os
import shutil
import sys
import json
import threading
import random
import numpy as np
from unittest.mock import MagicMock, patch
import logging
import gc
import pickle
import zlib

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import smart cache
from smart_cache import SmartCache, cached_api, cached_embedding, cached_result

class TestSmartCacheComprehensive(unittest.TestCase):
    """Comprehensive test suite for the SmartCache system."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SmartCache(
            cache_dir=self.temp_dir, 
            memory_size=100, 
            disk_size=200,
            default_ttl=60  # 60 seconds for faster testing
        )
        
    def tearDown(self):
        """Clean up test environment after each test."""
        self.cache.shutdown()
        # Force garbage collection to release file handles
        gc.collect()
        # Use retry logic for directory cleanup due to possible file handle delays on Windows
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(self.temp_dir)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to remove temp directory: {e}")
                else:
                    time.sleep(0.5)  # Short delay before retry
    
    #---------------------------------------------------------------------------
    # BASIC FUNCTIONALITY TESTS
    #---------------------------------------------------------------------------
    
    def test_set_get_basic(self):
        """Test basic set and get operations with various data types."""
        # Test with string
        self.cache.set("string_key", "string_value")
        value, found = self.cache.get("string_key")
        self.assertTrue(found)
        self.assertEqual(value, "string_value")
        
        # Test with int
        self.cache.set("int_key", 12345)
        value, found = self.cache.get("int_key")
        self.assertTrue(found)
        self.assertEqual(value, 12345)
        
        # Test with float
        self.cache.set("float_key", 123.45)
        value, found = self.cache.get("float_key")
        self.assertTrue(found)
        self.assertEqual(value, 123.45)
        
        # Test with bool
        self.cache.set("bool_key", True)
        value, found = self.cache.get("bool_key")
        self.assertTrue(found)
        self.assertEqual(value, True)
        
        # Test with None
        self.cache.set("none_key", None)
        value, found = self.cache.get("none_key")
        self.assertTrue(found)
        self.assertIsNone(value)
    
    def test_set_get_complex(self):
        """Test with complex data structures."""
        # Test with list
        test_list = [1, "two", 3.0, {"four": 4}]
        self.cache.set("list_key", test_list)
        value, found = self.cache.get("list_key")
        self.assertTrue(found)
        self.assertEqual(value, test_list)
        
        # Test with nested dict
        test_dict = {
            "name": "test",
            "nested_dict": {"a": 1, "b": 2},
            "nested_list": [1, 2, 3],
            "mixed": {"list": [1, 2, {"three": 3}]}
        }
        self.cache.set("dict_key", test_dict)
        value, found = self.cache.get("dict_key")
        self.assertTrue(found)
        self.assertEqual(value, test_dict)
        
        # Test with tuple
        test_tuple = (1, 2, "three", (4, 5))
        self.cache.set("tuple_key", test_tuple)
        value, found = self.cache.get("tuple_key")
        self.assertTrue(found)
        self.assertEqual(value, test_tuple)
        
        # Test with set (note: will be converted to list by pickle)
        test_set = {1, 2, 3, 4, 5}
        self.cache.set("set_key", test_set)
        value, found = self.cache.get("set_key")
        self.assertTrue(found)
        self.assertEqual(set(value), test_set)  # Convert back to set for comparison
    
    def test_set_get_numpy(self):
        """Test with numpy arrays, common in embedding scenarios."""
        # Test with small numpy array
        small_array = np.array([1, 2, 3, 4, 5])
        self.cache.set("small_array", small_array)
        value, found = self.cache.get("small_array")
        self.assertTrue(found)
        np.testing.assert_array_equal(value, small_array)
        
        # Test with large numpy array
        large_array = np.random.random((100, 100))
        self.cache.set("large_array", large_array)
        value, found = self.cache.get("large_array")
        self.assertTrue(found)
        np.testing.assert_array_equal(value, large_array)
        
        # Test with embedding-like vector
        embedding = np.random.random(768)  # Common embedding size
        self.cache.set("embedding", embedding)
        value, found = self.cache.get("embedding")
        self.assertTrue(found)
        np.testing.assert_array_equal(value, embedding)
    
    def test_custom_classes(self):
        """Test with custom classes to ensure proper pickling/unpickling."""
        class TestObject:
            def __init__(self, value, name):
                self.value = value
                self.name = name
                
        obj = TestObject(42, "test_object")
        self.cache.set("obj_key", obj)
        value, found = self.cache.get("obj_key")
        self.assertTrue(found)
        self.assertEqual(value.value, obj.value)
        self.assertEqual(value.name, obj.name)
    
    def test_non_existent_key(self):
        """Test getting a non-existent key."""
        value, found = self.cache.get("non_existent_key")
        self.assertFalse(found)
        self.assertIsNone(value)
    
    def test_delete(self):
        """Test deleting keys from cache."""
        # Set and confirm
        self.cache.set("delete_key", "delete_value")
        value, found = self.cache.get("delete_key")
        self.assertTrue(found)
        
        # Delete and confirm
        self.assertTrue(self.cache.delete("delete_key"))
        value, found = self.cache.get("delete_key")
        self.assertFalse(found)
        
        # Delete non-existent
        self.assertFalse(self.cache.delete("non_existent_key"))
    
    def test_clear(self):
        """Test clearing all cache entries."""
        # Set multiple entries
        for i in range(10):
            self.cache.set(f"clear_key_{i}", f"clear_value_{i}")
            
        # Verify entries exist
        for i in range(10):
            value, found = self.cache.get(f"clear_key_{i}")
            self.assertTrue(found)
            
        # Clear cache
        self.cache.clear()
        
        # Verify entries are gone
        for i in range(10):
            value, found = self.cache.get(f"clear_key_{i}")
            self.assertFalse(found)
    
    #---------------------------------------------------------------------------
    # TTL (TIME-TO-LIVE) TESTS
    #---------------------------------------------------------------------------
    
    def test_ttl_expiration(self):
        """Test that items expire after TTL."""
        # Set with 1 second TTL
        self.cache.set("ttl_key", "ttl_value", ttl=1)
        
        # Verify it exists
        value, found = self.cache.get("ttl_key")
        self.assertTrue(found)
        self.assertEqual(value, "ttl_value")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Verify it's gone
        value, found = self.cache.get("ttl_key")
        self.assertFalse(found)
    
    def test_ttl_different_values(self):
        """Test with different TTL values."""
        # Set multiple items with different TTLs
        self.cache.set("ttl_1", "value_1", ttl=1)
        self.cache.set("ttl_2", "value_2", ttl=2)
        self.cache.set("ttl_3", "value_3", ttl=3)
        
        # Wait 1.1 seconds
        time.sleep(1.1)
        
        # Check values
        _, found_1 = self.cache.get("ttl_1")
        _, found_2 = self.cache.get("ttl_2")
        _, found_3 = self.cache.get("ttl_3")
        
        self.assertFalse(found_1)  # Should be expired
        self.assertTrue(found_2)   # Should still exist
        self.assertTrue(found_3)   # Should still exist
        
        # Wait 1 more second
        time.sleep(1.0)
        
        # Check values
        _, found_2 = self.cache.get("ttl_2")
        _, found_3 = self.cache.get("ttl_3")
        
        self.assertFalse(found_2)  # Should be expired
        self.assertTrue(found_3)   # Should still exist
    
    def test_ttl_default(self):
        """Test default TTL behavior."""
        # Set with default TTL
        original_default = self.cache.default_ttl
        self.cache.default_ttl = 1  # Set to 1 second for testing
        
        self.cache.set("default_ttl", "default_value")  # No TTL provided
        
        # Verify it exists
        value, found = self.cache.get("default_ttl")
        self.assertTrue(found)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Verify it's gone
        value, found = self.cache.get("default_ttl")
        self.assertFalse(found)
        
        # Restore original default
        self.cache.default_ttl = original_default
    
    def test_cleanup_expired(self):
        """Test that cleanup removes expired items."""
        # Set items that expire quickly
        for i in range(10):
            self.cache.set(f"expired_{i}", f"value_{i}", ttl=1)
            
        # Wait for expiration
        time.sleep(1.1)
        
        # Run cleanup
        self.cache.cleanup()
        
        # Check memory and disk
        for i in range(10):
            value, found = self.cache.get(f"expired_{i}")
            self.assertFalse(found)
            
            # Verify disk file is gone
            path = self.cache._get_disk_cache_path(f"expired_{i}")
            self.assertFalse(os.path.exists(path))

if __name__ == '__main__':
    unittest.main() 