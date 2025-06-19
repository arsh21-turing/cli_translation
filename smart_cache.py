import os
import time
import pickle
import threading
import hashlib
from typing import Any, Tuple, Optional, Union, Callable
from functools import wraps
import tempfile
import shutil


class SmartCache:
    """A smart caching system with memory and disk tiers and TTL support."""
    
    def __init__(self, cache_dir: str = None, memory_size: int = 100, 
                 disk_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the SmartCache.
        
        Args:
            cache_dir: Directory for disk cache. If None, uses temp directory
            memory_size: Maximum number of items in memory cache
            disk_size: Maximum number of items in disk cache
            default_ttl: Default time-to-live in seconds
        """
        self.memory_size = memory_size
        self.disk_size = disk_size
        self.default_ttl = default_ttl
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="smart_cache_")
            self._temp_dir = True
        else:
            self.cache_dir = cache_dir
            self._temp_dir = False
            os.makedirs(cache_dir, exist_ok=True)
        
        # Memory cache: key -> (value, expiry_time)
        self._memory_cache = {}
        self._memory_access_order = []  # For LRU eviction
        
        # Disk cache metadata: key -> expiry_time
        self._disk_metadata = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing disk metadata
        self._load_disk_metadata()
    
    def _load_disk_metadata(self):
        """Load disk cache metadata."""
        metadata_file = os.path.join(self.cache_dir, "_metadata.pkl")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    self._disk_metadata = pickle.load(f)
            except Exception:
                self._disk_metadata = {}
    
    def _save_disk_metadata(self):
        """Save disk cache metadata."""
        metadata_file = os.path.join(self.cache_dir, "_metadata.pkl")
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(self._disk_metadata, f)
        except Exception:
            pass  # Fail silently
    
    def _get_disk_cache_path(self, key: str) -> str:
        """Get the file path for a disk cache entry."""
        # Create a safe filename from the key
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{key_hash}.pkl")
    
    def _is_expired(self, expiry_time: float) -> bool:
        """Check if an item has expired."""
        return time.time() > expiry_time
    
    def _evict_memory_lru(self):
        """Evict least recently used item from memory."""
        if not self._memory_access_order:
            return
        
        lru_key = self._memory_access_order.pop(0)
        if lru_key in self._memory_cache:
            del self._memory_cache[lru_key]
    
    def _update_memory_access(self, key: str):
        """Update memory access order for LRU."""
        if key in self._memory_access_order:
            self._memory_access_order.remove(key)
        self._memory_access_order.append(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds. If None, uses default_ttl
        """
        if ttl is None:
            ttl = self.default_ttl
        
        expiry_time = time.time() + ttl
        
        with self._lock:
            # Store in memory cache
            self._memory_cache[key] = (value, expiry_time)
            self._update_memory_access(key)
            
            # Evict if memory cache is full
            while len(self._memory_cache) > self.memory_size:
                self._evict_memory_lru()
            
            # Also store to disk
            try:
                cache_path = self._get_disk_cache_path(key)
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                self._disk_metadata[key] = expiry_time
                self._save_disk_metadata()
                
                # Evict old disk entries if needed
                while len(self._disk_metadata) > self.disk_size:
                    # Remove oldest entry
                    oldest_key = min(self._disk_metadata.keys(), 
                                   key=lambda k: self._disk_metadata[k])
                    self._remove_disk_entry(oldest_key)
                    
            except Exception:
                pass  # Disk operations fail silently
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (value, found) where found is True if key exists and not expired
        """
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                value, expiry_time = self._memory_cache[key]
                if not self._is_expired(expiry_time):
                    self._update_memory_access(key)
                    return value, True
                else:
                    # Expired, remove it
                    del self._memory_cache[key]
                    if key in self._memory_access_order:
                        self._memory_access_order.remove(key)
            
            # Check disk cache
            if key in self._disk_metadata:
                expiry_time = self._disk_metadata[key]
                if not self._is_expired(expiry_time):
                    try:
                        cache_path = self._get_disk_cache_path(key)
                        if os.path.exists(cache_path):
                            with open(cache_path, 'rb') as f:
                                value = pickle.load(f)
                            
                            # Promote to memory cache
                            self._memory_cache[key] = (value, expiry_time)
                            self._update_memory_access(key)
                            
                            # Evict if memory cache is full
                            while len(self._memory_cache) > self.memory_size:
                                self._evict_memory_lru()
                            
                            return value, True
                    except Exception:
                        pass  # Disk read failed
                else:
                    # Expired, remove it
                    self._remove_disk_entry(key)
            
            return None, False
    
    def _remove_disk_entry(self, key: str):
        """Remove a disk cache entry."""
        if key in self._disk_metadata:
            del self._disk_metadata[key]
            self._save_disk_metadata()
        
        cache_path = self._get_disk_cache_path(key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed, False otherwise
        """
        with self._lock:
            found = False
            
            # Remove from memory
            if key in self._memory_cache:
                del self._memory_cache[key]
                found = True
            
            if key in self._memory_access_order:
                self._memory_access_order.remove(key)
            
            # Remove from disk
            if key in self._disk_metadata:
                self._remove_disk_entry(key)
                found = True
            
            return found
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Clear memory
            self._memory_cache.clear()
            self._memory_access_order.clear()
            
            # Clear disk
            for key in list(self._disk_metadata.keys()):
                self._remove_disk_entry(key)
            
            self._disk_metadata.clear()
            self._save_disk_metadata()
    
    def cleanup(self):
        """Remove expired entries."""
        current_time = time.time()
        
        with self._lock:
            # Clean memory cache
            expired_keys = []
            for key, (value, expiry_time) in self._memory_cache.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                if key in self._memory_access_order:
                    self._memory_access_order.remove(key)
            
            # Clean disk cache
            expired_disk_keys = []
            for key, expiry_time in self._disk_metadata.items():
                if current_time > expiry_time:
                    expired_disk_keys.append(key)
            
            for key in expired_disk_keys:
                self._remove_disk_entry(key)
    
    def shutdown(self):
        """Shutdown the cache and cleanup resources."""
        with self._lock:
            self._save_disk_metadata()
            
            # Clean up temp directory if we created it
            if self._temp_dir and os.path.exists(self.cache_dir):
                try:
                    shutil.rmtree(self.cache_dir)
                except Exception:
                    pass


# Global cache instance
_global_cache = None


def get_global_cache() -> SmartCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache


def _create_cache_key(*args, **kwargs) -> str:
    """Create a cache key from function arguments."""
    import json
    
    def serialize_arg(arg):
        if hasattr(arg, '__dict__'):
            return str(arg.__dict__)
        else:
            return str(arg)
    
    args_str = '_'.join(serialize_arg(arg) for arg in args)
    kwargs_str = '_'.join(f"{k}={serialize_arg(v)}" for k, v in sorted(kwargs.items()))
    
    full_str = f"{args_str}_{kwargs_str}"
    # Create a hash for very long keys
    if len(full_str) > 200:
        full_str = hashlib.md5(full_str.encode()).hexdigest()
    
    return full_str


def cached_api(ttl: int = 3600):
    """
    Decorator for caching API function results.
    
    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            cache_key = f"api_{func.__name__}_{_create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result, found = cache.get(cache_key)
            if found:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


def cached_embedding(ttl: int = 7200):
    """
    Decorator for caching embedding function results.
    
    Args:
        ttl: Time-to-live in seconds (default 2 hours)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            cache_key = f"embedding_{func.__name__}_{_create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result, found = cache.get(cache_key)
            if found:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


def cached_result(ttl: int = 1800):
    """
    Decorator for caching general function results.
    
    Args:
        ttl: Time-to-live in seconds (default 30 minutes)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            cache_key = f"result_{func.__name__}_{_create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result, found = cache.get(cache_key)
            if found:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator
