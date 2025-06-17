"""embedding_cache.py
A very small, dependency-free in-memory cache for embedding vectors.
It is intentionally lightweight so the project can run in constrained
execution environments where a full external key-value store is
unavailable.
"""
from __future__ import annotations

from typing import Dict, Any
import hashlib
import threading


class EmbeddingCache:  # pragma: no cover – simple utility
    """A trivial max-size in-memory cache for embeddings.

    The implementation is intentionally naive (no LRU eviction) because
    embeddings are usually reused only within a short-lived CLI session
    or a single unit-test run.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._data: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _make_key(self, text: str, model_name: str) -> str:
        h = hashlib.md5()
        h.update(text.encode())
        h.update(b"__")
        h.update(model_name.encode())
        return h.hexdigest()

    # ------------------------------------------------------------------
    def get(self, text: str, model_name: str):
        key = self._make_key(text, model_name)
        with self._lock:
            if key in self._data:
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    # ------------------------------------------------------------------
    def set(self, text: str, model_name: str, vector) -> bool:  # type: ignore[override]
        with self._lock:
            if len(self._data) >= self.max_size:
                return False
            key = self._make_key(text, model_name)
            self._data[key] = vector
            return True

    # ------------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float | int]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


# Global shared instance that lives for the entire interpreter session.
embedding_cache = EmbeddingCache() 