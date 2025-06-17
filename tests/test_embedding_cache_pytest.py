import sys
import os
from copy import deepcopy

# Make project root importable when tests executed from repository root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # type: ignore

from embedding_cache import embedding_cache, EmbeddingCache

# -----------------------------------------------------------------------------
# Lightweight generator that exercises the same cache API as the real one but
# without the heavyweight model dependencies.
# -----------------------------------------------------------------------------

class LiteEmbeddingGenerator:  # pragma: no cover – test helper
    """Very small replacement for the full EmbeddingGenerator.

    It relies solely on *embedding_cache* so we can test cache logic without
    pulling in Sentence-Transformers / Torch.
    """

    def __init__(self, model_name: str = "mock-model", *, use_cache: bool = True):
        self.model_name = model_name
        self.use_cache = use_cache

    # ------------------------------------------------------------------
    def generate_embedding(self, text):  # noqa: D401 – simple stub
        # Degenerate 0-vector for falsy or non-string inputs
        if not isinstance(text, str) or not text:
            return np.zeros(3)

        if self.use_cache:
            cached = embedding_cache.get(text, self.model_name)
            if cached is not None:
                return cached

        # Deterministic mock vector (length-based) – 3-D for convenience
        val = float(len(text))
        vec = np.array([val, val / 2.0, val / 3.0], dtype=float)

        if self.use_cache:
            embedding_cache.set(text, self.model_name, vec)
        return vec


# -----------------------------------------------------------------------------
# PyTest fixtures and tests
# -----------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Start each test with a clean global cache."""
    embedding_cache.clear()
    yield
    embedding_cache.clear()


# ------------------------------------------------------------------
# Basic caching behaviour
# ------------------------------------------------------------------

def test_first_call_miss():
    gen = LiteEmbeddingGenerator()
    txt = "Cache me if you can."

    stats0 = embedding_cache.stats()
    assert stats0["hits"] == stats0["misses"] == stats0["size"] == 0

    vec1 = gen.generate_embedding(txt)
    stats1 = embedding_cache.stats()

    assert stats1["misses"] == 1 and stats1["hits"] == 0 and stats1["size"] == 1
    # Cached vector stored
    assert np.array_equal(vec1, embedding_cache.get(txt, gen.model_name))


def test_second_call_hit():
    gen = LiteEmbeddingGenerator()
    txt = "Once cached, always cached."

    vec1 = gen.generate_embedding(txt)  # miss
    vec1_copy = deepcopy(vec1)
    vec2 = gen.generate_embedding(txt)  # hit

    stats = embedding_cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["size"] == 1
    # Same numerical content
    assert np.array_equal(vec1_copy, vec2)

# ------------------------------------------------------------------
# Size limit enforcement
# ------------------------------------------------------------------

def test_max_size_never_exceeded():
    original = embedding_cache.max_size
    embedding_cache.max_size = 5
    try:
        gen = LiteEmbeddingGenerator()
        for i in range(10):
            gen.generate_embedding(f"sentence {i}")
            assert embedding_cache.stats()["size"] <= 5
    finally:
        embedding_cache.max_size = original

# ------------------------------------------------------------------
# Model-name differentiation
# ------------------------------------------------------------------

def test_separate_models_create_separate_entries():
    g1 = LiteEmbeddingGenerator(model_name="m1")
    g2 = LiteEmbeddingGenerator(model_name="m2")
    txt = "Same sentence different models"

    g1.generate_embedding(txt)
    g2.generate_embedding(txt)

    st = embedding_cache.stats()
    assert st["size"] == 2 and st["misses"] == 2

    # Hits on second round
    g1.generate_embedding(txt)
    g2.generate_embedding(txt)
    st2 = embedding_cache.stats()
    assert st2["hits"] == 2 and st2["size"] == 2

# ------------------------------------------------------------------
# Cache disabled
# ------------------------------------------------------------------

def test_cache_disabled():
    gen = LiteEmbeddingGenerator(use_cache=False)
    txt = "Disable cache for me."
    gen.generate_embedding(txt)
    gen.generate_embedding(txt)
    st = embedding_cache.stats()
    # Still zero because we never touched the cache
    assert st["size"] == st["hits"] == st["misses"] == 0

# ------------------------------------------------------------------
# Key generation logic unit-test
# ------------------------------------------------------------------

def test_key_generation_consistency():
    cache = EmbeddingCache()
    k1 = cache._make_key("txt", "model")  # pylint: disable=protected-access
    k2 = cache._make_key("txt", "model")
    k3 = cache._make_key("other", "model")

    assert k1 == k2 and k1 != k3 