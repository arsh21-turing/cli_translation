"""weight_config_tool.py
Utility to manage scoring weights for :pyclass:`DualAnalysisSystem` and other
components that may consume the same *weight map*.

The class is intentionally dependency-free – it only requires a duck-typed
`ConfigManager` that exposes ``get(key, default)`` / ``set(key, value)`` /
``save_config()``.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WeightConfigTool:
    """Simple manager for weight presets and persistence."""

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "embedding_similarity": 0.4,
        "accuracy": 0.2,
        "fluency": 0.15,
        "terminology": 0.15,
        "style": 0.1,
    }

    _BUILTIN_PRESETS: Dict[str, Dict[str, float]] = {
        "balanced": _DEFAULT_WEIGHTS,
        "semantic_focus": {
            "embedding_similarity": 0.6,
            "accuracy": 0.2,
            "fluency": 0.1,
            "terminology": 0.05,
            "style": 0.05,
        },
        "accuracy_focus": {
            "embedding_similarity": 0.3,
            "accuracy": 0.5,
            "fluency": 0.1,
            "terminology": 0.05,
            "style": 0.05,
        },
        "fluency_focus": {
            "embedding_similarity": 0.2,
            "accuracy": 0.2,
            "fluency": 0.4,
            "terminology": 0.1,
            "style": 0.1,
        },
        "technical": {
            "embedding_similarity": 0.3,
            "accuracy": 0.3,
            "fluency": 0.1,
            "terminology": 0.3,
            "style": 0.0,
        },
        "creative": {
            "embedding_similarity": 0.1,
            "accuracy": 0.2,
            "fluency": 0.4,
            "terminology": 0.1,
            "style": 0.2,
        },
    }

    # ------------------------------------------------------------------
    def __init__(self, config_manager: Optional[Any] = None) -> None:
        self.config = config_manager
        self.presets: Dict[str, Dict[str, float]] = {
            name: dict(weights) for name, weights in self._BUILTIN_PRESETS.items()
        }
        self._load_custom_presets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_current_weights(self) -> Dict[str, float]:
        if self.config:
            w = self.config.get("scoring_weights", None)  # type: ignore[attr-defined]
            if w and self._validate_weights(w):
                return dict(w)
        return self._DEFAULT_WEIGHTS.copy()

    def set_weights(self, weights: Dict[str, float], *, save: bool = False) -> bool:
        if not self._validate_weights(weights):
            return False
        norm = self._normalise(weights)
        if save and self.config:
            try:
                self.config.set("scoring_weights", norm)  # type: ignore[attr-defined]
                self.config.save_config()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not save weights: %s", exc)
        return True

    def reset_to_defaults(self, *, save: bool = False) -> Dict[str, float]:
        if save and self.config:
            try:
                self.config.set("scoring_weights", self._DEFAULT_WEIGHTS)  # type: ignore[attr-defined]
                self.config.save_config()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not save defaults: %s", exc)
        return self._DEFAULT_WEIGHTS.copy()

    def get_available_presets(self) -> Dict[str, Dict[str, float]]:
        return {name: dict(w) for name, w in self.presets.items()}

    def apply_preset(self, name: str, *, save: bool = False) -> Optional[Dict[str, float]]:
        if name not in self.presets:
            logger.error("Unknown preset '%s'", name)
            return None
        weights = self.presets[name]
        self.set_weights(weights, save=save)
        return dict(weights)

    def save_custom_preset(self, name: str, weights: Dict[str, float]) -> bool:
        if name in self._BUILTIN_PRESETS:
            name = f"custom_{name}"
        if not self._validate_weights(weights):
            return False
        self.presets[name] = self._normalise(weights)
        if self.config:
            try:
                custom = self.config.get("weight_presets", {})  # type: ignore[attr-defined]
                custom[name] = weights
                self.config.set("weight_presets", custom)  # type: ignore[attr-defined]
                self.config.save_config()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not persist preset: %s", exc)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_custom_presets(self):
        if not self.config:
            return
        try:
            extra = self.config.get("weight_presets", {})  # type: ignore[attr-defined]
            for name, weights in extra.items():
                if self._validate_weights(weights):
                    self.presets[name] = self._normalise(weights)
        except Exception as exc:  # pragma: no cover
            logger.debug("No custom presets in config: %s", exc)

    @staticmethod
    def _normalise(w: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(v, 0.0) for v in w.values())
        if total <= 0:
            return WeightConfigTool._DEFAULT_WEIGHTS.copy()
        return {k: max(v, 0.0) / total for k, v in w.items()}

    def _validate_weights(self, w: Dict[str, float]) -> bool:
        keys = set(self._DEFAULT_WEIGHTS)
        if not keys.issubset(w):
            logger.error("Weights missing keys: %s", keys - set(w))
            return False
        if any(v < 0 for v in w.values()):
            logger.error("Weights must be non-negative")
            return False
        if sum(w.values()) <= 0:
            logger.error("Sum of weights must be > 0")
            return False
        return True

# -----------------------------------------------------------------------------
# End of public API – no CLI helper in this stripped-down version.
# ----------------------------------------------------------------------------- 