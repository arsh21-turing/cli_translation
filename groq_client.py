import os
import logging
from typing import List, Dict, Any, Optional

try:
    import groq  # type: ignore
except ImportError:  # pragma: no cover
    groq = None  # The package may not be installed in some environments

logger = logging.getLogger(__name__)


class GroqClient:
    """Minimal wrapper around the Groq Python SDK used by the CLI project.

    This helper intentionally keeps the public surface area extremely small –
    just enough for translation-quality evaluation prompts.  All interaction
    goes through *generate_completion* (single-prompt) or *generate_chat_completion*
    (multi-turn chat prompts).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "llama3-8b-8192",
    ) -> None:
        # Allow callers to obtain the key from env to avoid passing secrets
        self.api_key: Optional[str] = api_key or os.getenv("GROQ_API_KEY")
        self.default_model: str = model

        if groq is None:
            logger.warning(
                "groq package not available – GroqClient will operate in stub mode"
            )
            self._client = None
        elif self.api_key is None:
            logger.warning("GROQ_API_KEY not set – GroqClient will operate in stub mode")
            self._client = None
        else:
            # Lazy import / client creation – avoids overhead when unused
            try:
                self._client = groq.Client(api_key=self.api_key)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                logger.error("Could not initialise Groq client: %s", exc)
                self._client = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Return the list of models the current key has access to."""
        if not self._client:
            return []
        try:
            models = self._client.models.list()
            # The Groq SDK returns an *Object* with a ``data`` attribute
            return list(models.data)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            logger.error("Error fetching Groq model list: %s", exc)
            return []

    # ------------------------------------------------------------------
    def generate_completion(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return a classic *completion* (single prompt / single response).

        The dictionary returned always contains at least the *text* key; if the
        call failed it also contains an *error* key with a description.
        """
        if not self._client:
            return {
                "text": "",
                "error": "Groq client not initialised – check API key and installation",
            }
        try:
            rsp = self._client.completions.create(  # type: ignore[attr-defined]
                model=model or self.default_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = rsp.choices[0]
            return {
                "text": choice.text,
                "model": rsp.model,
                "finish_reason": choice.finish_reason,
                "usage": rsp.usage.to_dict() if hasattr(rsp, "usage") else {},
            }
        except Exception as exc:  # pragma: no cover
            logger.error("Groq completion error: %s", exc)
            return {"text": "", "error": str(exc)}

    # ------------------------------------------------------------------
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return a chat completion following the OpenAI-compatible schema."""
        if not self._client:
            return {
                "content": "",
                "error": "Groq client not initialised – check API key and installation",
            }
        try:
            rsp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = rsp.choices[0]
            return {
                "content": choice.message.content,
                "model": rsp.model,
                "finish_reason": choice.finish_reason,
                "usage": rsp.usage.to_dict() if hasattr(rsp, "usage") else {},
            }
        except Exception as exc:  # pragma: no cover
            logger.error("Groq chat completion error: %s", exc)
            return {"content": "", "error": str(exc)}

    # ------------------------------------------------------------------
    @staticmethod
    def count_tokens(text: str) -> int:
        """Rudimentary token approximation – 1 token ~= 4 characters for English."""
        return max(1, len(text) // 4) 