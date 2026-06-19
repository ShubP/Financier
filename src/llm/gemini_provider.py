"""Optional Google Gemini provider.

A lightweight alternative backend to show multi-provider support.
Claude model ids in the config map to a single Gemini model, since ids
are not interchangeable across vendors.
"""

from __future__ import annotations

from typing import Any

from .base import LLMError, LLMProvider, Message

# Course-recommended default: generous free tier, low latency.
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiProvider(LLMProvider):
    """Talk to Gemini via ``google-generativeai``."""

    name = "gemini"
    supports_thinking = False

    def __init__(
        self,
        *,
        api_key: str,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        import google.generativeai as genai  # optional dependency

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = gemini_model

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> str:
        gmodel = self._genai.GenerativeModel(
            self._model_name, system_instruction=system
        )
        contents = [
            {
                "role": (
                    "model"
                    if m.get("role") == "assistant"
                    else "user"
                ),
                "parts": [m["content"]],
            }
            for m in messages
        ]
        try:
            resp = gmodel.generate_content(
                contents,
                generation_config={
                    "max_output_tokens": (
                        max_tokens or self.max_tokens
                    )
                },
            )
        except Exception as exc:
            raise LLMError(f"gemini request failed: {exc}") from exc
        return (getattr(resp, "text", "") or "").strip()
