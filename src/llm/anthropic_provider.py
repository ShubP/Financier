"""Claude provider.

Handles both the direct Anthropic API and Claude-on-Bedrock — the only
differences are the client object and the model-id transform (Bedrock
prefixes ids with ``anthropic.``), both injected, so one class covers
both.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import LLMError, LLMProvider, Message


class AnthropicProvider(LLMProvider):
    """Talk to Claude via the official ``anthropic`` SDK."""

    name = "anthropic"
    supports_thinking = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model_id_fn: Callable[[str], str] | None = None,
        name_override: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if name_override:
            self.name = name_override
        if client is not None:
            self._client = client
        else:
            from anthropic import Anthropic  # required dependency

            self._client = (
                Anthropic(api_key=api_key) if api_key else Anthropic()
            )
        self._model_id_fn = model_id_fn or (lambda m: m)

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> str:
        model_id = self._model_id_fn(model or self.default_model)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if thinking:
            # Adaptive thinking: model decides how much to reason;
            # display is omitted, so we read only text blocks.
            kwargs["thinking"] = {"type": "adaptive"}
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(
                f"{self.name} request failed: {exc}"
            ) from exc

        text = "".join(
            b.text
            for b in resp.content
            if getattr(b, "type", None) == "text"
        ).strip()
        stop = getattr(resp, "stop_reason", None)
        if not text and stop == "refusal":
            return "I'm not able to help with that request."
        return text
