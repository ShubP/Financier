"""Amazon Bedrock provider via the model-agnostic Converse API.

Uses ``boto3``'s ``bedrock-runtime.converse`` so it works with any
Bedrock chat model (Amazon Nova, Meta Llama, Mistral, Claude, ...).
The default config uses Amazon Nova, which is inexpensive and enabled
instantly (no third-party model approval). Auth uses the standard AWS
credential chain (env vars, shared config, or a role).
"""

from __future__ import annotations

from typing import Any, Callable

from .base import LLMError, LLMProvider, Message


def _normalize(messages: list[Message]) -> list[dict[str, str]]:
    """Coerce history into the alternating, user-first shape Converse needs."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if not out and role == "assistant":
            continue  # the first message must be from the user
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    if not out:
        out = [{"role": "user", "content": "Hello"}]
    return out


class BedrockConverseProvider(LLMProvider):
    """Call any Bedrock chat model through the Converse API."""

    name = "bedrock"
    supports_thinking = False

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        model_id_fn: Callable[[str], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        import boto3  # in core requirements

        self._client = boto3.client(
            "bedrock-runtime", region_name=region
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
        conv = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in _normalize(messages)
        ]
        params: dict[str, Any] = {
            "modelId": model_id,
            "messages": conv,
            "inferenceConfig": {
                "maxTokens": max_tokens or self.max_tokens,
                "temperature": 0.3,
            },
        }
        if system:
            params["system"] = [{"text": system}]
        try:
            resp = self._client.converse(**params)
        except Exception as exc:
            raise LLMError(f"bedrock converse failed: {exc}") from exc
        blocks = (
            resp.get("output", {}).get("message", {}).get("content", [])
        )
        return "".join(
            b.get("text", "") for b in blocks if "text" in b
        ).strip()


def build_bedrock_converse_provider(
    *, region: str = "us-east-1", **kwargs: Any
) -> BedrockConverseProvider:
    """Construct a :class:`BedrockConverseProvider`."""
    return BedrockConverseProvider(region=region, **kwargs)
