"""Claude-on-Amazon-Bedrock provider.

Reuses :class:`AnthropicProvider` (same SDK surface) with a Bedrock
client. Configure full Bedrock model / inference-profile ids via
``llm.bedrock.models.*`` (e.g. ``global.anthropic.claude-sonnet-4-6``);
a bare ``claude-...`` id is expanded to the global endpoint. Auth uses
the standard AWS credential chain (env vars, shared config, or a role).
"""

from __future__ import annotations

from typing import Any

from .anthropic_provider import AnthropicProvider


def _to_bedrock_id(model: str) -> str:
    # Pass full Bedrock ids through; expand a bare "claude-..." id.
    if model.startswith("claude-"):
        return f"global.anthropic.{model}"
    return model


def build_bedrock_provider(
    *, region: str = "us-east-1", **kwargs: Any
) -> AnthropicProvider:
    """Construct an ``AnthropicProvider`` wired to Amazon Bedrock."""
    from anthropic import AnthropicBedrock  # requires boto3

    client = AnthropicBedrock(aws_region=region)
    return AnthropicProvider(
        client=client,
        model_id_fn=_to_bedrock_id,
        name_override="bedrock",
        **kwargs,
    )
