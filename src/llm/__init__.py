"""Pluggable LLM providers.

One :class:`~src.llm.base.LLMProvider` interface is implemented by
Claude (direct API), Claude-on-Bedrock, Gemini, and a dependency-free
mock. Use :func:`~src.llm.factory.build_provider` to construct the
right one from config and available credentials.
"""

from .base import LLMError, LLMProvider, Message
from .factory import build_provider

__all__ = ["LLMProvider", "LLMError", "Message", "build_provider"]
