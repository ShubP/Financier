"""Provider selection.

``build_provider`` resolves the configured provider (or auto-detects
one from available credentials) and always falls back to the mock
provider, so the app can never fail to start for lack of a key.
"""

from __future__ import annotations

from ..core.config import Config, env, load_config
from ..core.logging_utils import get_logger
from .base import LLMProvider
from .mock_provider import MockProvider

logger = get_logger(__name__)


def _base_kwargs(cfg: Config) -> dict:
    return {
        "router_model": cfg.get(
            "llm.models.router", "claude-haiku-4-5"
        ),
        "default_model": cfg.get(
            "llm.models.default", "claude-opus-4-8"
        ),
        "max_tokens": int(cfg.get("llm.max_tokens", 3000)),
        "adaptive_thinking": bool(
            cfg.get("llm.adaptive_thinking", True)
        ),
    }


def _aws_creds_available() -> bool:
    """True if AWS credentials resolve via env, shared file, or role."""
    try:
        import boto3

        return boto3.session.Session().get_credentials() is not None
    except Exception:
        has_keys = bool(
            env("AWS_ACCESS_KEY_ID") and env("AWS_SECRET_ACCESS_KEY")
        )
        return has_keys or bool(env("AWS_PROFILE"))


def build_provider(cfg: Config | None = None) -> LLMProvider:
    """Return a ready LLM provider from config + environment."""
    cfg = cfg or load_config()
    choice = str(cfg.get("llm.provider", "auto")).lower()
    kwargs = _base_kwargs(cfg)
    fallback_to_mock = bool(cfg.get("llm.fallback_to_mock", True))

    def make_anthropic() -> LLMProvider | None:
        if not env("ANTHROPIC_API_KEY"):
            return None
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=env("ANTHROPIC_API_KEY"), **kwargs
        )

    def make_bedrock() -> LLMProvider | None:
        if not _aws_creds_available():
            return None
        bkwargs = dict(kwargs)
        router = cfg.get("llm.bedrock.models.router")
        default = cfg.get("llm.bedrock.models.default")
        if router:
            bkwargs["router_model"] = router
        if default:
            bkwargs["default_model"] = default
        region = cfg.get("llm.bedrock.region", "us-east-1")
        engine = str(cfg.get("llm.bedrock.engine", "converse")).lower()
        try:
            if engine == "anthropic":
                from .bedrock_provider import build_bedrock_provider

                return build_bedrock_provider(region=region, **bkwargs)
            from .bedrock_converse_provider import (
                build_bedrock_converse_provider,
            )

            return build_bedrock_converse_provider(
                region=region, **bkwargs
            )
        except Exception as exc:  # missing boto3 / bad config
            logger.warning("Bedrock unavailable: %s", exc)
            return None

    def make_gemini() -> LLMProvider | None:
        if not env("GOOGLE_API_KEY"):
            return None
        try:
            from .gemini_provider import GeminiProvider

            return GeminiProvider(
                api_key=env("GOOGLE_API_KEY"), **kwargs
            )
        except Exception as exc:  # missing google-generativeai
            logger.warning("Gemini unavailable: %s", exc)
            return None

    explicit = {
        "anthropic": make_anthropic,
        "bedrock": make_bedrock,
        "gemini": make_gemini,
        "mock": lambda: MockProvider(**kwargs),
    }

    if choice in explicit:
        provider = explicit[choice]()
        if provider is not None:
            logger.info("LLM provider: %s (explicit)", provider.name)
            return provider
        logger.warning(
            "Requested provider '%s' unavailable; falling back.",
            choice,
        )

    for builder in (make_anthropic, make_bedrock, make_gemini):
        provider = builder()
        if provider is not None:
            logger.info(
                "LLM provider: %s (auto-detected)", provider.name
            )
            return provider

    if not fallback_to_mock:
        raise RuntimeError(
            "No LLM credentials found and fallback_to_mock is "
            "disabled. Configure AWS (Bedrock) or set "
            "ANTHROPIC_API_KEY / GOOGLE_API_KEY."
        )
    logger.info("LLM provider: mock (no credentials — demo mode)")
    return MockProvider(**kwargs)
