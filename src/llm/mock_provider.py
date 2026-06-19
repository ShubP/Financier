"""Dependency-free mock provider.

Lets the whole pipeline run with no API keys: every node executes, the
router keyword-classifies, and agents produce plausible (clearly
labelled) text. Powers offline demos/tests and is the final fallback
in the provider chain.
"""

from __future__ import annotations

from typing import Any, get_args

from pydantic import BaseModel

from .base import LLMProvider, Message

# Ordered keyword map used to fake the router. The first intent whose
# keywords appear in the query wins; default is "qa".
_INTENT_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "portfolio",
        [
            "portfolio", "holding", "allocation", "diversif",
            "rebalanc", "asset mix", "my stocks", "my investments",
        ],
    ),
    (
        "market",
        [
            "price", "quote", "ticker", "market", "trend",
            "trading at", "stock price", "how is", "$",
        ],
    ),
    (
        "goal",
        [
            "goal", "retire", "save", "saving", "target",
            "projection", "afford", "how much", "year",
        ],
    ),
    ("smalltalk", ["hi", "hello", "hey", "thanks", "thank you"]),
]


class MockProvider(LLMProvider):
    """Deterministic, offline stand-in for a real LLM."""

    name = "mock"
    supports_thinking = False

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> str:
        user = next(
            (
                m["content"]
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        return (
            "_(Demo mode — running on the built-in mock model; add an "
            "API key to .env for real Claude responses.)_\n\n"
            "Here is a general, educational overview related to: "
            f"“{user.strip()[:160]}”. In a real run a specialist "
            "agent would answer using the knowledge base and live "
            "market data."
        )

    def structured(
        self,
        schema: type[BaseModel],
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_retries: int = 1,
    ) -> Any:
        """Build a valid instance without an LLM.

        Special-cases an ``intent`` field (router decisions) via
        keyword matching; otherwise fills required fields with type
        defaults.
        """
        fields = schema.model_fields
        values: dict[str, Any] = {}
        if "intent" in fields:
            values["intent"] = self._keyword_intent(
                user, fields["intent"].annotation
            )
        for fname, finfo in fields.items():
            if fname in values or not finfo.is_required():
                continue
            values[fname] = _default_for(finfo.annotation)
        if "reason" in fields and "reason" not in values:
            values["reason"] = "Keyword routing (mock provider)."
        return schema.model_validate(values)

    @staticmethod
    def _keyword_intent(user: str, annotation: Any) -> str:
        allowed = {str(a) for a in get_args(annotation)} or {"qa"}
        text = user.lower()
        for intent, keywords in _INTENT_KEYWORDS:
            if intent in allowed and any(
                k in text for k in keywords
            ):
                return intent
        if "qa" in allowed:
            return "qa"
        return next(iter(allowed))


def _default_for(annotation: Any) -> Any:
    """A type-appropriate placeholder for a required field."""
    origin = getattr(annotation, "__origin__", None)
    if annotation in (int, float):
        return 0
    if annotation is bool:
        return False
    if annotation is str:
        return ""
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if isinstance(annotation, type) and issubclass(
        annotation, BaseModel
    ):
        return {}
    return ""
