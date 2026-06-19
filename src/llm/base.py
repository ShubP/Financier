"""LLM provider interface shared by every backend.

The interface is tiny — one abstract ``complete()`` — with convenience
wrappers (``reason``/``classify``) and a provider-agnostic
``structured()`` that coerces free text into a validated Pydantic
model. Keeping structured output here (prompt + robust JSON parse)
means it works identically across Claude, Bedrock, Gemini, and mock.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

# Chat message: {"role": "user"|"assistant", "content": str}.
Message = dict[str, str]

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a provider call fails or is unusable."""


class LLMProvider(ABC):
    """Abstract base for all LLM backends.

    Subclasses implement :meth:`complete`; the rest builds on it.
    """

    name: str = "base"
    supports_thinking: bool = False

    def __init__(
        self,
        *,
        router_model: str,
        default_model: str,
        max_tokens: int = 3000,
        adaptive_thinking: bool = True,
    ) -> None:
        self.router_model = router_model
        self.default_model = default_model
        self.max_tokens = max_tokens
        self.adaptive_thinking = adaptive_thinking

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ) -> str:
        """Return the assistant's text reply."""

    def reason(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Answer from the default model (adaptive thinking)."""
        return self.complete(
            system,
            messages,
            model=self.default_model,
            max_tokens=max_tokens or self.max_tokens,
            thinking=self.adaptive_thinking and self.supports_thinking,
        )

    def classify(self, system: str, user: str) -> str:
        """Fast single-label answer from the router model."""
        return self.complete(
            system,
            [{"role": "user", "content": user}],
            model=self.router_model,
            max_tokens=128,
            thinking=False,
        )

    def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_retries: int = 1,
    ) -> T:
        """Return a validated ``schema`` instance from the reply."""
        json_schema = schema.model_json_schema()
        sys = (
            f"{system}\n\n"
            "Respond with ONLY a single JSON object conforming to "
            "this JSON Schema. No markdown fences, no commentary.\n"
            f"{json.dumps(json_schema)}"
        )
        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            raw = self.complete(
                sys,
                [{"role": "user", "content": user}],
                model=model or self.default_model,
                max_tokens=self.max_tokens,
                thinking=False,
            )
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_err = exc
        raise LLMError(
            f"Could not parse structured output: {last_err}"
        )


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object from a reply.

    Tolerates code fences and surrounding prose.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                return json.loads(text[start:end])
    raise ValueError("unbalanced JSON object in model output")
