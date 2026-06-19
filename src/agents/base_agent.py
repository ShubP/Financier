"""Shared agent helpers."""

from __future__ import annotations

from typing import Any

from ..llm.base import Message

_MAX_HISTORY_TURNS = 6


def history_messages(state: dict[str, Any], user: str) -> list[Message]:
    """Build a message list from recent history plus the new turn."""
    messages: list[Message] = []
    history = state.get("history", []) or []
    for turn in history[-_MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages
