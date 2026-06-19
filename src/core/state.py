"""Shared state passed between LangGraph nodes.

One ``AgentState`` dict flows: query -> router -> specialist agent(s)
-> response. Each node reads what it needs and writes its part back.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# Intents the router can pick; each maps to one agent.
Intent = Literal["qa", "portfolio", "market", "goal", "smalltalk"]


class RetrievedDoc(TypedDict):
    """A knowledge-base chunk with attribution."""

    text: str
    source: str
    score: float


class AgentState(TypedDict, total=False):
    """End-to-end state for a single user turn.

    ``total=False`` so each node only sets the keys it owns.
    """

    # Input
    query: str
    history: list[dict[str, str]]
    profile: dict[str, Any]

    # Routing
    intent: Intent
    route_reason: str

    # Working data
    context: list[RetrievedDoc]
    market: dict[str, Any]
    analysis: dict[str, Any]

    # Output
    response: str
    sources: list[str]
    agent_trace: list[str]
    error: str


def new_state(
    query: str,
    history: list[dict[str, str]] | None = None,
    profile: dict[str, Any] | None = None,
) -> AgentState:
    """Build a fresh state for one user turn."""
    return AgentState(
        query=query,
        history=history or [],
        profile=profile or {},
        context=[],
        market={},
        analysis={},
        sources=[],
        agent_trace=[],
    )
