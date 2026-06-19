"""Finance Q&A agent: retrieval-augmented education."""

from __future__ import annotations

from typing import Any

from ..core.state import RetrievedDoc
from ..llm.base import LLMProvider
from ..rag.knowledge_base import KnowledgeBase
from .base_agent import history_messages

_SYSTEM = (
    "You are Financier, a warm, plain-spoken financial educator for "
    "beginners. Answer the user's question clearly and concisely (a "
    "few short paragraphs). Prefer the provided context when it is "
    "relevant and weave it in naturally; if the context does not "
    "cover the question, answer from general knowledge and say so "
    "briefly. Define any jargon. Do NOT give individualized "
    "investment advice or specific buy/sell recommendations."
)


class QAAgent:
    """Answers concept questions grounded in the knowledge base."""

    name = "qa"

    def __init__(
        self, llm: LLMProvider, kb: KnowledgeBase | None
    ) -> None:
        self.llm = llm
        self.kb = kb

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        docs: list[RetrievedDoc] = (
            self.kb.retrieve(query) if self.kb else []
        )
        system = _SYSTEM
        if docs:
            blocks = "\n\n".join(
                f"[{d['source']}]\n{d['text']}" for d in docs
            )
            system = f"{_SYSTEM}\n\nContext:\n{blocks}"
        answer = self.llm.reason(system, history_messages(state, query))
        sources = list(dict.fromkeys(d["source"] for d in docs))
        trace = state.get("agent_trace", []) + ["qa"]
        return {
            "response": answer,
            "context": docs,
            "sources": sources,
            "agent_trace": trace,
        }
