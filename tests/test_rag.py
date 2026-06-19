"""Knowledge-base retrieval tests."""

from src.rag.knowledge_base import get_knowledge_base


def test_kb_has_chunks() -> None:
    kb = get_knowledge_base()
    assert kb.num_chunks > 0


def test_retrieve_compound_interest() -> None:
    kb = get_knowledge_base()
    hits = kb.retrieve("how does compound interest work?")
    assert hits
    sources = [h["source"] for h in hits]
    assert any("Compound" in s for s in sources)
