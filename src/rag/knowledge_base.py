"""Knowledge base: load docs, chunk them, retrieve with attribution.

Markdown documents under ``rag.knowledge_dir`` are split into
overlapping chunks and indexed by the configured retriever. Each hit
carries its source document so responses can cite where facts came
from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.config import Config, load_config
from ..core.logging_utils import get_logger
from ..core.state import RetrievedDoc
from .retrievers import Backend, build_backend

logger = get_logger(__name__)


@dataclass
class Document:
    text: str
    source: str


@dataclass
class Chunk:
    text: str
    source: str


def _humanize(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").title()


def load_documents(directory: Path) -> list[Document]:
    """Load every ``*.md`` file as a :class:`Document`."""
    docs: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(text=text, source=_humanize(path.stem)))
    return docs


def chunk_documents(
    docs: list[Document], size: int, overlap: int
) -> list[Chunk]:
    """Split documents into overlapping, paragraph-aware chunks."""
    chunks: list[Chunk] = []
    for doc in docs:
        paras = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        buf = ""
        for para in paras:
            if buf and len(buf) + len(para) + 1 > size:
                chunks.append(Chunk(text=buf.strip(), source=doc.source))
                tail = buf[-overlap:] if overlap else ""
                buf = (tail + "\n" + para).strip()
            else:
                buf = (buf + "\n" + para).strip() if buf else para
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), source=doc.source))
    return chunks


class KnowledgeBase:
    """Loads, indexes, and retrieves finance-education chunks."""

    def __init__(self, cfg: Config | None = None) -> None:
        cfg = cfg or load_config()
        directory = cfg.path("rag.knowledge_dir", "src/data/knowledge")
        size = int(cfg.get("rag.chunk_size", 800))
        overlap = int(cfg.get("rag.chunk_overlap", 120))
        self._top_k = int(cfg.get("rag.top_k", 4))
        mode = str(cfg.get("rag.embeddings", "auto"))

        docs = load_documents(directory)
        self._chunks = chunk_documents(docs, size, overlap)
        self._backend: Backend | None = None
        if self._chunks:
            self._backend = build_backend(
                [c.text for c in self._chunks], mode=mode
            )
        else:
            logger.warning("No knowledge docs found in %s", directory)
        logger.info(
            "Knowledge base: %d chunks from %d docs",
            len(self._chunks),
            len(docs),
        )

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[RetrievedDoc]:
        """Return the top-k knowledge chunks for a query."""
        if not self._backend:
            return []
        hits = self._backend.query(query, k or self._top_k)
        out: list[RetrievedDoc] = []
        for idx, score in hits:
            if 0 <= idx < len(self._chunks):
                chunk = self._chunks[idx]
                out.append(
                    RetrievedDoc(
                        text=chunk.text,
                        source=chunk.source,
                        score=round(score, 4),
                    )
                )
        return out


_INSTANCE: KnowledgeBase | None = None


def get_knowledge_base(cfg: Config | None = None) -> KnowledgeBase:
    """Return a process-wide :class:`KnowledgeBase` singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = KnowledgeBase(cfg)
    return _INSTANCE
