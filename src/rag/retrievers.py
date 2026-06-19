"""Retrieval backends behind one tiny ``query`` interface.

- ``SemanticBackend``: sentence-transformers embeddings + a FAISS
  inner-product index (requires the optional extras).
- ``TfidfBackend``: scikit-learn TF-IDF + cosine similarity, always
  available and dependency-light.

``build_backend`` prefers the semantic backend when the extras are
installed, otherwise transparently falls back to TF-IDF.
"""

from __future__ import annotations

from typing import Protocol

from ..core.logging_utils import get_logger

logger = get_logger(__name__)


class Backend(Protocol):
    name: str

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        ...


class TfidfBackend:
    """Lexical retriever: TF-IDF vectors + cosine similarity."""

    name = "tfidf"

    def __init__(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(texts)

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        from sklearn.metrics.pairwise import linear_kernel

        q = self._vec.transform([text])
        sims = linear_kernel(q, self._matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


class SemanticBackend:
    """Dense retriever: sentence-transformers + FAISS (cosine)."""

    name = "semantic-faiss"

    def __init__(
        self, texts: list[str], model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        self._model = SentenceTransformer(model_name)
        emb = self._model.encode(texts, normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        q = self._model.encode([text], normalize_embeddings=True)
        q = self._np.asarray(q, dtype="float32")
        scores, idxs = self._index.search(q, k)
        out: list[tuple[int, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i >= 0:
                out.append((int(i), float(s)))
        return out


def build_backend(texts: list[str], mode: str = "auto") -> Backend:
    """Return the best available retriever for ``mode``."""
    if mode == "tfidf":
        return TfidfBackend(texts)
    try:
        backend = SemanticBackend(texts)
        logger.info("RAG retriever: FAISS + sentence-transformers")
        return backend
    except Exception as exc:
        if mode == "sentence-transformers":
            logger.warning("Semantic backend unavailable: %s", exc)
        else:
            logger.info(
                "Semantic extras absent; using TF-IDF retriever."
            )
        return TfidfBackend(texts)
