"""
Unified retrieval for both domains: one index, two collections.

  policy_wordings  coverage, exclusions, sub-limits -> motor claims
  annual_reports   audited figures, auditor opinion -> underwriting

Status:
  - Chunking and InMemoryTfidfStore: real, tested, dependency-free
    (scikit-learn is already a requirement). Good enough to build and
    test everything above retrieval today.
  - PgVectorStore: scaffold only. The pgvector database is already
    running in docker-compose.yml; implementing this class is Phase 2
    (see README build order). It needs sentence-transformers (e.g.
    BAAI/bge-small-en-v1.5) and the `pgvector` Python package, which are
    kept out of requirements.txt so the default install stays light.
  - policy.py's stub is intentionally NOT rewired yet. When PgVectorStore
    works, make PolicyService.get_excluded_categories query the
    policy_wordings collection; its signature, and pipeline.py, stay the
    same.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.case import Sensitivity

POLICY_WORDINGS = "policy_wordings"
ANNUAL_REPORTS = "annual_reports"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    collection: str
    text: str
    source: str
    page: int | None
    sensitivity: Sensitivity


def chunk_text(text: str, source: str, collection: str, sensitivity: Sensitivity,
               page: int | None = None, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Paragraph-aware fixed-size chunking: split on blank lines, pack
    paragraphs up to max_chars, carry `overlap` characters forward so a
    figure near a boundary keeps its label."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        while len(p) > max_chars:  # very long paragraph: hard split
            head, p = p[:max_chars], p[max_chars - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(p) + 2 > max_chars and current:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + p if overlap else p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)

    def cid(i: int, body: str) -> str:
        return f"{collection}:{hashlib.sha1(f'{source}|{page}|{i}|{body}'.encode()).hexdigest()[:10]}"

    return [Chunk(cid(i, c), collection, c, source, page, sensitivity) for i, c in enumerate(chunks)]


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, collection: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


class InMemoryTfidfStore(VectorStore):
    """Lexical baseline. Refits on add, which is fine for a few thousand
    chunks and keeps it simple. Also the honest baseline to beat in the
    RAG evaluation: dense embeddings have to do better than this."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])

    def search(self, query: str, collection: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self._chunks or self._vectorizer is None:
            return []
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(
            ((c, float(s)) for c, s in zip(self._chunks, scores) if c.collection == collection),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [pair for pair in ranked[:k] if pair[1] > 0]


class PgVectorStore(VectorStore):
    """Phase 2. Sketch of the intended implementation:

      - table chunk(chunk_id PK, collection, text, source, page,
        sensitivity, embedding vector(384))
      - embed with sentence-transformers BAAI/bge-small-en-v1.5
        (fits easily on CPU or the 4 GB GPU)
      - search: ORDER BY embedding <=> :query_embedding LIMIT k,
        filtered by collection
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PgVectorStore is a Phase 2 scaffold -- use InMemoryTfidfStore until it is built.")

    def add(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, query: str, collection: str, k: int = 5):  # pragma: no cover
        raise NotImplementedError


class RAGService:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or InMemoryTfidfStore()

    def index_document(self, text: str, source: str, collection: str, sensitivity: Sensitivity, page: int | None = None) -> list[Chunk]:
        chunks = chunk_text(text, source, collection, sensitivity, page=page)
        self.store.add(chunks)
        return chunks

    def retrieve(self, query: str, collection: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return self.store.search(query, collection, k)
