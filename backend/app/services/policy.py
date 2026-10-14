"""
Policy-document handling -- deliberately a stub, not a half-finished RAG
system pretending to be real.

The build plan puts multi-modal document intelligence (treating policy
PDFs as retrievable, embeddable chunks in pgvector) as later-phase work,
after the consistency engine and its evaluation are solid. Faking it now
would be worse than being upfront that it isn't built yet: this always
returns the same synthetic exclusion set from app/ml/domain.py regardless
of which policy PDF was uploaded.

When you build the real version: chunk the uploaded policy PDF, embed
chunks with a Sentence-Transformers model, store in pgvector, and query
for "what categories does this policy exclude" per claim instead of this
fixed set. Nothing outside this file needs to change -- get_excluded_categories
is the only thing pipeline.py calls.
"""
from __future__ import annotations

from app.ml.domain import POLICY_EXCLUDED_CATEGORIES


class PolicyService:
    def get_excluded_categories(self, policy_path: str | None) -> set[str]:
        return set(POLICY_EXCLUDED_CATEGORIES)
