from app.core.case import Sensitivity
from app.services.rag import ANNUAL_REPORTS, POLICY_WORDINGS, RAGService, chunk_text


def test_chunking_respects_max_size_and_ids_are_stable():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    a = chunk_text(text, "doc", ANNUAL_REPORTS, Sensitivity.PUBLIC, max_chars=500)
    b = chunk_text(text, "doc", ANNUAL_REPORTS, Sensitivity.PUBLIC, max_chars=500)
    assert len(a) > 1 and all(len(c.text) <= 500 + 150 + 2 for c in a)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_retrieval_is_scoped_to_collection():
    rag = RAGService()
    rag.index_document("Exclusions: cosmetic detailing and consumables are not covered.", "policy.pdf",
                       POLICY_WORDINGS, Sensitivity.PUBLIC)
    rag.index_document("Revenue from operations was 1,234 crore. The auditor issued an unqualified opinion.",
                       "ar.pdf", ANNUAL_REPORTS, Sensitivity.PUBLIC)
    hits = rag.retrieve("which items are excluded from cover", POLICY_WORDINGS)
    assert hits and all(c.collection == POLICY_WORDINGS for c, _ in hits)
    hits = rag.retrieve("auditor opinion", ANNUAL_REPORTS)
    assert hits and "unqualified" in hits[0][0].text
