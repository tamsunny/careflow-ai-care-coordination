import re

import pytest

from careflow.policy_index import LocalPolicyIndex
from careflow.policy_models import (
    PolicyDocument,
    PolicyPageText,
    PolicySearchFilters,
    PolicySearchResult,
    PolicySection,
)
from careflow.policy_retrieval import (
    CrossEncoderReranker,
    PolicyHybridRetriever,
    PolicyRerankerError,
    PolicyRetrievalConfigurationError,
)

EMBEDDING_MODEL = "careflow-hybrid-test-v1"
TERMS = (
    "appointment",
    "cardiology",
    "copay",
    "dermatology",
    "referral",
    "scheduling",
    "specialist",
    "syn",
    "plan",
    "12",
    "120",
)


class DeterministicEmbedder:
    def __call__(self, texts):
        return [
            [
                0.01 + re.findall(r"[a-z0-9]+", text.casefold()).count(term)
                for term in TERMS
            ]
            for text in texts
        ]


def create_document(
    source_id,
    text,
    *,
    specialty="cardiology",
    plan_ids=("SYN-PLAN-12",),
    version="1.0",
    category="copay_schedule",
):
    return PolicyDocument(
        source_id=source_id,
        title=f"Fictional policy {source_id}",
        category=category,
        specialty=specialty,
        plan_ids=plan_ids,
        version=version,
        effective_date="2026-04-01",
        sections=(
            PolicySection(
                name="Synthetic Plan Rules",
                text=text,
                pages=(PolicyPageText(page_number=7, text=text),),
            ),
        ),
    )


def create_index(path, documents):
    index = LocalPolicyIndex(
        DeterministicEmbedder(),
        embedding_model_id=EMBEDDING_MODEL,
        embedding_dimension=len(TERMS),
        persist_directory=path,
        collection_name="synthetic_hybrid_test",
    )
    index.index_documents(documents)
    return index


def test_bm25_matches_exact_synthetic_identifiers_without_near_id_leakage(tmp_path):
    documents = (
        create_document(
            "SYN-POLICY-TEST-201",
            "SYN-PLAN-12 has a fictional cardiology specialist copay.",
        ),
        create_document(
            "SYN-POLICY-TEST-202",
            "SYN-PLAN-120 has a fictional cardiology specialist copay.",
            plan_ids=("SYN-PLAN-120",),
        ),
    )
    index = create_index(tmp_path / "exact-identifier", documents)
    retriever = PolicyHybridRetriever(index)

    lexical = retriever.search_bm25("SYN-PLAN-12", top_k=10)

    assert lexical
    assert "SYN-POLICY-TEST-201" in {result.source_id for result in lexical}
    assert "SYN-POLICY-TEST-202" not in {result.source_id for result in lexical}
    assert all(result.bm25_score is not None for result in lexical)
    assert all(result.dense_score is None for result in lexical)
    index.close()


def test_filters_are_consistent_across_all_retrieval_modes(tmp_path):
    documents = (
        create_document(
            "SYN-POLICY-TEST-211",
            "SYN-PLAN-12 cardiology specialist copay and appointment scheduling.",
            version="1.0",
        ),
        create_document(
            "SYN-POLICY-TEST-212",
            "SYN-PLAN-12 cardiology specialist copay and appointment scheduling.",
            version="2.0",
        ),
        create_document(
            "SYN-POLICY-TEST-213",
            "SYN-PLAN-120 cardiology specialist copay and appointment scheduling.",
            plan_ids=("SYN-PLAN-120",),
            version="2.0",
        ),
        create_document(
            "SYN-POLICY-TEST-214",
            "SYN-PLAN-12 dermatology specialist copay and appointment scheduling.",
            specialty="dermatology",
            version="2.0",
        ),
    )
    index = create_index(tmp_path / "consistent-filters", documents)
    reranker = CrossEncoderReranker(lambda pairs: [0.5] * len(pairs))
    retriever = PolicyHybridRetriever(index, reranker=reranker)
    filters = PolicySearchFilters(
        specialty="cardiology",
        plan_id="SYN-PLAN-12",
        version="2.0",
    )
    query = "SYN-PLAN-12 cardiology specialist copay"

    results_by_mode = (
        retriever.search_dense(query, top_k=10, filters=filters),
        retriever.search_bm25(query, top_k=10, filters=filters),
        retriever.search_hybrid(
            query, top_k=10, candidate_k=10, filters=filters
        ),
        retriever.search_hybrid_reranked(
            query, top_k=10, candidate_k=10, filters=filters
        ),
    )
    index.close()

    for results in results_by_mode:
        assert results
        assert {result.source_id for result in results} == {"SYN-POLICY-TEST-212"}
        assert all(result.specialty == "cardiology" for result in results)
        assert all(result.plan_ids == ("SYN-PLAN-12",) for result in results)
        assert all(result.version == "2.0" for result in results)
        assert all(result.page_start == 7 for result in results)


def test_rrf_and_reranker_keep_component_scores_separate(tmp_path):
    documents = (
        create_document(
            "SYN-POLICY-TEST-221",
            "SYN-PLAN-12 fictional specialist copay and referral scheduling.",
        ),
        create_document(
            "SYN-POLICY-TEST-222",
            "SYN-PLAN-120 fictional specialist copay and referral scheduling.",
            plan_ids=("SYN-PLAN-120",),
        ),
        create_document(
            "SYN-POLICY-TEST-223",
            "SYN-PLAN-12 cardiology appointment availability and referral workflow.",
            category="specialty_handbook",
        ),
    )
    index = create_index(tmp_path / "score-separation", documents)
    reranker = CrossEncoderReranker(
        lambda pairs: [
            0.95 if "SYN-PLAN-120" in passage else 0.15
            for _, passage in pairs
        ]
    )
    retriever = PolicyHybridRetriever(index, reranker=reranker)
    query = "SYN-PLAN-12 specialist copay referral"

    dense = retriever.search_dense(query, top_k=10)
    lexical = retriever.search_bm25(query, top_k=10)
    hybrid = retriever.search_hybrid(query, top_k=10, candidate_k=10)
    reranked = retriever.search_hybrid_reranked(
        query, top_k=10, candidate_k=10
    )
    index.close()

    assert dense and dense[0].dense_score is not None
    assert all(result.bm25_score is None for result in dense)
    assert lexical and lexical[0].bm25_score is not None
    assert all(result.dense_score is None and result.score is None for result in lexical)
    assert hybrid and hybrid[0].rrf_score is not None
    assert all(result.reranker_score is None for result in hybrid)
    assert reranked and reranked[0].reranker_score == 0.95
    assert reranked[0].source_id == "SYN-POLICY-TEST-222"
    assert reranked[0].rrf_score is not None
    assert reranked[0].bm25_score is not None


def test_ranking_ties_are_deterministic_by_chunk_id(tmp_path):
    text = "Fictional referral scheduling and specialist appointment details."
    documents = (
        create_document("SYN-POLICY-TEST-231", text),
        create_document("SYN-POLICY-TEST-232", text),
        create_document("SYN-POLICY-TEST-233", text),
    )
    index = create_index(tmp_path / "ranking-ties", documents)
    retriever = PolicyHybridRetriever(index, rrf_k=10)

    dense = retriever.search_dense("referral scheduling", top_k=10)
    lexical = retriever.search_bm25("referral scheduling", top_k=10)
    hybrid = retriever.search_hybrid(
        "referral scheduling", top_k=10, candidate_k=10
    )
    index.close()

    assert [result.chunk_id for result in dense] == sorted(
        result.chunk_id for result in dense
    )
    assert [result.chunk_id for result in lexical] == sorted(
        result.chunk_id for result in lexical
    )
    assert [result.chunk_id for result in hybrid] == sorted(
        result.chunk_id for result in hybrid
    )


def test_empty_results_and_invalid_queries_are_handled_explicitly(tmp_path):
    index = LocalPolicyIndex(
        DeterministicEmbedder(),
        embedding_model_id=EMBEDDING_MODEL,
        embedding_dimension=len(TERMS),
        persist_directory=tmp_path / "empty-index",
        collection_name="synthetic_hybrid_empty",
    )
    reranker = CrossEncoderReranker(
        lambda pairs: pytest.fail("No candidates should be reranked")
    )
    retriever = PolicyHybridRetriever(index, reranker=reranker)

    assert retriever.search_dense("unmatched", top_k=5) == ()
    assert retriever.search_bm25("unmatched", top_k=5) == ()
    assert retriever.search_bm25("!!!", top_k=5) == ()
    assert retriever.search_hybrid("unmatched", top_k=5, candidate_k=5) == ()
    assert retriever.search_hybrid_reranked(
        "unmatched", top_k=5, candidate_k=5
    ) == ()
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search_bm25("   ")
    index.close()


def test_reranker_rejects_bad_scores_and_missing_configuration(tmp_path):
    index = LocalPolicyIndex(
        DeterministicEmbedder(),
        embedding_model_id=EMBEDDING_MODEL,
        embedding_dimension=len(TERMS),
        persist_directory=tmp_path / "reranker-errors",
        collection_name="synthetic_no_reranker",
    )
    retriever = PolicyHybridRetriever(index)
    with pytest.raises(PolicyRetrievalConfigurationError, match="Inject"):
        retriever.search_hybrid_reranked(
            "policy", top_k=1, candidate_k=1
        )
    index.close()

    candidate = PolicySearchResult(
        chunk_id="candidate",
        source_id="SYN-POLICY-TEST-241",
        title="Fictional test policy",
        section_name="Test section",
        text="Fictional policy text.",
        score=0.5,
    )
    with pytest.raises(PolicyRerankerError, match="one score per candidate"):
        CrossEncoderReranker(lambda _pairs: []).rerank("policy", (candidate,))
    with pytest.raises(PolicyRerankerError, match="finite numeric"):
        CrossEncoderReranker(lambda _pairs: [float("nan")]).rerank(
            "policy", (candidate,)
        )
