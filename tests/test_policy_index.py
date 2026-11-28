import pytest

from careflow.policy_index import (
    LocalPolicyIndex,
    PolicyEmbeddingError,
    PolicyIndexConfigurationError,
)
from careflow.policy_models import PolicyDocument, PolicySection


EMBEDDING_MODEL = "careflow-test-embedding-v1"
EMBEDDING_DIMENSION = 8
TERMS = (
    "appointment",
    "referral",
    "insurance",
    "copay",
    "eligibility",
    "specialist",
    "scheduling",
    "status",
)


class DeterministicFakeEmbedder:
    def __call__(self, texts):
        return [
            [
                float(text.lower().count(term))
                for term in TERMS
            ]
            for text in texts
        ]


def create_index(path, embedder=None, **options):
    return LocalPolicyIndex(
        embedder or DeterministicFakeEmbedder(),
        embedding_model_id=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        persist_directory=path,
        **options,
    )


def test_indexing_chunks_fictional_sections_with_metadata(tmp_path):
    document = PolicyDocument(
        source_id="SYN-POLICY-TEST-001",
        title="Fictional Appointment Policy",
        sections=(
            PolicySection(
                name="Appointment Changes",
                text=(
                    "Appointment changes are handled by the scheduling coordinator. "
                    "Requests should arrive one business day before the visit. "
                    "Late appointment requests receive human review."
                ),
            ),
        ),
    )
    index = create_index(tmp_path / "policy-index", max_chunk_chars=80)
    try:
        outcome = index.index_documents((document,))
        results = index.search(
            "appointment changes",
            top_k=10,
            source_id="SYN-POLICY-TEST-001",
        )
    finally:
        index.close()

    assert outcome.documents_indexed == 1
    assert outcome.chunks_indexed > 1
    assert len(results) == outcome.chunks_indexed
    assert all(result.source_id == document.source_id for result in results)
    assert all(result.title == document.title for result in results)
    assert all(result.section_name == "Appointment Changes" for result in results)
    assert all(len(result.text) <= 80 for result in results)
    assert all(isinstance(result.score, float) for result in results)


def test_search_supports_top_k_and_metadata_filters(tmp_path):
    index = create_index(tmp_path / "filtered-index")
    try:
        index.index_documents()
        all_results = index.search("specialist copay plan", top_k=20)
        limited_results = index.search("specialist copay plan", top_k=2)
        filtered_results = index.search(
            "specialist copay plan",
            top_k=10,
            source_id="SYN-POLICY-INS-001",
            section_name="Synthetic Specialist Copays",
        )
    finally:
        index.close()

    assert len(all_results) == 6
    assert len(limited_results) == 2
    assert filtered_results
    assert all(
        result.source_id == "SYN-POLICY-INS-001"
        and result.section_name == "Synthetic Specialist Copays"
        for result in filtered_results
    )
    assert filtered_results[0].score >= filtered_results[-1].score


def test_index_persists_across_reopen_and_repeated_indexing_is_idempotent(tmp_path):
    database_path = tmp_path / "persistent-policy-index"
    first_index = create_index(database_path)
    first_outcome = first_index.index_documents()
    initial_results = first_index.search(
        "insurance eligibility", top_k=20, source_id="SYN-POLICY-INS-001"
    )
    first_index.close()

    reopened_index = create_index(database_path)
    second_outcome = reopened_index.index_documents()
    persisted_results = reopened_index.search(
        "insurance eligibility", top_k=20, source_id="SYN-POLICY-INS-001"
    )
    reopened_index.close()

    assert first_outcome.chunks_indexed == second_outcome.chunks_indexed == 6
    assert second_outcome.stale_chunks_removed == 0
    assert len(initial_results) == len(persisted_results) == 2
    assert {result.chunk_id for result in persisted_results} == {
        result.chunk_id for result in initial_results
    }


def test_missing_document_filter_returns_no_results(tmp_path):
    index = create_index(tmp_path / "missing-policy-index")
    try:
        index.index_documents()
        results = index.search(
            "synthetic policy",
            source_id="SYN-POLICY-MISSING-001",
        )
    finally:
        index.close()

    assert results == ()


def test_embedding_dimensions_are_validated_before_upsert(tmp_path):
    class WrongDimensionEmbedder:
        def __call__(self, texts):
            return [[0.1, 0.2] for _ in texts]

    index = create_index(tmp_path / "invalid-vector-index", WrongDimensionEmbedder())
    with pytest.raises(PolicyEmbeddingError, match="dimension"):
        index.index_documents()
    index.close()

    reopened_index = create_index(tmp_path / "invalid-vector-index")
    try:
        assert reopened_index.search("insurance", top_k=5) == ()
    finally:
        reopened_index.close()


def test_existing_index_rejects_changed_embedding_model(tmp_path):
    path = tmp_path / "model-bound-index"
    index = create_index(path)
    index.index_documents()
    index.close()

    with pytest.raises(PolicyIndexConfigurationError, match="different"):
        LocalPolicyIndex(
            DeterministicFakeEmbedder(),
            embedding_model_id="different-test-model",
            embedding_dimension=EMBEDDING_DIMENSION,
            persist_directory=path,
        )
