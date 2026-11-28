import json

import pytest

from careflow.policy_index import (
    LocalPolicyIndex,
    PolicyEmbeddingError,
    PolicyIndexConfigurationError,
)
from careflow.policy_ingestion import load_policy_documents
from careflow.policy_models import PolicyDocument, PolicyPageText, PolicySection


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
        category="scheduling",
        version="2.1",
        effective_date="2026-02-01",
        sections=(
            PolicySection(
                name="Appointment Changes",
                text="Appointment changes are handled by the scheduling coordinator.",
                pages=(
                    PolicyPageText(
                        page_number=2,
                        text=(
                            "Appointment changes are handled by the scheduling "
                            "coordinator."
                        ),
                    ),
                    PolicyPageText(
                        page_number=3,
                        text=(
                            "Requests should arrive one business day before the visit. "
                            "Late appointment requests receive human review."
                        ),
                    ),
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
    assert all(result.category == "scheduling" for result in results)
    assert all(result.version == "2.1" for result in results)
    assert all(result.effective_date.isoformat() == "2026-02-01" for result in results)
    assert {result.page_start for result in results} == {2, 3}
    assert all(result.page_start == result.page_end for result in results)


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


def test_skip_if_unchanged_validates_existing_points_without_reembedding(tmp_path):
    class CountingEmbedder:
        def __init__(self):
            self.calls = 0

        def __call__(self, texts):
            self.calls += 1
            return DeterministicFakeEmbedder()(texts)

    embedder = CountingEmbedder()
    documents = load_policy_documents()[:1]
    index = create_index(tmp_path / "validate-index", embedder=embedder)
    try:
        initial = index.index_documents(documents)
        calls_after_initial_index = embedder.calls
        validated = index.index_documents(
            documents,
            skip_if_unchanged=True,
        )
    finally:
        index.close()

    assert calls_after_initial_index == 1
    assert embedder.calls == calls_after_initial_index
    assert validated.chunks_indexed == initial.chunks_indexed
    assert validated.stale_chunks_removed == 0


def test_reindex_cannot_be_combined_with_skip_if_unchanged(tmp_path):
    index = create_index(tmp_path / "invalid-index-options")
    try:
        with pytest.raises(
            PolicyIndexConfigurationError,
            match="cannot be combined",
        ):
            index.index_documents(reindex=True, skip_if_unchanged=True)
    finally:
        index.close()


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


def test_pdf_metadata_and_page_citations_survive_qdrant_indexing(tmp_path):
    documents = load_policy_documents()
    index = create_index(tmp_path / "pdf-policy-index")
    try:
        result = index.index_documents(documents)
        repeated_result = index.index_documents(documents)
        retrieved = index.search(
            "cardiology appointment referral scheduling",
            top_k=5,
            source_id="SYN-POLICY-SPEC-102",
            section_name="Scheduling Directory",
        )
        plan_filtered = index.search(
            "specialist copay plan",
            top_k=10,
            category="copay_schedule",
            plan_id="SYN-PLAN-12",
            version="1.0",
        )
    finally:
        index.close()

    assert result.documents_indexed == 14
    assert result.chunks_indexed == 28
    assert repeated_result.chunks_indexed == 28
    assert repeated_result.stale_chunks_removed == 0
    assert len(retrieved) == 1
    passage = retrieved[0]
    assert passage.category == "specialty_handbook"
    assert passage.version == "1.0"
    assert passage.effective_date.isoformat() == "2026-04-01"
    assert passage.specialty == "cardiology"
    assert passage.plan_ids == ()
    assert passage.page_start == passage.page_end == 1
    assert passage.synthetic_data is True
    assert plan_filtered
    assert all(result.category == "copay_schedule" for result in plan_filtered)
    assert all("SYN-PLAN-12" in result.plan_ids for result in plan_filtered)
    assert all(result.version == "1.0" for result in plan_filtered)


def test_corpus_version_change_requires_explicit_reindex(tmp_path):
    index = create_index(tmp_path / "versioned-corpus-index")
    documents = load_policy_documents()
    index.index_documents(documents)
    changed = (
        documents[0].model_copy(update={"version": "2.0"}),
        *documents[1:],
    )

    with pytest.raises(PolicyIndexConfigurationError, match="corpus changed"):
        index.index_documents(changed)

    index.index_documents(changed, reindex=True)
    updated = index.search(
        "referral intake routing",
        top_k=5,
        source_id=documents[0].source_id,
    )
    index.close()
    assert updated
    assert all(result.version == "2.0" for result in updated)


def test_parser_and_chunking_changes_require_explicit_reindex(tmp_path):
    parser_path = tmp_path / "parser-version-index"
    parser_index = create_index(parser_path)
    parser_index.index_documents()
    parser_index.close()

    changed_parser_index = create_index(
        parser_path, parser_version="pypdf-section-v3"
    )
    with pytest.raises(PolicyIndexConfigurationError, match="incompatible M3"):
        changed_parser_index.index_documents()
    changed_parser_index.index_documents(reindex=True)
    changed_parser_index.close()

    chunking_path = tmp_path / "chunking-version-index"
    chunking_index = create_index(chunking_path)
    chunking_index.index_documents()
    chunking_index.close()

    changed_chunking_index = create_index(chunking_path, max_chunk_chars=120)
    with pytest.raises(PolicyIndexConfigurationError, match="incompatible M3"):
        changed_chunking_index.index_documents()
    changed_chunking_index.index_documents(reindex=True)
    changed_chunking_index.close()


def test_index_schema_change_requires_explicit_reindex(tmp_path):
    path = tmp_path / "schema-version-index"
    index = create_index(path)
    index.index_documents()
    index.close()

    manifest_path = path / ".policy-index-synthetic_policy_handbook.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_schema_version"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    changed_index = create_index(path)
    with pytest.raises(PolicyIndexConfigurationError, match="incompatible M3"):
        changed_index.index_documents()
    changed_index.index_documents(reindex=True)
    changed_index.close()


def test_interrupted_index_sync_requires_explicit_reindex(tmp_path):
    path = tmp_path / "interrupted-index"
    index = create_index(path)
    index.index_documents()
    index.close()

    manifest_path = path / ".policy-index-synthetic_policy_handbook.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_state"] = "indexing"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    interrupted_index = create_index(path)
    with pytest.raises(PolicyIndexConfigurationError, match="incomplete"):
        interrupted_index.search("insurance")
    with pytest.raises(PolicyIndexConfigurationError, match="incompatible M3"):
        interrupted_index.index_documents()
    interrupted_index.index_documents(reindex=True)
    interrupted_index.close()


def test_legacy_m3_index_remains_searchable_and_needs_explicit_reindex(tmp_path):
    path = tmp_path / "legacy-m3-index"
    index = create_index(path)
    index.index_documents()
    index.close()

    manifest_path = path / ".policy-index-synthetic_policy_handbook.json"
    manifest_path.write_text(
        (
            '{"version":1,"collection_name":"synthetic_policy_handbook",'
            f'"embedding_model_id":"{EMBEDDING_MODEL}",'
            f'"embedding_dimension":{EMBEDDING_DIMENSION}'
            "}"
        ),
        encoding="utf-8",
    )

    legacy_index = create_index(path)
    assert legacy_index.search("specialist copay", top_k=1)
    with pytest.raises(PolicyIndexConfigurationError, match="incompatible M3"):
        legacy_index.index_documents()
    legacy_index.index_documents(reindex=True)
    legacy_index.close()
