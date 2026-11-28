import json

import pytest

from careflow import policy_evaluation_cli as cli
from careflow.langfuse_evaluation import LangfuseLoggingResult
from careflow.policy_evaluation import (
    RetrievalEvaluationReport,
    RetrievalModeMetrics,
)
from careflow.policy_providers import EmbeddingProviderConfig
from careflow.policy_retrieval import CrossEncoderReranker


def report():
    metrics = RetrievalModeMetrics(
        mode="dense",
        precision_at_5=0.2,
        recall_at_5=1.0,
        mrr=1.0,
        mrr_at_5=1.0,
        mean_latency_ms=2.0,
        p50_latency_ms=2.0,
        p95_latency_ms=2.0,
        answerable_case_count=1,
        insufficient_evidence_case_count=0,
        skipped_human_review_case_count=0,
        queries_evaluated=1,
        case_results=(),
    )
    return RetrievalEvaluationReport(
        dataset_version="synthetic-test-v1",
        case_count=1,
        mode_reports={"dense": metrics},
    )


def test_acceptance_runner_uses_dedicated_m4_index_and_injected_dependencies(
    monkeypatch,
):
    calls = {}
    embedding_config = EmbeddingProviderConfig(
        model_id="provider/synthetic-model",
        dimension=3,
    )
    fake_reranker = CrossEncoderReranker(lambda pairs: [0.0] * len(pairs))

    class FakeIndex:
        def __init__(self, embedder, **kwargs):
            calls["index_options"] = kwargs
            calls["embedder"] = embedder

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls["closed"] = True

        def index_documents(self, documents, *, skip_if_unchanged):
            calls["documents"] = documents
            calls["skip_if_unchanged"] = skip_if_unchanged

    monkeypatch.setattr(
        cli,
        "load_embedding_provider_config",
        lambda: embedding_config,
    )
    monkeypatch.setattr(cli, "create_litellm_embedder", lambda _config: "embedder")
    monkeypatch.setattr(
        cli,
        "create_cpu_cross_encoder_reranker",
        lambda: fake_reranker,
    )
    monkeypatch.setattr(cli, "load_policy_documents", lambda: ("synthetic-pdfs",))
    monkeypatch.setattr(
        cli,
        "load_policy_evaluation_dataset",
        lambda: "synthetic-evaluation-dataset",
    )
    monkeypatch.setattr(cli, "LocalPolicyIndex", FakeIndex)

    def run_evaluation(retriever, dataset, *, reranker):
        calls["retriever"] = retriever
        calls["dataset"] = dataset
        calls["reranker"] = reranker
        return report()

    monkeypatch.setattr(cli, "run_policy_retrieval_evaluation", run_evaluation)

    result, logging_result = cli.run_acceptance_evaluation(index_version="v2")

    assert result.dataset_version == "synthetic-test-v1"
    assert logging_result is None
    expected_directory, expected_collection = cli.versioned_m4_index_target("v2")
    assert calls["index_options"]["collection_name"] == expected_collection
    assert calls["index_options"]["persist_directory"] == expected_directory
    assert calls["index_options"]["parser_version"] == cli.PARSER_VERSION
    assert calls["documents"] == ("synthetic-pdfs",)
    assert calls["skip_if_unchanged"] is True
    assert calls["dataset"] == "synthetic-evaluation-dataset"
    assert calls["reranker"] is fake_reranker
    assert calls["closed"] is True


def test_versioned_m4_target_rejects_path_like_versions():
    with pytest.raises(ValueError, match="short alphanumeric M4 version"):
        cli.versioned_m4_index_target("..\\m3")


def test_cli_emits_report_even_when_optional_logging_fails(monkeypatch, capsys):
    evaluated = report()
    logging_result = LangfuseLoggingResult(
        success=False,
        trace_id="1234567890abcdef1234567890abcdef",
        error_code="LangfuseLoggingError",
    )
    monkeypatch.setattr(
        cli,
        "run_acceptance_evaluation",
        lambda *, log_langfuse, index_version: (evaluated, logging_result),
    )

    assert cli.main(["--log-langfuse"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["retrieval_report"]["dataset_version"] == "synthetic-test-v1"
    assert output["retrieval_report"]["mode_reports"]["dense"]["mrr_at_5"] == 1.0
    assert output["langfuse"]["success"] is False
    assert output["langfuse"]["error_code"] == "LangfuseLoggingError"
