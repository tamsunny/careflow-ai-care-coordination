import sys
from types import ModuleType

import pytest

from careflow.langfuse_evaluation import (
    LangfuseEvaluationLogger,
    LangfuseLoggingError,
    LangfuseUnavailableError,
    create_langfuse_evaluation_logger_from_environment,
)
from careflow.policy_evaluation import (
    GroundedAnswerEvaluationReport,
    PolicyEvaluationCase,
    PolicyEvaluationExecutionError,
    PolicyEvaluationTarget,
    RetrievalModeMetrics,
    RetrievalEvaluationReport,
    load_policy_evaluation_dataset,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_grounded_answer_evaluation,
    run_policy_retrieval_evaluation,
)
from careflow.policy_ingestion import load_policy_documents
from careflow.policy_models import PolicySearchResult


def target(source_id, section_name, page_number):
    return PolicyEvaluationTarget(
        source_id=source_id,
        section_name=section_name,
        page_number=page_number,
    )


def result(
    source_id,
    section_name,
    page_start,
    *,
    chunk_id=None,
    text="Fictional policy evidence.",
    page_end=None,
):
    return PolicySearchResult(
        chunk_id=chunk_id or f"{source_id}-{section_name}-{page_start}",
        source_id=source_id,
        title="Fictional evaluation policy",
        section_name=section_name,
        text=text,
        score=0.8,
        category="general",
        specialty=None,
        plan_ids=(),
        version="1.0",
        page_start=page_start,
        page_end=page_end or page_start,
        synthetic_data=True,
    )


def small_dataset():
    return (
        PolicyEvaluationCase(
            case_id="two-targets",
            question="Which fictional referral fields are used?",
            expected_behavior="answer",
            expected_passages=(
                target("SYN-POLICY-TEST-101", "Fields", 1),
                target("SYN-POLICY-TEST-102", "Routing", 2),
            ),
        ),
        PolicyEvaluationCase(
            case_id="no-policy-hit",
            question="What fictional specialty copay applies to SYN-PLAN-999?",
            expected_behavior="insufficient_evidence",
            expected_passages=(),
        ),
        PolicyEvaluationCase(
            case_id="clinical-review",
            question="What exact medication dose should I take?",
            expected_behavior="human_review",
            expected_passages=(),
        ),
    )


class FixtureRetriever:
    def __init__(self):
        self.calls = []
        self.mode_results = {}
        self.fail_mode = None

    def _search(self, mode, query, *, top_k, **kwargs):
        self.calls.append((mode, query, top_k, kwargs))
        if self.fail_mode == mode:
            raise RuntimeError("synthetic retrieval failure")
        return self.mode_results.get((mode, query), ())

    def search_dense(self, query, *, top_k, filters):
        return self._search("dense", query, top_k=top_k, filters=filters)

    def search_bm25(self, query, *, top_k, filters):
        return self._search("bm25", query, top_k=top_k, filters=filters)

    def search_hybrid(self, query, *, top_k, candidate_k, filters):
        return self._search(
            "hybrid",
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=filters,
        )

    def search_hybrid_reranked(
        self, query, *, top_k, candidate_k, filters, reranker=None
    ):
        return self._search(
            "hybrid_reranked",
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=filters,
            reranker=reranker,
        )


def test_default_labels_resolve_to_real_synthetic_pdf_sections_and_pages():
    dataset = load_policy_evaluation_dataset()
    documents = {
        document.source_id: document for document in load_policy_documents()
    }

    assert dataset.synthetic_data is True
    assert len(dataset.cases) == 10
    for case in dataset.cases:
        for expected in case.expected_passages:
            document = documents[expected.source_id]
            section = next(
                section
                for section in document.sections
                if section.name == expected.section_name
            )
            assert expected.page_number in {
                page.page_number for page in section.pages
            }


def test_precision_recall_and_mrr_at_five_are_calculated_correctly():
    expected = (
        target("SYN-POLICY-TEST-101", "Fields", 1),
        target("SYN-POLICY-TEST-102", "Routing", 2),
    )
    results = (
        result("SYN-POLICY-TEST-200", "Noise", 1),
        result("SYN-POLICY-TEST-101", "Fields", 1),
        result("SYN-POLICY-TEST-102", "Routing", 2),
    )

    assert precision_at_k(results, expected) == pytest.approx(0.4)
    assert recall_at_k(results, expected) == pytest.approx(1.0)
    assert reciprocal_rank(results, expected) == pytest.approx(0.5)
    assert precision_at_k(results[:1], expected) == 0.0
    assert recall_at_k(results[:1], expected) == 0.0
    assert reciprocal_rank(results[:1], expected) == 0.0


def test_duplicate_chunks_do_not_inflate_precision_or_recall():
    expected = (target("SYN-POLICY-TEST-101", "Fields", 1),)
    results = (
        result("SYN-POLICY-TEST-101", "Fields", 1, chunk_id="first"),
        result("SYN-POLICY-TEST-101", "Fields", 1, chunk_id="duplicate"),
    )

    assert precision_at_k(results, expected) == pytest.approx(0.2)
    assert recall_at_k(results, expected) == pytest.approx(1.0)
    assert reciprocal_rank(results, expected) == 1.0


def test_runner_compares_all_modes_and_reports_latency_and_abstention():
    cases = small_dataset()[:2]
    query = cases[0].question
    expected = cases[0].expected_passages
    noise = result("SYN-POLICY-TEST-200", "Noise", 1)
    first_hit = result("SYN-POLICY-TEST-101", "Fields", 1)
    second_hit = result("SYN-POLICY-TEST-102", "Routing", 2)
    retriever = FixtureRetriever()
    retriever.mode_results[("dense", query)] = (noise, first_hit, second_hit)
    retriever.mode_results[("bm25", query)] = (first_hit, noise, second_hit)
    retriever.mode_results[("hybrid", query)] = (first_hit, second_hit)
    retriever.mode_results[("hybrid_reranked", query)] = (
        second_hit,
        first_hit,
    )

    report = run_policy_retrieval_evaluation(
        retriever,
        cases,
        dataset_version="unit-test-v1",
    )

    assert set(report.mode_reports) == {
        "dense",
        "bm25",
        "hybrid",
        "hybrid_reranked",
    }
    assert report.mode_reports["dense"].precision_at_5 == pytest.approx(0.4)
    assert report.mode_reports["dense"].recall_at_5 == pytest.approx(1.0)
    assert report.mode_reports["dense"].mrr == pytest.approx(0.5)
    assert report.mode_reports["dense"].mrr_at_5 == pytest.approx(0.5)
    assert report.mode_reports["bm25"].mrr == pytest.approx(1.0)
    assert report.mode_reports["hybrid_reranked"].mrr == pytest.approx(1.0)
    assert report.mode_reports["bm25"].abstention_accuracy == 1.0
    assert report.mode_reports["dense"].abstention_accuracy == 1.0
    assert report.mode_reports["hybrid"].mean_latency_ms >= 0
    assert report.mode_reports["hybrid"].queries_evaluated == 2
    assert report.mode_reports["hybrid"].case_results[1].outcome_correct is True
    assert all(call[2] == 5 for call in retriever.calls)
    assert all(call[0] != "human_review" for call in retriever.calls)


def test_runner_skips_clinical_retrieval_and_reports_classifier_measurement():
    retriever = FixtureRetriever()

    report = run_policy_retrieval_evaluation(retriever, small_dataset())

    assert report.human_review_classifier_accuracy == 1.0
    assert all(
        metrics.skipped_human_review_case_count == 1
        for metrics in report.mode_reports.values()
    )
    assert not any(call[1] == "What exact medication dose should I take?" for call in retriever.calls)


def test_runner_surfaces_retrieval_errors_instead_of_returning_partial_metrics():
    retriever = FixtureRetriever()
    retriever.fail_mode = "bm25"

    with pytest.raises(
        PolicyEvaluationExecutionError,
        match="mode 'bm25', case 'two-targets'",
    ):
        run_policy_retrieval_evaluation(retriever, small_dataset()[:1])


def test_runner_requires_candidate_depth_of_at_least_five():
    with pytest.raises(ValueError, match="at least 5"):
        run_policy_retrieval_evaluation(
            FixtureRetriever(),
            small_dataset()[:1],
            candidate_k=4,
        )


def answer_payload(passage, *, quote=None, answer=None, chunk_id=None):
    quote = passage.text if quote is None else quote
    answer = quote if answer is None else answer
    chunk_id = chunk_id or passage.chunk_id
    return {
        "status": "answered",
        "answer": answer,
        "citations": [
            {
                "chunk_id": chunk_id,
                "source_id": passage.source_id,
                "section_name": passage.section_name,
                "page_start": passage.page_start,
                "page_end": passage.page_end,
                "specialty": passage.specialty,
                "plan_ids": list(passage.plan_ids),
                "version": passage.version,
            }
        ],
        "supporting_quotes": [{"chunk_id": chunk_id, "text": quote}],
    }


def test_grounded_answer_evaluation_separates_citation_from_quote_support():
    case = PolicyEvaluationCase(
        case_id="copay-answer",
        question="What is the synthetic copay?",
        expected_behavior="answer",
        expected_passages=(target("SYN-POLICY-COPAY-101", "Schedule", 1),),
    )
    passage = result(
        "SYN-POLICY-COPAY-101",
        "Schedule",
        1,
        text="The synthetic co-pay is $20.",
    )
    valid = answer_payload(passage)
    invalid_quote = answer_payload(
        passage,
        quote="The synthetic co-pay is $999.",
        answer="The synthetic co-pay is $999.",
    )

    report = run_grounded_answer_evaluation(
        (case,),
        {"copay-answer": (passage,)},
        {"copay-answer": valid},
    )
    bad_report = run_grounded_answer_evaluation(
        (case,),
        {"copay-answer": (passage,)},
        {"copay-answer": invalid_quote},
    )

    good_case = report.case_results[0]
    bad_case = bad_report.case_results[0]
    assert good_case.citation_valid is True
    assert good_case.expected_passage_cited is True
    assert good_case.quoted_evidence_valid is True
    assert good_case.unsupported_claims_detected is False
    assert good_case.semantic_groundedness == "not_assessed"
    assert good_case.passed is True
    assert bad_case.citation_valid is True
    assert bad_case.expected_passage_cited is True
    assert bad_case.quoted_evidence_valid is False
    assert bad_case.unsupported_claims_detected is True
    assert bad_case.passed is False


def test_grounded_answer_evaluation_detects_fabricated_citation_id():
    case = PolicyEvaluationCase(
        case_id="citation-answer",
        question="What does the synthetic policy say?",
        expected_behavior="answer",
        expected_passages=(target("SYN-POLICY-TEST-201", "Fields", 1),),
    )
    passage = result(
        "SYN-POLICY-TEST-201",
        "Fields",
        1,
        chunk_id="real-chunk",
        text="Referral fields are synthetic.",
    )
    payload = answer_payload(passage, chunk_id="fake-chunk")

    report = run_grounded_answer_evaluation(
        (case,),
        {"citation-answer": (passage,)},
        {"citation-answer": payload},
    )

    assert report.case_results[0].citation_valid is False
    assert report.case_results[0].expected_passage_cited is False
    assert report.case_results[0].quoted_evidence_valid is False
    assert report.case_results[0].passed is False


def test_grounded_answer_requires_gold_passage_relevance_not_just_valid_citation():
    case = PolicyEvaluationCase(
        case_id="wrong-gold-source",
        question="What does the synthetic policy say?",
        expected_behavior="answer",
        expected_passages=(target("SYN-POLICY-TEST-201", "Fields", 1),),
    )
    passage = result(
        "SYN-POLICY-TEST-202",
        "Routing",
        1,
        chunk_id="retrieved-other-policy",
        text="Referral routing is fictional.",
    )
    report = run_grounded_answer_evaluation(
        (case,),
        {"wrong-gold-source": (passage,)},
        {"wrong-gold-source": answer_payload(passage)},
    )

    outcome = report.case_results[0]
    assert outcome.citation_valid is True
    assert outcome.expected_passage_cited is False
    assert outcome.quoted_evidence_valid is True
    assert outcome.passed is False


@pytest.mark.parametrize(
    ("case", "status", "answer"),
    [
        (
            PolicyEvaluationCase(
                case_id="insufficient-case",
                question="What is not in this fictional handbook?",
                expected_behavior="insufficient_evidence",
                expected_passages=(),
            ),
            "insufficient_evidence",
            "I can't verify that policy or coverage rule from the supplied passages.",
        ),
        (
            small_dataset()[2],
            "human_review",
            "This request needs review by a qualified human. No clinical advice was provided.",
        ),
    ],
)
def test_grounded_answer_evaluation_checks_refusal_outcomes(case, status, answer):
    report = run_grounded_answer_evaluation(
        (case,),
        {case.case_id: ()},
        {
            case.case_id: {
                "status": status,
                "answer": answer,
                "citations": [],
                "supporting_quotes": [],
            }
        },
    )

    assert report.case_results[0].refusal_correct is True
    assert report.case_results[0].expected_outcome_correct is True
    assert report.case_results[0].passed is True
    assert report.refusal_accuracy == 1.0


def retrieval_report():
    return RetrievalModeMetrics(
        mode="dense",
        precision_at_5=0.4,
        recall_at_5=0.8,
        mrr=0.5,
        mean_latency_ms=1.5,
        p50_latency_ms=1.2,
        p95_latency_ms=2.0,
        outcome_accuracy=0.75,
        abstention_accuracy=1.0,
        answerable_case_count=4,
        insufficient_evidence_case_count=1,
        skipped_human_review_case_count=1,
        queries_evaluated=5,
        case_results=(),
    )


def test_langfuse_adapter_logs_aggregate_metrics_only():
    class FakeTrace:
        def __init__(self, trace_id, scores):
            self.trace_id = trace_id
            self.scores = scores

        def score(self, **kwargs):
            self.scores.append({**kwargs, "trace_id": self.trace_id})

    class FakeLangfuse:
        def __init__(self):
            self.scores = []
            self.traces = []

        def trace(self, **kwargs):
            self.traces.append(kwargs)
            return FakeTrace(kwargs["id"], self.scores)

    client = FakeLangfuse()
    report = run_policy_retrieval_evaluation(
        FixtureRetriever(),
        small_dataset()[:1],
        dataset_version="synthetic-v1",
    )
    count = LangfuseEvaluationLogger(client).log_retrieval_report(
        report,
        trace_id="1234567890abcdef1234567890abcdef",
    )

    assert count == len(client.scores)
    assert count > 0
    assert all(
        score["trace_id"] == "1234567890abcdef1234567890abcdef"
        for score in client.scores
    )
    assert all(score["data_type"] == "NUMERIC" for score in client.scores)
    assert len(client.traces) == 1
    assert client.traces[0]["id"] == "1234567890abcdef1234567890abcdef"
    assert set(client.traces[0]["metadata"]) == {
        "dataset_version_hash",
        "case_count",
        "synthetic_data",
    }
    assert client.traces[0]["metadata"]["synthetic_data"] is True
    assert "question" not in client.traces[0]["metadata"]


def test_langfuse_unavailability_is_explicit_and_does_not_change_metrics():
    report = RetrievalModeMetrics(
        mode="dense",
        precision_at_5=0.0,
        recall_at_5=0.0,
        mrr=0.0,
        answerable_case_count=1,
        insufficient_evidence_case_count=0,
        skipped_human_review_case_count=0,
        queries_evaluated=1,
        case_results=(),
    )
    aggregate = RetrievalEvaluationReport(
        dataset_version="synthetic-v1",
        case_count=1,
        mode_reports={"dense": report},
    )

    with pytest.raises(LangfuseUnavailableError, match="trace\\(\\)"):
        LangfuseEvaluationLogger(None).log_retrieval_report(
            aggregate,
            trace_id="1234567890abcdef1234567890abcdef",
        )


def test_langfuse_trace_id_rejects_non_trace_identifiers():
    class FakeLangfuse:
        def create_score(self, **_kwargs):
            pytest.fail("Invalid trace ID must not be sent")

    aggregate = RetrievalEvaluationReport(
        dataset_version="synthetic-v1",
        case_count=1,
        mode_reports={"dense": retrieval_report()},
    )
    with pytest.raises(ValueError, match="32-character lowercase hex"):
        LangfuseEvaluationLogger(FakeLangfuse()).log_retrieval_report(
            aggregate,
            trace_id="not-a-trace-or-credential",
        )


def test_langfuse_write_failure_is_explicit_and_sanitized():
    secret_like_text = "synthetic-backend-error-secret-never-return-it"

    class OfflineLangfuse:
        def trace(self, **_kwargs):
            raise RuntimeError(secret_like_text)

    aggregate = RetrievalEvaluationReport(
        dataset_version="synthetic-v1",
        case_count=1,
        mode_reports={"dense": retrieval_report()},
    )

    with pytest.raises(LangfuseLoggingError) as error:
        LangfuseEvaluationLogger(OfflineLangfuse()).log_retrieval_report(
            aggregate,
            trace_id="1234567890abcdef1234567890abcdef",
        )
    assert secret_like_text not in str(error.value)


def test_nonfatal_langfuse_trace_failure_preserves_evaluation_report():
    secret_like_error = "synthetic-langfuse-secret-like-error"

    class OfflineLangfuse:
        def trace(self, **_kwargs):
            raise RuntimeError(secret_like_error)

    aggregate = RetrievalEvaluationReport(
        dataset_version="synthetic-v1",
        case_count=1,
        mode_reports={"dense": retrieval_report()},
    )
    report_before = aggregate.model_dump_json()

    result = LangfuseEvaluationLogger(OfflineLangfuse()).log_retrieval_report_nonfatal(
        aggregate,
        trace_id="1234567890abcdef1234567890abcdef",
    )

    assert result.success is False
    assert result.error_code == "LangfuseLoggingError"
    assert secret_like_error not in result.model_dump_json()
    assert aggregate.model_dump_json() == report_before


def test_langfuse_environment_factory_uses_only_passed_environment(monkeypatch):
    seen = {}

    class FakeTrace:
        def score(self, **_kwargs):
            return None

    class FakeLangfuse:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def trace(self, **_kwargs):
            return FakeTrace()

    fake_module = ModuleType("langfuse")
    fake_module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    logger = create_langfuse_evaluation_logger_from_environment(
        {
            "LANGFUSE_PUBLIC_KEY": "synthetic-public-key",
            "LANGFUSE_SECRET_KEY": "synthetic-secret-key",
            "LANGFUSE_HOST": "https://langfuse.invalid",
        }
    )

    assert isinstance(logger, LangfuseEvaluationLogger)
    assert seen == {
        "public_key": "synthetic-public-key",
        "secret_key": "synthetic-secret-key",
        "host": "https://langfuse.invalid",
    }


def test_langfuse_logs_grounded_answer_aggregates_and_flushes():
    class FakeTrace:
        def __init__(self, trace_id, scores):
            self.trace_id = trace_id
            self.scores = scores

        def score(self, **kwargs):
            self.scores.append({**kwargs, "trace_id": self.trace_id})

    class FakeLangfuse:
        def __init__(self):
            self.scores = []
            self.traces = []
            self.flushed = False

        def trace(self, **kwargs):
            self.traces.append(kwargs)
            return FakeTrace(kwargs["id"], self.scores)

        def flush(self):
            self.flushed = True

    client = FakeLangfuse()
    report = GroundedAnswerEvaluationReport(
        dataset_version="synthetic-answer-v1",
        case_count=3,
        citation_validity_rate=0.8,
        quoted_evidence_rate=0.75,
        unsupported_claim_rate=0.2,
        refusal_accuracy=1.0,
        expected_outcome_accuracy=0.9,
        case_results=(),
    )

    count = LangfuseEvaluationLogger(client).log_grounded_answer_report(
        report,
        trace_id="1234567890abcdef1234567890abcdef",
    )

    assert count == 5
    assert client.flushed is True
    assert len(client.traces) == 1
    assert client.traces[0]["name"] == "careflow-m4-policy-answer-evaluation"
    assert {item["name"] for item in client.scores} == {
        "policy_answers.citation_validity_rate",
        "policy_answers.quoted_evidence_rate",
        "policy_answers.unsupported_claim_rate",
        "policy_answers.refusal_accuracy",
        "policy_answers.expected_outcome_accuracy",
    }
