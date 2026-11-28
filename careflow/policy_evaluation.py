"""Synthetic retrieval and grounded-answer evaluation for M4."""

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .intake import is_clinical_advice_request
from .policy_answers import (
    HUMAN_REVIEW_ANSWER,
    INSUFFICIENT_EVIDENCE_ANSWER,
    GroundedPolicyAnswer,
)
from .policy_models import PolicySearchFilters, PolicySearchResult
from .policy_retrieval import CrossEncoderReranker, PolicyHybridRetriever

RETRIEVAL_K = 5
DEFAULT_EVALUATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "policies"
    / "evaluation_cases.json"
)
RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_reranked"]
ExpectedBehavior = Literal["answer", "insufficient_evidence", "human_review"]
SEMANTIC_GROUNDEDNESS_NOT_ASSESSED = "not_assessed"


class PolicyEvaluationError(RuntimeError):
    """Base error for synthetic policy evaluation."""


class PolicyEvaluationDataError(PolicyEvaluationError):
    """The labeled synthetic evaluation set is invalid."""


class PolicyEvaluationExecutionError(PolicyEvaluationError):
    """A retrieval mode failed; partial evaluation results are not returned."""


class PolicyEvaluationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(
        min_length=16,
        max_length=80,
        pattern=r"^SYN-POLICY-[A-Z0-9-]+$",
    )
    section_name: str = Field(min_length=1, max_length=120)
    page_number: int = Field(ge=1)


class PolicyEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=3, max_length=100, pattern=r"^[a-z0-9-]+$")
    question: str = Field(min_length=1, max_length=1000)
    expected_behavior: ExpectedBehavior
    expected_passages: tuple[PolicyEvaluationTarget, ...]
    filters: PolicySearchFilters = Field(default_factory=PolicySearchFilters)

    @model_validator(mode="after")
    def validate_expected_passages(self) -> "PolicyEvaluationCase":
        keys = [
            (item.source_id, item.section_name, item.page_number)
            for item in self.expected_passages
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Expected passage labels must be unique")
        if self.expected_behavior == "answer" and not self.expected_passages:
            raise ValueError("Answerable cases require at least one labeled passage")
        if self.expected_behavior != "answer" and self.expected_passages:
            raise ValueError("Refusal cases must not have relevant passage labels")
        if (
            self.expected_behavior == "human_review"
            and not is_clinical_advice_request(self.question)
        ):
            raise ValueError("Human-review cases must match the clinical-advice guard")
        return self


class PolicyEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1]
    dataset_version: str = Field(min_length=1, max_length=80)
    synthetic_data: Literal[True]
    cases: tuple[PolicyEvaluationCase, ...] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> "PolicyEvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique")
        return self


class RetrievalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    expected_behavior: ExpectedBehavior
    latency_ms: float | None = Field(default=None, ge=0)
    result_count: int = Field(ge=0)
    relevant_ranks: tuple[int, ...] = ()
    precision_at_5: float | None = Field(default=None, ge=0, le=1)
    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    reciprocal_rank: float | None = Field(default=None, ge=0, le=1)
    outcome_correct: bool | None = None


class RetrievalModeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RetrievalMode
    precision_at_5: float | None = Field(default=None, ge=0, le=1)
    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    mrr: float | None = Field(default=None, ge=0, le=1)
    mrr_at_5: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float | None = Field(default=None, ge=0)
    p50_latency_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    outcome_accuracy: float | None = Field(default=None, ge=0, le=1)
    abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    answerable_case_count: int = Field(ge=0)
    insufficient_evidence_case_count: int = Field(ge=0)
    skipped_human_review_case_count: int = Field(ge=0)
    queries_evaluated: int = Field(ge=0)
    case_results: tuple[RetrievalCaseResult, ...]


class RetrievalEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    case_count: int = Field(ge=1)
    human_review_classifier_accuracy: float | None = Field(default=None, ge=0, le=1)
    mode_reports: dict[str, RetrievalModeMetrics]


class GroundedAnswerCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    expected_behavior: ExpectedBehavior
    answer_status: str | None = None
    citation_valid: bool | None = None
    expected_passage_cited: bool | None = None
    quoted_evidence_valid: bool | None = None
    unsupported_claims_detected: bool | None = None
    refusal_correct: bool | None = None
    expected_outcome_correct: bool
    semantic_groundedness: Literal["not_assessed"] = SEMANTIC_GROUNDEDNESS_NOT_ASSESSED
    passed: bool


class GroundedAnswerEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    case_count: int = Field(ge=1)
    citation_validity_rate: float | None = Field(default=None, ge=0, le=1)
    expected_passage_citation_rate: float | None = Field(default=None, ge=0, le=1)
    quoted_evidence_rate: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    refusal_accuracy: float | None = Field(default=None, ge=0, le=1)
    expected_outcome_accuracy: float | None = Field(default=None, ge=0, le=1)
    semantic_groundedness: Literal["not_assessed"] = SEMANTIC_GROUNDEDNESS_NOT_ASSESSED
    case_results: tuple[GroundedAnswerCaseResult, ...]


def load_policy_evaluation_dataset(
    path: str | Path = DEFAULT_EVALUATION_PATH,
) -> PolicyEvaluationDataset:
    """Load and validate the source-controlled synthetic evaluation labels."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return PolicyEvaluationDataset.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise PolicyEvaluationDataError(
            "The synthetic policy evaluation dataset is unreadable or invalid"
        ) from exc


def precision_at_k(
    results: Sequence[PolicySearchResult],
    expected: Sequence[PolicyEvaluationTarget],
    *,
    k: int = RETRIEVAL_K,
) -> float:
    _validate_k(k)
    return len(_matched_target_keys(results, expected, k=k)) / k


def recall_at_k(
    results: Sequence[PolicySearchResult],
    expected: Sequence[PolicyEvaluationTarget],
    *,
    k: int = RETRIEVAL_K,
) -> float:
    _validate_k(k)
    if not expected:
        raise ValueError("Recall requires at least one expected passage")
    return len(_matched_target_keys(results, expected, k=k)) / len(expected)


def reciprocal_rank(
    results: Sequence[PolicySearchResult],
    expected: Sequence[PolicyEvaluationTarget],
    *,
    k: int = RETRIEVAL_K,
) -> float:
    _validate_k(k)
    expected_keys = _target_key_set(expected)
    if not expected_keys:
        raise ValueError("Reciprocal rank requires at least one expected passage")
    for rank, result in enumerate(results[:k], start=1):
        if _matching_target_key(result, expected_keys) is not None:
            return 1.0 / rank
    return 0.0


def run_policy_retrieval_evaluation(
    retriever: PolicyHybridRetriever,
    dataset: PolicyEvaluationDataset | Sequence[PolicyEvaluationCase],
    *,
    dataset_version: str | None = None,
    candidate_k: int = 20,
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalEvaluationReport:
    """Compare four retrieval modes without indexing or mutating the collection."""
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise ValueError("candidate_k must be an integer")
    if candidate_k < RETRIEVAL_K:
        raise ValueError("candidate_k must be at least 5")

    cases, resolved_version = _resolve_dataset(dataset, dataset_version)
    modes: tuple[RetrievalMode, ...] = (
        "dense",
        "bm25",
        "hybrid",
        "hybrid_reranked",
    )
    reports = {
        mode: _evaluate_retrieval_mode(
            retriever,
            cases,
            mode,
            candidate_k=candidate_k,
            reranker=reranker,
        )
        for mode in modes
    }
    human_accuracy = (
        sum(
            is_clinical_advice_request(case.question)
            == (case.expected_behavior == "human_review")
            for case in cases
        )
        / len(cases)
    )
    return RetrievalEvaluationReport(
        dataset_version=resolved_version,
        case_count=len(cases),
        human_review_classifier_accuracy=human_accuracy,
        mode_reports=reports,
    )


def run_grounded_answer_evaluation(
    dataset: PolicyEvaluationDataset | Sequence[PolicyEvaluationCase],
    retrieved_by_case: Mapping[str, Sequence[PolicySearchResult]],
    answers_by_case: Mapping[str, GroundedPolicyAnswer | Mapping[str, Any] | str],
    *,
    dataset_version: str | None = None,
) -> GroundedAnswerEvaluationReport:
    """Score answer output structure, citations, quote matching, and refusal labels.

    Semantic entailment is deliberately not inferred from citation correctness
    or lexical overlap; the report marks it as ``not_assessed``.
    """
    cases, resolved_version = _resolve_dataset(dataset, dataset_version)
    case_results = tuple(
        _evaluate_grounded_answer_case(
            case,
            tuple(retrieved_by_case.get(case.case_id, ())),
            answers_by_case.get(case.case_id),
        )
        for case in cases
    )
    answered = tuple(
        result for result in case_results if result.expected_behavior == "answer"
    )
    refusal_cases = tuple(
        result for result in case_results if result.expected_behavior != "answer"
    )
    citation_values = tuple(
        result.citation_valid
        for result in answered
        if result.citation_valid is not None
    )
    expected_citation_values = tuple(
        result.expected_passage_cited
        for result in answered
        if result.expected_passage_cited is not None
    )
    quote_values = tuple(
        result.quoted_evidence_valid
        for result in answered
        if result.quoted_evidence_valid is not None
    )
    unsupported_values = tuple(
        result.unsupported_claims_detected
        for result in answered
        if result.unsupported_claims_detected is not None
    )
    refusal_values = tuple(
        result.refusal_correct
        for result in refusal_cases
        if result.refusal_correct is not None
    )
    expected_outcome_values = tuple(
        float(result.expected_outcome_correct) for result in case_results
    )
    return GroundedAnswerEvaluationReport(
        dataset_version=resolved_version,
        case_count=len(cases),
        citation_validity_rate=_mean(citation_values),
        expected_passage_citation_rate=_mean(expected_citation_values),
        quoted_evidence_rate=_mean(quote_values),
        unsupported_claim_rate=_mean(unsupported_values),
        refusal_accuracy=_mean(refusal_values),
        expected_outcome_accuracy=_mean(expected_outcome_values),
        case_results=case_results,
    )


def _evaluate_retrieval_mode(
    retriever: PolicyHybridRetriever,
    cases: Sequence[PolicyEvaluationCase],
    mode: RetrievalMode,
    *,
    candidate_k: int,
    reranker: CrossEncoderReranker | None,
) -> RetrievalModeMetrics:
    results_by_case: list[RetrievalCaseResult] = []
    answerable_scores: list[tuple[float, float, float]] = []
    outcome_values: list[bool] = []
    abstention_values: list[bool] = []
    latencies: list[float] = []
    answerable_count = 0
    insufficient_count = 0
    human_count = 0

    for case in cases:
        if case.expected_behavior == "human_review":
            human_count += 1
            results_by_case.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    expected_behavior=case.expected_behavior,
                    result_count=0,
                )
            )
            continue

        started = time.perf_counter()
        try:
            results = _search_mode(
                retriever,
                mode,
                case.question,
                case.filters,
                candidate_k=candidate_k,
                reranker=reranker,
            )
        except Exception as exc:
            raise PolicyEvaluationExecutionError(
                f"Retrieval evaluation failed for mode {mode!r}, case {case.case_id!r}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(results, (list, tuple)) or any(
            not isinstance(result, PolicySearchResult) for result in results
        ):
            raise PolicyEvaluationExecutionError(
                f"Retrieval mode {mode!r} returned invalid results for "
                f"case {case.case_id!r}"
            )
        latencies.append(latency_ms)

        relevant_ranks = _relevant_ranks(results, case.expected_passages)
        if case.expected_behavior == "answer":
            answerable_count += 1
            precision = precision_at_k(
                results, case.expected_passages, k=RETRIEVAL_K
            )
            recall = recall_at_k(results, case.expected_passages, k=RETRIEVAL_K)
            rank_score = reciprocal_rank(
                results, case.expected_passages, k=RETRIEVAL_K
            )
            answerable_scores.append((precision, recall, rank_score))
            outcome_correct = bool(relevant_ranks)
        else:
            insufficient_count += 1
            precision = recall = rank_score = None
            outcome_correct = not results
            abstention_values.append(outcome_correct)
        outcome_values.append(outcome_correct)
        results_by_case.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                expected_behavior=case.expected_behavior,
                latency_ms=latency_ms,
                result_count=len(results),
                relevant_ranks=tuple(relevant_ranks),
                precision_at_5=precision,
                recall_at_5=recall,
                reciprocal_rank=rank_score,
                outcome_correct=outcome_correct,
            )
        )

    mrr_at_5 = _mean(tuple(score[2] for score in answerable_scores))
    return RetrievalModeMetrics(
        mode=mode,
        precision_at_5=_mean(tuple(score[0] for score in answerable_scores)),
        recall_at_5=_mean(tuple(score[1] for score in answerable_scores)),
        mrr=mrr_at_5,
        mrr_at_5=mrr_at_5,
        mean_latency_ms=_mean(tuple(latencies)),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        outcome_accuracy=_mean(tuple(float(value) for value in outcome_values)),
        abstention_accuracy=_mean(tuple(float(value) for value in abstention_values)),
        answerable_case_count=answerable_count,
        insufficient_evidence_case_count=insufficient_count,
        skipped_human_review_case_count=human_count,
        queries_evaluated=len(latencies),
        case_results=tuple(results_by_case),
    )


def _search_mode(
    retriever: PolicyHybridRetriever,
    mode: RetrievalMode,
    query: str,
    filters: PolicySearchFilters,
    *,
    candidate_k: int,
    reranker: CrossEncoderReranker | None,
) -> Sequence[PolicySearchResult]:
    if mode == "dense":
        return retriever.search_dense(query, top_k=RETRIEVAL_K, filters=filters)
    if mode == "bm25":
        return retriever.search_bm25(query, top_k=RETRIEVAL_K, filters=filters)
    if mode == "hybrid":
        return retriever.search_hybrid(
            query,
            top_k=RETRIEVAL_K,
            candidate_k=candidate_k,
            filters=filters,
        )
    return retriever.search_hybrid_reranked(
        query,
        top_k=RETRIEVAL_K,
        candidate_k=candidate_k,
        filters=filters,
        reranker=reranker,
    )


def _evaluate_grounded_answer_case(
    case: PolicyEvaluationCase,
    evidence: tuple[PolicySearchResult, ...],
    answer: GroundedPolicyAnswer | Mapping[str, Any] | str | None,
) -> GroundedAnswerCaseResult:
    payload = _answer_payload(answer)
    status = payload.get("status")
    answer_text = payload.get("answer")
    citations = payload.get("citations", ())
    quotes = payload.get("supporting_quotes", ())
    citations = citations if isinstance(citations, (list, tuple)) else ()
    quotes = quotes if isinstance(quotes, (list, tuple)) else ()

    if case.expected_behavior != "answer":
        expected_answer = (
            INSUFFICIENT_EVIDENCE_ANSWER
            if case.expected_behavior == "insufficient_evidence"
            else HUMAN_REVIEW_ANSWER
        )
        refusal_correct = (
            status == case.expected_behavior
            and answer_text == expected_answer
            and not citations
            and not quotes
        )
        return GroundedAnswerCaseResult(
            case_id=case.case_id,
            expected_behavior=case.expected_behavior,
            answer_status=status if isinstance(status, str) else None,
            refusal_correct=refusal_correct,
            expected_outcome_correct=refusal_correct,
            passed=refusal_correct,
        )

    if status != "answered":
        return GroundedAnswerCaseResult(
            case_id=case.case_id,
            expected_behavior=case.expected_behavior,
            answer_status=status if isinstance(status, str) else None,
            expected_outcome_correct=False,
            passed=False,
        )

    evidence_by_id = {result.chunk_id: result for result in evidence}
    citation_valid = bool(citations) and all(
        _citation_matches(citation, evidence_by_id)
        for citation in citations
    )
    expected_passage_cited = any(
        _citation_matches(citation, evidence_by_id)
        and _citation_matches_target(citation, case.expected_passages)
        for citation in citations
    )
    quoted_evidence_valid = bool(quotes) and all(
        _quote_matches_evidence(quote, evidence_by_id)
        for quote in quotes
    )
    quote_texts = [
        quote.get("text")
        for quote in quotes
        if isinstance(quote, Mapping) and isinstance(quote.get("text"), str)
    ]
    answer_matches_quotes = (
        isinstance(answer_text, str)
        and len(quote_texts) == len(quotes)
        and answer_text == "\n".join(quote_texts)
    )
    cited_text = " ".join(
        result.text
        for result in evidence
        if any(
            isinstance(citation, Mapping)
            and citation.get("chunk_id") == result.chunk_id
            for citation in citations
        )
    )
    unsupported_numbers = (
        set(_number_tokens(answer_text)) - set(_number_tokens(cited_text))
        if isinstance(answer_text, str)
        else set()
    )
    unsupported = (
        not answer_matches_quotes
        or not quoted_evidence_valid
        or bool(unsupported_numbers)
    )
    passed = (
        status == "answered"
        and citation_valid
        and expected_passage_cited
        and quoted_evidence_valid
        and answer_matches_quotes
        and not unsupported
    )
    return GroundedAnswerCaseResult(
        case_id=case.case_id,
        expected_behavior=case.expected_behavior,
        answer_status=status if isinstance(status, str) else None,
        citation_valid=citation_valid,
        expected_passage_cited=expected_passage_cited,
        quoted_evidence_valid=quoted_evidence_valid,
        unsupported_claims_detected=unsupported,
        expected_outcome_correct=True,
        passed=passed,
    )


def _citation_matches(
    citation: Any,
    evidence_by_id: Mapping[str, PolicySearchResult],
) -> bool:
    if not isinstance(citation, Mapping):
        return False
    result = evidence_by_id.get(citation.get("chunk_id"))
    if result is None:
        return False
    expected = {
        "source_id": result.source_id,
        "section_name": result.section_name,
        "page_start": result.page_start,
        "page_end": result.page_end,
        "specialty": result.specialty,
        "plan_ids": list(result.plan_ids),
        "version": result.version,
    }
    return all(citation.get(key) == value for key, value in expected.items())


def _quote_matches_evidence(
    quote: Any,
    evidence_by_id: Mapping[str, PolicySearchResult],
) -> bool:
    if not isinstance(quote, Mapping):
        return False
    result = evidence_by_id.get(quote.get("chunk_id"))
    text = quote.get("text")
    return result is not None and isinstance(text, str) and text in result.text


def _citation_matches_target(
    citation: Any,
    expected: Sequence[PolicyEvaluationTarget],
) -> bool:
    if not isinstance(citation, Mapping):
        return False
    page_start = citation.get("page_start")
    page_end = citation.get("page_end") or page_start
    return any(
        citation.get("source_id") == target.source_id
        and citation.get("section_name") == target.section_name
        and isinstance(page_start, int)
        and isinstance(page_end, int)
        and page_start <= target.page_number <= page_end
        for target in expected
    )


def _answer_payload(
    answer: GroundedPolicyAnswer | Mapping[str, Any] | str | None,
) -> Mapping[str, Any]:
    if isinstance(answer, GroundedPolicyAnswer):
        return answer.model_dump(mode="json")
    if isinstance(answer, Mapping):
        return answer
    if isinstance(answer, str):
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _resolve_dataset(
    dataset: PolicyEvaluationDataset | Sequence[PolicyEvaluationCase],
    dataset_version: str | None,
) -> tuple[tuple[PolicyEvaluationCase, ...], str]:
    if isinstance(dataset, PolicyEvaluationDataset):
        cases = dataset.cases
        resolved_version = dataset_version or dataset.dataset_version
    else:
        cases = tuple(dataset)
        resolved_version = dataset_version or "caller-supplied"
    if not cases or any(
        not isinstance(case, PolicyEvaluationCase) for case in cases
    ):
        raise PolicyEvaluationDataError(
            "Evaluation requires at least one validated synthetic policy case"
        )
    if not isinstance(resolved_version, str) or not resolved_version.strip():
        raise PolicyEvaluationDataError("dataset_version must not be empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise PolicyEvaluationDataError("Evaluation case IDs must be unique")
    return tuple(cases), resolved_version.strip()


def _matched_target_keys(
    results: Sequence[PolicySearchResult],
    expected: Sequence[PolicyEvaluationTarget],
    *,
    k: int,
) -> set[tuple[str, str, int]]:
    expected_keys = _target_key_set(expected)
    matched: set[tuple[str, str, int]] = set()
    for result in results[:k]:
        target_key = _matching_target_key(result, expected_keys)
        if target_key is not None:
            matched.add(target_key)
    return matched


def _matching_target_key(
    result: PolicySearchResult,
    expected_keys: set[tuple[str, str, int]],
) -> tuple[str, str, int] | None:
    for source_id, section_name, page_number in expected_keys:
        page_end = result.page_end or result.page_start
        if (
            result.source_id == source_id
            and result.section_name == section_name
            and result.page_start is not None
            and page_end is not None
            and result.page_start <= page_number <= page_end
        ):
            return source_id, section_name, page_number
    return None


def _relevant_ranks(
    results: Sequence[PolicySearchResult],
    expected: Sequence[PolicyEvaluationTarget],
) -> list[int]:
    expected_keys = _target_key_set(expected)
    seen: set[tuple[str, str, int]] = set()
    ranks = []
    for rank, result in enumerate(results[:RETRIEVAL_K], start=1):
        target_key = _matching_target_key(result, expected_keys)
        if target_key is not None and target_key not in seen:
            seen.add(target_key)
            ranks.append(rank)
    return ranks


def _target_key_set(
    expected: Sequence[PolicyEvaluationTarget],
) -> set[tuple[str, str, int]]:
    return {
        (target.source_id, target.section_name, target.page_number)
        for target in expected
    }


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _mean(values: Sequence[float | bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _number_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text))
