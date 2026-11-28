"""Optional, aggregate-only Langfuse score logging."""

import hashlib
import math
import os
import re
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .policy_evaluation import (
    GroundedAnswerEvaluationReport,
    RetrievalEvaluationReport,
)


class LangfuseLoggingError(RuntimeError):
    """A Langfuse evaluation score could not be recorded."""


class LangfuseUnavailableError(LangfuseLoggingError):
    """No compatible Langfuse client was supplied."""


class LangfuseLoggingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempted: bool = True
    success: bool
    trace_id: str
    scores_written: int | None = Field(default=None, ge=0)
    error_code: str | None = None


class LangfuseEvaluationLogger:
    """Create an aggregate-only evaluation trace and attach metric scores.

    The adapter imports no Langfuse package and reads no credentials. Callers
    configure and supply their client; only allowlisted aggregate metrics and
    hashed dataset metadata are sent, never questions, answers, or passages.
    """

    def __init__(self, client: Any | None) -> None:
        self._client = client

    def workflow_tracer(self):
        """Reuse the configured M4 Langfuse client for sanitized M7 spans."""
        from .observability import LangfuseWorkflowTracer

        return LangfuseWorkflowTracer(self._client)

    def log_retrieval_report(
        self,
        report: RetrievalEvaluationReport,
        *,
        trace_id: str | None = None,
    ) -> int:
        scores: list[tuple[str, float, dict[str, Any]]] = []
        for metrics in report.mode_reports.values():
            mode = metrics.mode
            metadata = {
                "dataset_version_hash": _version_digest(report.dataset_version),
                "case_count": report.case_count,
                "mode": mode,
            }
            for metric_name in (
                "precision_at_5",
                "recall_at_5",
                "mrr",
                "mrr_at_5",
                "mean_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "outcome_accuracy",
                "abstention_accuracy",
            ):
                value = getattr(metrics, metric_name)
                if value is not None:
                    scores.append(
                        (
                            f"policy_retrieval.{mode}.{metric_name}",
                            float(value),
                            metadata,
                        )
                    )
        return self._write_scores(
            scores,
            trace_id=trace_id or uuid4().hex,
            trace_name="careflow-m4-policy-retrieval-evaluation",
            trace_metadata={
                "dataset_version_hash": _version_digest(report.dataset_version),
                "case_count": report.case_count,
                "synthetic_data": True,
            },
        )

    def log_grounded_answer_report(
        self,
        report: GroundedAnswerEvaluationReport,
        *,
        trace_id: str | None = None,
    ) -> int:
        metadata = {
            "dataset_version_hash": _version_digest(report.dataset_version),
            "case_count": report.case_count,
        }
        scores = [
            (f"policy_answers.{metric_name}", float(value), metadata)
            for metric_name in (
                "citation_validity_rate",
                "expected_passage_citation_rate",
                "quoted_evidence_rate",
                "unsupported_claim_rate",
                "refusal_accuracy",
                "expected_outcome_accuracy",
            )
            if (value := getattr(report, metric_name)) is not None
        ]
        return self._write_scores(
            scores,
            trace_id=trace_id or uuid4().hex,
            trace_name="careflow-m4-policy-answer-evaluation",
            trace_metadata={
                "dataset_version_hash": _version_digest(report.dataset_version),
                "case_count": report.case_count,
                "synthetic_data": True,
            },
        )

    def _write_scores(
        self,
        scores: list[tuple[str, float, dict[str, Any]]],
        *,
        trace_id: str,
        trace_name: str,
        trace_metadata: dict[str, Any],
    ) -> int:
        if not isinstance(trace_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", trace_id
        ):
            raise ValueError("trace_id must be a 32-character lowercase hex ID")
        for _, value, _ in scores:
            if not math.isfinite(value):
                raise LangfuseLoggingError(
                    "Evaluation score is not a finite numeric value"
                )

        trace = self._create_trace(trace_id, trace_name, trace_metadata)
        trace_score = getattr(trace, "score", None)
        create_score = getattr(self._client, "create_score", None)
        if not callable(trace_score) and not callable(create_score):
            raise LangfuseUnavailableError(
                "Langfuse trace client does not support score creation"
            )

        written = 0
        for name, value, metadata in scores:
            try:
                if callable(trace_score):
                    trace_score(
                        name=name,
                        value=value,
                        data_type="NUMERIC",
                        comment="Synthetic CareFlow evaluation aggregate",
                    )
                else:
                    create_score(
                        trace_id=trace_id,
                        name=name,
                        value=value,
                        data_type="NUMERIC",
                        comment="Synthetic CareFlow evaluation aggregate",
                        metadata=dict(metadata),
                    )
            except Exception as exc:
                raise LangfuseLoggingError(
                    f"Langfuse score write failed for {name} "
                    f"(exception type: {type(exc).__name__})"
                ) from None
            written += 1
        flush = getattr(self._client, "flush", None)
        if written and callable(flush):
            try:
                flush()
            except Exception as exc:
                raise LangfuseLoggingError(
                    "Langfuse score flush failed "
                    f"(exception type: {type(exc).__name__})"
                ) from None
        return written

    def _create_trace(
        self,
        trace_id: str,
        trace_name: str,
        metadata: dict[str, Any],
    ) -> Any:
        create_trace = getattr(self._client, "trace", None)
        if not callable(create_trace):
            raise LangfuseUnavailableError(
                "A configured Langfuse client with trace() is required"
            )
        try:
            return create_trace(
                id=trace_id,
                name=trace_name,
                metadata=dict(metadata),
                tags=["synthetic", "policy-evaluation"],
            )
        except Exception as exc:
            raise LangfuseLoggingError(
                "Langfuse trace creation failed "
                f"(exception type: {type(exc).__name__})"
            ) from None

    def log_retrieval_report_nonfatal(
        self,
        report: RetrievalEvaluationReport,
        *,
        trace_id: str | None = None,
    ) -> LangfuseLoggingResult:
        return self._log_nonfatal(
            lambda resolved_id: self.log_retrieval_report(
                report,
                trace_id=resolved_id,
            ),
            trace_id,
        )

    def log_grounded_answer_report_nonfatal(
        self,
        report: GroundedAnswerEvaluationReport,
        *,
        trace_id: str | None = None,
    ) -> LangfuseLoggingResult:
        return self._log_nonfatal(
            lambda resolved_id: self.log_grounded_answer_report(
                report,
                trace_id=resolved_id,
            ),
            trace_id,
        )

    def _log_nonfatal(
        self,
        log_call: Callable[[str], int],
        trace_id: str | None,
    ) -> LangfuseLoggingResult:
        if trace_id is not None and not re.fullmatch(r"[0-9a-f]{32}", trace_id):
            return LangfuseLoggingResult(
                attempted=False,
                success=False,
                trace_id=uuid4().hex,
                error_code="ValueError",
            )
        resolved_id = trace_id or uuid4().hex
        try:
            written = log_call(resolved_id)
        except Exception as exc:
            return LangfuseLoggingResult(
                success=False,
                trace_id=resolved_id,
                error_code=type(exc).__name__,
            )
        return LangfuseLoggingResult(
            success=True,
            trace_id=resolved_id,
            scores_written=written,
        )


def _version_digest(dataset_version: str) -> str:
    return hashlib.sha256(dataset_version.encode("utf-8")).hexdigest()[:16]


def create_langfuse_evaluation_logger_from_environment(
    environment: Mapping[str, str] | None = None,
) -> LangfuseEvaluationLogger:
    """Create a v2 Langfuse client from process environment, never a dotenv file."""
    values = os.environ if environment is None else environment
    public_key = values.get("LANGFUSE_PUBLIC_KEY")
    secret_key = values.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return LangfuseEvaluationLogger(None)
    try:
        from langfuse import Langfuse

        options = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        host = values.get("LANGFUSE_HOST")
        if host:
            options["host"] = host
        client = Langfuse(
            **options,
        )
    except Exception:
        return LangfuseEvaluationLogger(None)
    return LangfuseEvaluationLogger(client)
