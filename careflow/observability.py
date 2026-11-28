"""Sanitized, optional Langfuse tracing for synthetic workflow stages."""

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any, Protocol

SAFE_STAGES = frozenset(
    {
        "clinical_safety",
        "intake",
        "eligibility_lookup",
        "policy_retrieval",
        "referral_handling",
        "human_approval",
        "human_approval_resume",
        "mcp_tool",
    }
)
SAFE_OUTCOMES = frozenset(
    {
        "started",
        "success",
        "failed",
        "unavailable",
        "circuit_open",
        "retrying",
        "pending_review",
        "approved",
        "rejected",
        "denied",
        "skipped",
        "human_review",
        "insufficient_evidence",
        "timeout",
    }
)
SAFE_METADATA_KEYS = frozenset(
    {
        "attempt",
        "attempts",
        "duration_ms",
        "error_type",
        "reason_code",
        "decision",
        "tool",
        "retry_count",
        "circuit_state",
    }
)
SAFE_STRING_VALUES = {
    "error_type": frozenset(
        {
            "TimeoutError",
            "ConnectionError",
            "TransientServiceError",
            "RuntimeError",
            "parse_failure",
            "service_error",
            "retrieval_error",
        }
    ),
    "reason_code": frozenset(
        {
            "clinical_advice_request",
            "clinical_safety_check_failed",
            "intake_requires_human_review",
            "identity_mismatch",
            "request_type_or_specialty_unclear",
            "referral_input_missing",
            "eligibility_service_unavailable",
            "eligibility_lookup_failed",
            "human_approval_required_before_referral_routing",
            "eligibility_not_confirmed",
            "circuit_open",
            "capacity_timeout",
            "timeout",
            "transient_failure",
            "retry_exhausted",
            "eligible",
            "ineligible",
            "not_found",
            "routed",
            "unavailable",
        }
    ),
    "decision": frozenset({"approve", "reject"}),
    "circuit_state": frozenset({"closed", "open", "half_open"}),
    "tool": frozenset(
        {
            "lookup_appointments",
            "check_insurance_eligibility",
            "calculate_copay",
            "route_referral",
            "lookup_referral",
            "flag_for_human",
        }
    ),
}
_REQUEST_ID = re.compile(r"^SYN-REQ-[A-Z0-9-]{1,90}$")
_SESSION_ID = re.compile(r"^SYN-SESSION-[a-f0-9]{32}$")


class WorkflowObserver(Protocol):
    def record(
        self,
        *,
        stage: str,
        outcome: str,
        request_id: str,
        session_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool: ...


class LangfuseWorkflowTracer:
    """Write workflow spans to a supplied Langfuse v2-compatible client.

    Only synthetic request/session correlation IDs and allowlisted scalar
    metadata are recorded. Request text, outputs, member IDs, credentials,
    policy passages, reviewer identifiers, and exception messages are omitted.
    All client errors are nonfatal.
    """

    def __init__(self, client: Any | None) -> None:
        self._client = client

    def record(
        self,
        *,
        stage: str,
        outcome: str,
        request_id: str,
        session_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        if self._client is None:
            return False
        if (
            stage not in SAFE_STAGES
            or outcome not in SAFE_OUTCOMES
            or not isinstance(request_id, str)
            or not _REQUEST_ID.fullmatch(request_id)
            or not isinstance(session_id, str)
            or not _SESSION_ID.fullmatch(session_id)
        ):
            return False
        safe_metadata = _sanitize_metadata(metadata or {})
        trace_id = hashlib.sha256(
            f"{session_id}:{request_id}".encode("utf-8")
        ).hexdigest()[:32]
        try:
            trace = self._client.trace(
                id=trace_id,
                name="careflow-synthetic-workflow",
                metadata={
                    "synthetic_data": True,
                    "request_id": request_id,
                    "session_id": session_id,
                },
                tags=["synthetic", "careflow-m7"],
            )
            span = trace.span(
                name=stage,
                input={
                    "synthetic_request_id": request_id,
                    "synthetic_session_id": session_id,
                },
                output={"outcome": outcome},
                metadata=safe_metadata,
            )
            finish = getattr(span, "end", None)
            if callable(finish):
                finish()
            return True
        except Exception:
            return False


def _sanitize_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key not in SAFE_METADATA_KEYS:
            continue
        if isinstance(value, bool):
            sanitized[key] = value
        elif isinstance(value, int):
            sanitized[key] = value
        elif isinstance(value, float) and math.isfinite(value) and value >= 0:
            sanitized[key] = value
        elif (
            isinstance(value, str)
            and value in SAFE_STRING_VALUES.get(key, frozenset())
        ):
            sanitized[key] = value
    return sanitized
