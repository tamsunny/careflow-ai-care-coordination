"""LangGraph-based, persistent synthetic intake and referral workflow."""

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .agent import SchedulingInsuranceAgent
from .intake import is_clinical_advice_request, parse_patient_message
from .memory import SQLiteConversationMemory
from .models import PatientRequest
from .observability import WorkflowObserver
from .resilience import ServiceUnavailableError
from .tool_models import (
    InsuranceEligibilityInput,
    InsuranceEligibilityOutput,
    ReferralRoutingInput,
    ReferralRoutingOutput,
)

ReviewerAuthorizer = Callable[[str, str, str], bool]
IntakeParser = Callable[[str], PatientRequest]
SafetyChecker = Callable[[str], bool]


class WorkflowToolExecutor(Protocol):
    def check_insurance_eligibility(
        self,
        request: InsuranceEligibilityInput,
    ) -> InsuranceEligibilityOutput: ...

    def route_referral(
        self,
        request: ReferralRoutingInput,
    ) -> ReferralRoutingOutput: ...


ReferralStatus = Literal[
    "not_started",
    "awaiting_human_approval",
    "routed",
    "blocked_eligibility",
    "not_found",
    "not_routed",
]
ApprovalState = Literal["not_required", "pending", "approved", "rejected"]
WorkflowStatus = Literal[
    "running",
    "awaiting_human",
    "completed",
    "rejected",
    "needs_followup",
    "human_reviewed_no_automation",
    "failed",
]


class IntakeResultState(TypedDict):
    request_type: str
    specialty: str | None
    symptoms: list[str]
    urgency_flag: bool
    insurance_id: str | None
    needs_human_review: bool
    reason_for_review: str | None


class EligibilityResultState(TypedDict):
    status: Literal["eligible", "ineligible", "not_found"]
    insurance_id: str
    eligible: bool | None
    plan_id: str | None
    coverage_end: str | None


class ReferralResultState(TypedDict):
    status: Literal["routed", "not_found"]
    request_id: str
    referral_id: str | None
    specialty: str
    destination: str | None


class WorkflowState(TypedDict):
    member_id: str
    session_id: str
    request_id: str
    idempotency_key: str
    request_text: str
    clinical_request: bool
    intake_result: IntakeResultState | None
    eligibility_result: EligibilityResultState | None
    referral_result: ReferralResultState | None
    referral_status: ReferralStatus
    errors: list[str]
    human_approval_state: ApprovalState
    reviewer_id: str | None
    review_reason: str | None
    needs_human_review: bool
    workflow_status: WorkflowStatus


class WorkflowOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    trusted_member_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(
        min_length=16,
        max_length=64,
        pattern=r"^SYN-SESSION-[a-f0-9]+$",
    )


class WorkflowIdentity(WorkflowOwner):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    request_id: str = Field(
        min_length=12,
        max_length=100,
        pattern=r"^SYN-REQ-[A-Z0-9-]+$",
    )
    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    request_text: str = Field(min_length=1, max_length=20_000)

    @field_validator("request_text")
    @classmethod
    def require_nonblank_request(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Request text must not be blank")
        return value


class WorkflowResumeIdentity(WorkflowOwner):
    request_id: str = Field(
        min_length=12,
        max_length=100,
        pattern=r"^SYN-REQ-[A-Z0-9-]+$",
    )
    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


class ReviewerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reviewer_id: str = Field(min_length=1, max_length=120)
    decision: Literal["approve", "reject"]


class WorkflowSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: str
    member_id: str
    session_id: str
    request_id: str
    idempotency_key: str
    intake_result: PatientRequest | None = None
    eligibility_result: InsuranceEligibilityOutput | None = None
    referral_result: ReferralRoutingOutput | None = None
    referral_status: ReferralStatus
    errors: tuple[str, ...] = ()
    human_approval_state: ApprovalState
    reviewer_id: str | None = None
    review_reason: str | None = None
    workflow_status: WorkflowStatus
    interrupt_payload: dict[str, Any] | None = None

    @property
    def interrupted(self) -> bool:
        return self.workflow_status == "awaiting_human"


class WorkflowError(RuntimeError):
    """Base error for safe synthetic workflow execution."""


class WorkflowIdentityError(WorkflowError):
    """Workflow identity is invalid or does not own the memory session."""


class WorkflowIdempotencyConflictError(WorkflowError):
    """A request or idempotency key was reused for different workflow input."""


class WorkflowInProgressError(WorkflowError):
    """The logical workflow is already executing in another call/process."""


class WorkflowResumeNotAvailableError(WorkflowError):
    """The workflow is not waiting at an authorized human checkpoint."""


class WorkflowNotFoundError(WorkflowError):
    """No workflow belongs to the supplied trusted member and session."""


class WorkflowUnauthorizedResumeError(WorkflowError):
    """The host did not authorize the supplied human reviewer."""


class WorkflowExecutionError(WorkflowError):
    """The workflow failed and was closed to automatic replay."""


class LangGraphWorkflow:
    """Execute an authorized workflow with a durable SQLite checkpoint.

    The host must create the memory session and supply its trusted member ID.
    A host-side reviewer authorization callback is mandatory; approval values
    are accepted only from ``resume_workflow``, never from an LLM response.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        memory: SQLiteConversationMemory,
        reviewer_authorizer: ReviewerAuthorizer,
        *,
        model: str = "mock/workflow-intake",
        completion_fn: Callable[..., Any] | None = None,
        intake_parser: IntakeParser | None = None,
        tool_executor: WorkflowToolExecutor | None = None,
        safety_checker: SafetyChecker | None = None,
        observer: WorkflowObserver | None = None,
    ) -> None:
        if not callable(reviewer_authorizer):
            raise ValueError("A host-side reviewer authorizer is required")
        if not model.strip():
            raise ValueError("An intake model identifier must not be empty")
        if safety_checker is not None and not callable(safety_checker):
            raise ValueError("safety_checker must be callable")
        if observer is not None and not callable(getattr(observer, "record", None)):
            raise ValueError("observer must provide a record method")

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = memory
        self._reviewer_authorizer = reviewer_authorizer
        self._tool_executor = tool_executor or SchedulingInsuranceAgent()
        self._safety_checker = safety_checker or is_clinical_advice_request
        self._observer = observer
        self._intake_parser = intake_parser or (
            lambda text: parse_patient_message(
                text,
                model,
                completion_fn=completion_fn,
            )
        )
        self._initialize_registry()
        self._checkpoint_context = SqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        )
        self._checkpointer = self._checkpoint_context.__enter__()
        try:
            self._checkpointer.setup()
            self._graph = self._build_graph()
        except Exception:
            self._checkpoint_context.__exit__(None, None, None)
            raise
        self._closed = False

    def __enter__(self) -> "LangGraphWorkflow":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._checkpoint_context.__exit__(None, None, None)
            self._closed = True

    def start_workflow(
        self,
        *,
        trusted_member_id: str,
        session_id: str,
        request_id: str,
        idempotency_key: str,
        request_text: str,
    ) -> WorkflowSnapshot:
        identity = WorkflowIdentity.model_validate(
            {
                "trusted_member_id": trusted_member_id,
                "session_id": session_id,
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "request_text": request_text,
            }
        )
        self._authorize_member_session(identity)
        thread_id = _workflow_thread_id(identity)
        request_hash = _request_hash(identity.request_text)
        existing, created = self._reserve_run(identity, thread_id, request_hash)
        if not created:
            status = existing["status"]
            if status in {"running", "resuming"}:
                raise WorkflowInProgressError(
                    "This workflow is already executing; do not start a duplicate"
                )
            if status == "failed":
                raise WorkflowExecutionError(
                    "This workflow failed and will not be automatically replayed"
                )
            return self._load_snapshot(thread_id)

        initial_state: WorkflowState = {
            "member_id": identity.trusted_member_id,
            "session_id": identity.session_id,
            "request_id": identity.request_id,
            "idempotency_key": identity.idempotency_key,
            "request_text": identity.request_text,
            "clinical_request": False,
            "intake_result": None,
            "eligibility_result": None,
            "referral_result": None,
            "referral_status": "not_started",
            "errors": [],
            "human_approval_state": "not_required",
            "reviewer_id": None,
            "review_reason": None,
            "needs_human_review": False,
            "workflow_status": "running",
        }
        try:
            self._graph.invoke(initial_state, self._config(thread_id))
            snapshot = self._load_snapshot(thread_id)
            self._save_snapshot(snapshot)
            return snapshot
        except Exception as exc:
            self._update_run(thread_id, "failed", None)
            raise WorkflowExecutionError(
                "Workflow execution failed "
                f"({type(exc).__name__}); automatic replay is blocked"
            ) from None

    def resume_workflow(
        self,
        *,
        trusted_member_id: str,
        session_id: str,
        request_id: str,
        idempotency_key: str,
        reviewer_id: str,
        decision: Literal["approve", "reject"],
    ) -> WorkflowSnapshot:
        identity = WorkflowResumeIdentity.model_validate(
            {
                "trusted_member_id": trusted_member_id,
                "session_id": session_id,
                "request_id": request_id,
                "idempotency_key": idempotency_key,
            }
        )
        self._authorize_member_session(identity)
        reviewer_decision = ReviewerDecision.model_validate(
            {"reviewer_id": reviewer_id, "decision": decision}
        )
        if not self._is_authorized_reviewer(
            reviewer_decision.reviewer_id,
            identity.model_dump(),
        ):
            self._record_observation(
                identity.request_id,
                identity.session_id,
                "human_approval_resume",
                "denied",
            )
            raise WorkflowUnauthorizedResumeError(
                "Host authorization denied this workflow resume"
            )
        self._record_observation(
            identity.request_id,
            identity.session_id,
            "human_approval_resume",
            "started",
            decision=reviewer_decision.decision,
        )

        thread_id = _workflow_thread_id(identity)
        record = self._claim_resume(identity, thread_id)
        if record["status"] in {
            "completed",
            "rejected",
            "needs_followup",
            "human_reviewed_no_automation",
        }:
            snapshot = self._load_snapshot(thread_id)
            self._record_observation(
                identity.request_id,
                identity.session_id,
                "human_approval_resume",
                "success",
                decision=reviewer_decision.decision,
            )
            return snapshot

        try:
            self._graph.invoke(
                Command(resume=reviewer_decision.model_dump()),
                self._config(thread_id),
            )
            snapshot = self._load_snapshot(thread_id)
            self._save_snapshot(snapshot)
            self._record_observation(
                identity.request_id,
                identity.session_id,
                "human_approval_resume",
                "success",
                decision=reviewer_decision.decision,
            )
            return snapshot
        except Exception as exc:
            self._update_run(thread_id, "failed", None)
            self._record_observation(
                identity.request_id,
                identity.session_id,
                "human_approval_resume",
                "failed",
                error_type=type(exc).__name__,
            )
            raise WorkflowExecutionError(
                f"Workflow resume failed ({type(exc).__name__}); duplicate resume is blocked"
            ) from None

    def get_workflow_snapshot(
        self,
        *,
        trusted_member_id: str,
        session_id: str,
        workflow_id: str,
    ) -> WorkflowSnapshot:
        owner = WorkflowOwner(
            trusted_member_id=trusted_member_id,
            session_id=session_id,
        )
        self._authorize_member_session(owner)
        with closing(
            sqlite3.connect(str(self.checkpoint_path), timeout=10)
        ) as connection:
            row = connection.execute(
                "SELECT thread_id FROM workflow_runs "
                "WHERE thread_id = ? AND member_id = ? AND session_id = ?",
                (workflow_id, owner.trusted_member_id, owner.session_id),
            ).fetchone()
        if row is None:
            raise WorkflowNotFoundError(
                "Workflow was not found for the authorized session"
            )
        return self._load_snapshot(workflow_id)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("safety_check", self._safety_check)
        graph.add_node("parse_intake", self._parse_intake)
        graph.add_node("check_eligibility", self._check_eligibility)
        graph.add_node("human_approval", self._human_approval)
        graph.add_node("route_referral", self._route_referral)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "safety_check")
        graph.add_conditional_edges(
            "safety_check",
            lambda state: (
                "human_approval"
                if state["clinical_request"]
                else "parse_intake"
            ),
            {
                "human_approval": "human_approval",
                "parse_intake": "parse_intake",
            },
        )
        graph.add_conditional_edges(
            "parse_intake",
            self._after_intake,
            {
                "human_approval": "human_approval",
                "check_eligibility": "check_eligibility",
            },
        )
        graph.add_edge("check_eligibility", "human_approval")
        graph.add_conditional_edges(
            "human_approval",
            self._after_approval,
            {
                "route_referral": "route_referral",
                "finalize": "finalize",
            },
        )
        graph.add_edge("route_referral", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _safety_check(self, state: WorkflowState) -> dict[str, Any]:
        try:
            is_clinical_request = (
                self._safety_checker(state["request_text"]) is True
            )
        except Exception:
            self._observe(state, "clinical_safety", "unavailable")
            return {
                "clinical_request": True,
                "needs_human_review": True,
                "review_reason": "clinical_safety_check_failed",
                "errors": _append_error(
                    state["errors"],
                    "clinical_safety_check_failed",
                ),
                "referral_status": "not_routed",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }
        if is_clinical_request:
            self._observe(
                state,
                "clinical_safety",
                "human_review",
                reason_code="clinical_advice_request",
            )
            return {
                "clinical_request": True,
                "needs_human_review": True,
                "review_reason": "clinical_advice_request",
                "referral_status": "not_routed",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }
        self._observe(state, "clinical_safety", "success")
        return {"clinical_request": False}

    def _parse_intake(self, state: WorkflowState) -> dict[str, Any]:
        self._observe(state, "intake", "started")
        try:
            parsed = self._intake_parser(state["request_text"])
            intake = (
                parsed
                if isinstance(parsed, PatientRequest)
                else PatientRequest.model_validate(parsed)
            )
        except Exception:
            self._observe(state, "intake", "failed", error_type="parse_failure")
            return {
                "intake_result": None,
                "errors": _append_error(state["errors"], "intake_parse_failed"),
                "needs_human_review": True,
                "review_reason": "intake_parse_failed",
                "referral_status": "not_routed",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }

        if intake.needs_human_review:
            self._observe(
                state,
                "intake",
                "pending_review",
                reason_code="intake_requires_human_review",
            )
            return {
                "intake_result": intake.model_dump(mode="json"),
                "needs_human_review": True,
                "review_reason": "intake_requires_human_review",
                "referral_status": "not_routed",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }
        if intake.insurance_id and intake.insurance_id != state["member_id"]:
            self._observe(
                state,
                "intake",
                "pending_review",
                reason_code="identity_mismatch",
            )
            return {
                "intake_result": intake.model_dump(mode="json"),
                "needs_human_review": True,
                "review_reason": "intake_identity_mismatch",
                "errors": _append_error(
                    state["errors"],
                    "intake_identity_mismatch",
                ),
                "referral_status": "not_routed",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }
        if (
            "referral" not in intake.request_type.casefold()
            or intake.specialty is None
        ):
            self._observe(
                state,
                "intake",
                "pending_review",
                reason_code="request_type_or_specialty_unclear",
            )
            return {
                "intake_result": intake.model_dump(mode="json"),
                "needs_human_review": True,
                "review_reason": "referral_type_or_specialty_unclear",
                "referral_status": "not_routed",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }
        self._observe(state, "intake", "success")
        return {"intake_result": intake.model_dump(mode="json")}

    @staticmethod
    def _after_intake(state: WorkflowState) -> str:
        if state["needs_human_review"]:
            return "human_approval"
        return "check_eligibility"

    def _check_eligibility(self, state: WorkflowState) -> dict[str, Any]:
        self._observe(state, "eligibility_lookup", "started")
        try:
            result = self._tool_executor.check_insurance_eligibility(
                InsuranceEligibilityInput(insurance_id=state["member_id"])
            )
        except ServiceUnavailableError as exc:
            self._observe(
                state,
                "eligibility_lookup",
                "unavailable",
                reason_code=exc.reason,
            )
            return {
                "errors": _append_error(
                    state["errors"],
                    "eligibility_service_unavailable",
                ),
                "needs_human_review": True,
                "review_reason": "eligibility_service_unavailable",
                "referral_status": "blocked_eligibility",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }
        except Exception:
            self._observe(
                state,
                "eligibility_lookup",
                "failed",
                error_type="service_error",
            )
            return {
                "errors": _append_error(state["errors"], "eligibility_lookup_failed"),
                "needs_human_review": True,
                "review_reason": "eligibility_lookup_failed",
                "referral_status": "blocked_eligibility",
                "human_approval_state": "pending",
                "workflow_status": "awaiting_human",
            }

        if result.status == "eligible" and result.eligible is True:
            reason = "human_approval_required_before_referral_routing"
            referral_status: ReferralStatus = "awaiting_human_approval"
        else:
            reason = "eligibility_not_confirmed"
            referral_status = "blocked_eligibility"
        self._observe(
            state,
            "eligibility_lookup",
            "success",
            reason_code=result.status,
        )
        return {
            "eligibility_result": result.model_dump(mode="json"),
            "needs_human_review": True,
            "review_reason": reason,
            "referral_status": referral_status,
            "human_approval_state": "pending",
            "workflow_status": "awaiting_human",
        }

    def _human_approval(self, state: WorkflowState) -> dict[str, Any]:
        intake = _coerce_model(state["intake_result"], PatientRequest)
        eligibility = _coerce_model(
            state["eligibility_result"],
            InsuranceEligibilityOutput,
        )
        self._observe(
            state,
            "human_approval",
            "pending_review",
            reason_code=state["review_reason"] or "human_review_required",
        )
        decision_value = interrupt(
            {
                "kind": "host_human_review",
                "request_id": state["request_id"],
                "member_id": state["member_id"],
                "session_id": state["session_id"],
                "reason": state["review_reason"] or "human_review_required",
                "request_text": state["request_text"],
                "intake": (
                    {
                        "request_type": intake.request_type,
                        "specialty": intake.specialty,
                        "needs_human_review": intake.needs_human_review,
                    }
                    if intake is not None
                    else None
                ),
                "eligibility": (
                    {
                        "status": eligibility.status,
                        "eligible": eligibility.eligible,
                        "plan_id": eligibility.plan_id,
                    }
                    if eligibility is not None
                    else None
                ),
                "allowed_decisions": ["approve", "reject"],
            }
        )
        try:
            decision = ReviewerDecision.model_validate(decision_value)
        except ValidationError:
            raise WorkflowUnauthorizedResumeError(
                "A valid host-side reviewer decision is required"
            ) from None
        if not self._is_authorized_reviewer(
            decision.reviewer_id,
            {
                "trusted_member_id": state["member_id"],
                "session_id": state["session_id"],
            },
        ):
            raise WorkflowUnauthorizedResumeError(
                "Host authorization denied this workflow resume"
            )

        if decision.decision == "reject":
            self._observe(
                state,
                "human_approval",
                "rejected",
                decision="reject",
            )
            return {
                "human_approval_state": "rejected",
                "reviewer_id": decision.reviewer_id,
                "referral_status": "not_routed",
                "workflow_status": "rejected",
            }
        self._observe(
            state,
            "human_approval",
            "approved",
            decision="approve",
        )
        return {
            "human_approval_state": "approved",
            "reviewer_id": decision.reviewer_id,
            "workflow_status": "running",
        }

    @staticmethod
    def _after_approval(state: WorkflowState) -> str:
        intake = _coerce_model(state["intake_result"], PatientRequest)
        eligibility = _coerce_model(
            state["eligibility_result"],
            InsuranceEligibilityOutput,
        )
        if (
            state["human_approval_state"] == "approved"
            and not state["clinical_request"]
            and state["needs_human_review"]
            and intake is not None
            and "referral" in intake.request_type.casefold()
            and intake.specialty is not None
            and eligibility is not None
            and eligibility.status == "eligible"
            and eligibility.eligible is True
        ):
            return "route_referral"
        return "finalize"

    def _route_referral(self, state: WorkflowState) -> dict[str, Any]:
        intake = _coerce_model(state["intake_result"], PatientRequest)
        if intake is None or intake.specialty is None:
            self._observe(
                state,
                "referral_handling",
                "unavailable",
                reason_code="referral_input_missing",
            )
            return {
                "errors": _append_error(state["errors"], "referral_input_missing"),
                "referral_status": "not_routed",
                "workflow_status": "needs_followup",
            }
        try:
            self._observe(state, "referral_handling", "started")
            result = self._tool_executor.route_referral(
                ReferralRoutingInput(
                    request_id=state["request_id"],
                    specialty=intake.specialty,
                )
            )
        except ServiceUnavailableError as exc:
            self._observe(
                state,
                "referral_handling",
                "unavailable",
                reason_code=exc.reason,
            )
            return {
                "errors": _append_error(
                    state["errors"],
                    "referral_service_unavailable",
                ),
                "referral_status": "not_routed",
                "workflow_status": "needs_followup",
            }
        except Exception:
            self._observe(
                state,
                "referral_handling",
                "failed",
                error_type="service_error",
            )
            return {
                "errors": _append_error(state["errors"], "referral_routing_failed"),
                "referral_status": "not_routed",
                "workflow_status": "needs_followup",
            }
        self._observe(
            state,
            "referral_handling",
            "success" if result.status == "routed" else "unavailable",
            reason_code=result.status,
        )
        return {
            "referral_result": result.model_dump(mode="json"),
            "referral_status": (
                "routed" if result.status == "routed" else "not_found"
            ),
            "workflow_status": (
                "completed" if result.status == "routed" else "needs_followup"
            ),
        }

    @staticmethod
    def _finalize(state: WorkflowState) -> dict[str, Any]:
        if state["clinical_request"]:
            return {
                "referral_status": "not_routed",
                "workflow_status": "human_reviewed_no_automation",
            }
        if state["workflow_status"] == "running":
            eligibility = _coerce_model(
                state["eligibility_result"],
                InsuranceEligibilityOutput,
            )
            if (
                state["human_approval_state"] == "approved"
                and eligibility is not None
                and (
                    eligibility.status != "eligible"
                    or eligibility.eligible is not True
                )
            ):
                return {
                    "referral_status": "blocked_eligibility",
                    "workflow_status": "needs_followup",
                }
            if state["human_approval_state"] == "approved":
                return {
                    "referral_status": "not_routed",
                    "workflow_status": "needs_followup",
                }
            return {
                "referral_status": "not_routed",
                "workflow_status": "completed",
            }
        return {}

    def _authorize_member_session(self, identity: WorkflowOwner) -> None:
        try:
            self._memory.load_facts(
                identity.session_id,
                identity.trusted_member_id,
            )
        except Exception:
            raise WorkflowIdentityError(
                "Trusted member identity does not own an authorized synthetic session"
            ) from None

    def _is_authorized_reviewer(
        self,
        reviewer_id: str,
        identity: Mapping[str, str],
    ) -> bool:
        try:
            return (
                self._reviewer_authorizer(
                    reviewer_id,
                    identity["trusted_member_id"],
                    identity["session_id"],
                )
                is True
            )
        except Exception:
            return False

    def _observe(
        self,
        state: WorkflowState,
        stage: str,
        outcome: str,
        **metadata: Any,
    ) -> None:
        self._record_observation(
            state["request_id"],
            state["session_id"],
            stage,
            outcome,
            **metadata,
        )

    def _record_observation(
        self,
        request_id: str,
        session_id: str,
        stage: str,
        outcome: str,
        **metadata: Any,
    ) -> None:
        if self._observer is None:
            return
        try:
            self._observer.record(
                stage=stage,
                outcome=outcome,
                request_id=request_id,
                session_id=session_id,
                metadata=metadata,
            )
        except Exception:
            return

    def _initialize_registry(self) -> None:
        with closing(
            sqlite3.connect(str(self.checkpoint_path), timeout=10)
        ) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    member_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    thread_id TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    snapshot_json TEXT,
                    PRIMARY KEY (member_id, session_id, idempotency_key),
                    UNIQUE (member_id, session_id, request_id)
                )
                """
            )

    def _reserve_run(
        self,
        identity: WorkflowIdentity,
        thread_id: str,
        request_hash: str,
    ) -> tuple[sqlite3.Row, bool]:
        with self._registry_transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM workflow_runs "
                "WHERE member_id = ? AND session_id = ? AND idempotency_key = ?",
                (
                    identity.trusted_member_id,
                    identity.session_id,
                    identity.idempotency_key,
                ),
            ).fetchone()
            if existing is not None:
                if (
                    existing["request_hash"] != request_hash
                    or existing["request_id"] != identity.request_id
                ):
                    raise WorkflowIdempotencyConflictError(
                        "Workflow idempotency key conflicts with an existing request"
                    )
                return existing, False

            request_id_row = connection.execute(
                "SELECT idempotency_key FROM workflow_runs "
                "WHERE member_id = ? AND session_id = ? AND request_id = ?",
                (
                    identity.trusted_member_id,
                    identity.session_id,
                    identity.request_id,
                ),
            ).fetchone()
            if request_id_row is not None:
                raise WorkflowIdempotencyConflictError(
                    "Workflow request ID is already associated with another key"
                )
            connection.execute(
                "INSERT INTO workflow_runs "
                "(member_id, session_id, request_id, idempotency_key, thread_id, "
                "request_hash, status) VALUES (?, ?, ?, ?, ?, ?, 'running')",
                (
                    identity.trusted_member_id,
                    identity.session_id,
                    identity.request_id,
                    identity.idempotency_key,
                    thread_id,
                    request_hash,
                ),
            )
            row = connection.execute(
                "SELECT * FROM workflow_runs WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            return row, True

    def _claim_resume(
        self,
        identity: WorkflowResumeIdentity,
        thread_id: str,
    ) -> sqlite3.Row:
        with self._registry_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_runs WHERE member_id = ? AND session_id = ? "
                "AND request_id = ? AND idempotency_key = ? AND thread_id = ?",
                (
                    identity.trusted_member_id,
                    identity.session_id,
                    identity.request_id,
                    identity.idempotency_key,
                    thread_id,
                ),
            ).fetchone()
            if row is None:
                raise WorkflowResumeNotAvailableError(
                    "No matching authorized workflow checkpoint was found"
                )
            if row["status"] in {
                "completed",
                "rejected",
                "needs_followup",
                "human_reviewed_no_automation",
            }:
                return row
            if row["status"] != "awaiting_human":
                raise WorkflowInProgressError(
                    "This workflow cannot be resumed in its current state"
                )
            connection.execute(
                "UPDATE workflow_runs SET status = 'resuming' WHERE thread_id = ?",
                (thread_id,),
            )
            row = connection.execute(
                "SELECT * FROM workflow_runs WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            return row

    def _save_snapshot(self, snapshot: WorkflowSnapshot) -> None:
        self._update_run(
            snapshot.workflow_id,
            snapshot.workflow_status,
            snapshot.model_dump_json(),
        )

    def _update_run(
        self,
        thread_id: str,
        status: str,
        snapshot_json: str | None,
    ) -> None:
        with self._registry_transaction() as connection:
            connection.execute(
                "UPDATE workflow_runs SET status = ?, snapshot_json = ? "
                "WHERE thread_id = ?",
                (status, snapshot_json, thread_id),
            )

    def _load_snapshot(self, thread_id: str) -> WorkflowSnapshot:
        graph_state = self._graph.get_state(self._config(thread_id))
        values = graph_state.values
        if not values:
            raise WorkflowResumeNotAvailableError(
                "Workflow checkpoint state is missing"
            )
        interrupt_payload = None
        for task in graph_state.tasks:
            for pending in task.interrupts:
                value = getattr(pending, "value", None)
                if isinstance(value, dict):
                    interrupt_payload = value
                    break
        return WorkflowSnapshot(
            workflow_id=thread_id,
            member_id=values["member_id"],
            session_id=values["session_id"],
            request_id=values["request_id"],
            idempotency_key=values["idempotency_key"],
            intake_result=_coerce_model(
                values.get("intake_result"),
                PatientRequest,
            ),
            eligibility_result=_coerce_model(
                values.get("eligibility_result"),
                InsuranceEligibilityOutput,
            ),
            referral_result=_coerce_model(
                values.get("referral_result"),
                ReferralRoutingOutput,
            ),
            referral_status=values["referral_status"],
            errors=tuple(values.get("errors", ())),
            human_approval_state=values["human_approval_state"],
            reviewer_id=values.get("reviewer_id"),
            review_reason=values.get("review_reason"),
            workflow_status=values["workflow_status"],
            interrupt_payload=interrupt_payload,
        )

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @contextmanager
    def _registry_transaction(self):
        connection = sqlite3.connect(str(self.checkpoint_path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _workflow_thread_id(
    identity: WorkflowIdentity | WorkflowResumeIdentity,
) -> str:
    serialized = json.dumps(
        [
            identity.trusted_member_id,
            identity.session_id,
            identity.idempotency_key,
        ],
        separators=(",", ":"),
    )
    return "careflow-m5-" + hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def _request_hash(request_text: str) -> str:
    return hashlib.sha256(request_text.encode("utf-8")).hexdigest()


def _append_error(errors: list[str], error_code: str) -> list[str]:
    return [*errors, error_code]


def _coerce_model(value: Any, model: type[BaseModel]) -> Any:
    if value is None or isinstance(value, model):
        return value
    if isinstance(value, Mapping):
        try:
            return model.model_validate(value)
        except ValidationError:
            return None
    return None
