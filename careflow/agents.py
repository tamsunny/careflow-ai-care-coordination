"""Explicit, small role adapters coordinated by the M5 LangGraph workflow."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent import SchedulingInsuranceAgent
from .intake import is_clinical_advice_request, parse_patient_message
from .memory import SQLiteConversationMemory
from .mock_data import MOCK_MEMBERS
from .models import PatientRequest
from .observability import WorkflowObserver
from .policy_answers import (
    HUMAN_REVIEW_ANSWER,
    INSUFFICIENT_EVIDENCE_ANSWER,
    GroundedPolicyAnswer,
    GroundedPolicyAnswerer,
)
from .policy_index import LocalPolicyIndex
from .resilience import ResilienceConfiguration, ResilientMockServices
from .tool_models import (
    InsuranceEligibilityInput,
    InsuranceEligibilityOutput,
    ReferralLookupInput,
    ReferralRoutingInput,
    ReferralRoutingOutput,
    ReferralTrackingRecord,
)
from .workflow import LangGraphWorkflow, ReviewerAuthorizer, WorkflowSnapshot


class SafetyReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["administrative", "human_review"]
    reason: str | None = None


class ClinicalSafetyReviewer:
    """Escalate detected clinical-advice requests without generating advice."""

    def review(self, message: str) -> SafetyReview:
        if is_clinical_advice_request(message):
            return SafetyReview(
                outcome="human_review",
                reason="clinical_advice_request",
            )
        return SafetyReview(outcome="administrative")

    def is_clinical_request(self, message: str) -> bool:
        return self.review(message).outcome == "human_review"

    @staticmethod
    def response() -> str:
        return HUMAN_REVIEW_ANSWER


class IntakeAgent:
    """Parse synthetic administrative requests into the existing M1 schema."""

    def __init__(
        self,
        *,
        model: str = "mock/m6-intake",
        completion_fn: Callable[..., Any] | None = None,
        parser: Callable[[str], PatientRequest] | None = None,
        safety_reviewer: ClinicalSafetyReviewer | None = None,
    ) -> None:
        self._safety_reviewer = safety_reviewer or ClinicalSafetyReviewer()
        self._parser = parser or (
            lambda message: parse_patient_message(
                message,
                model,
                completion_fn=completion_fn,
            )
        )

    def parse(self, message: str) -> PatientRequest:
        review = self._safety_reviewer.review(message)
        if review.outcome == "human_review":
            return PatientRequest(
                request_type="clinical_advice",
                needs_human_review=True,
                reason_for_review=review.reason,
            )
        result = self._parser(message)
        if isinstance(result, PatientRequest):
            return result
        return PatientRequest.model_validate(result)


class InsurancePolicyRAGAgent:
    """Use trusted synthetic eligibility tools and optional grounded policy RAG."""

    def __init__(
        self,
        tool_executor: Any | None = None,
        *,
        policy_index: LocalPolicyIndex | None = None,
        policy_answerer: GroundedPolicyAnswerer | None = None,
        observer: WorkflowObserver | None = None,
    ) -> None:
        if (policy_index is None) != (policy_answerer is None):
            raise ValueError("Policy index and grounded answerer must be configured together")
        self._tool_executor = tool_executor or SchedulingInsuranceAgent()
        self._policy_index = policy_index
        self._policy_answerer = policy_answerer
        self._observer = observer

    def check_insurance_eligibility(
        self,
        request: InsuranceEligibilityInput,
    ) -> InsuranceEligibilityOutput:
        return self._tool_executor.check_insurance_eligibility(request)

    def answer_policy(
        self,
        question: str,
        *,
        top_k: int = 3,
        specialty: str | None = None,
        plan_id: str | None = None,
        version: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> GroundedPolicyAnswer:
        if self._observer and request_id and session_id:
            _record_observation(
                self._observer,
                "policy_retrieval",
                "started",
                request_id,
                session_id,
            )
        if is_clinical_advice_request(question):
            answer = GroundedPolicyAnswer(
                status="human_review",
                answer=HUMAN_REVIEW_ANSWER,
            )
            if self._observer and request_id and session_id:
                _record_observation(
                    self._observer,
                    "policy_retrieval",
                    "human_review",
                    request_id,
                    session_id,
                )
            return answer
        if self._policy_index is None or self._policy_answerer is None:
            answer = GroundedPolicyAnswer(
                status="insufficient_evidence",
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
            )
            if self._observer and request_id and session_id:
                _record_observation(
                    self._observer,
                    "policy_retrieval",
                    "insufficient_evidence",
                    request_id,
                    session_id,
                )
            return answer
        try:
            evidence = self._policy_index.search(
                question,
                top_k=top_k,
                specialty=specialty,
                plan_id=plan_id,
                version=version,
            )
            answer = self._policy_answerer.answer(
                question,
                evidence,
                expected_specialty=specialty,
                expected_plan_id=plan_id,
                expected_version=version,
            )
        except Exception:
            if self._observer and request_id and session_id:
                _record_observation(
                    self._observer,
                    "policy_retrieval",
                    "failed",
                    request_id,
                    session_id,
                    error_type="retrieval_error",
                )
            raise
        if self._observer and request_id and session_id:
            _record_observation(
                self._observer,
                "policy_retrieval",
                answer.status if answer.status in {
                    "human_review",
                    "insufficient_evidence",
                } else "success",
                request_id,
                session_id,
            )
        return answer


class ReferralTrackingAgent:
    """Route referrals through M2 tools and read owner-scoped mock records."""

    def __init__(
        self,
        trusted_member_id: str,
        tool_executor: Any | None = None,
    ) -> None:
        if trusted_member_id not in MOCK_MEMBERS:
            raise ValueError("A supported trusted synthetic member is required")
        self._trusted_member_id = trusted_member_id
        self._tool_executor = tool_executor or SchedulingInsuranceAgent()

    def route_referral(
        self,
        request: ReferralRoutingInput,
    ) -> ReferralRoutingOutput:
        return self._tool_executor.route_referral(request)

    def lookup_referral(
        self,
        request: ReferralLookupInput | str,
    ) -> ReferralTrackingRecord | None:
        lookup = (
            request
            if isinstance(request, ReferralLookupInput)
            else ReferralLookupInput(referral_id=request)
        )
        return self._tool_executor.lookup_referral(
            lookup,
            self._trusted_member_id,
        )


class _RoleToolRouter:
    """Bind the workflow's typed tools to the corresponding M6 roles."""

    def __init__(
        self,
        trusted_member_id: str,
        insurance_agent: InsurancePolicyRAGAgent,
        referral_agent: ReferralTrackingAgent,
    ) -> None:
        self._trusted_member_id = trusted_member_id
        self._insurance_agent = insurance_agent
        self._referral_agent = referral_agent

    def check_insurance_eligibility(
        self,
        request: InsuranceEligibilityInput,
    ) -> InsuranceEligibilityOutput:
        if request.insurance_id != self._trusted_member_id:
            raise PermissionError("Eligibility lookup must use the trusted member")
        return self._insurance_agent.check_insurance_eligibility(request)

    def route_referral(
        self,
        request: ReferralRoutingInput,
    ) -> ReferralRoutingOutput:
        return self._referral_agent.route_referral(request)


class CareFlowAgentTeam:
    """Wire four explicit role agents into the persistent M5 LangGraph."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        memory: SQLiteConversationMemory,
        trusted_member_id: str,
        reviewer_authorizer: ReviewerAuthorizer,
        *,
        intake_agent: IntakeAgent | None = None,
        clinical_safety_reviewer: ClinicalSafetyReviewer | None = None,
        insurance_policy_agent: InsurancePolicyRAGAgent | None = None,
        referral_tracking_agent: ReferralTrackingAgent | None = None,
        tool_executor: SchedulingInsuranceAgent | None = None,
        resilience_config: ResilienceConfiguration | None = None,
        observer: WorkflowObserver | None = None,
    ) -> None:
        if trusted_member_id not in MOCK_MEMBERS:
            raise ValueError("A supported trusted synthetic member is required")
        base_tools = tool_executor or SchedulingInsuranceAgent()
        self.resilient_services = ResilientMockServices(
            base_tools,
            resilience_config,
        )
        self.trusted_member_id = trusted_member_id
        self.clinical_safety_reviewer = (
            clinical_safety_reviewer or ClinicalSafetyReviewer()
        )
        self.intake_agent = intake_agent or IntakeAgent(
            safety_reviewer=self.clinical_safety_reviewer
        )
        self.insurance_policy_agent = insurance_policy_agent or (
            InsurancePolicyRAGAgent(
                self.resilient_services,
                observer=observer,
            )
        )
        if insurance_policy_agent is not None and observer is not None:
            insurance_policy_agent._observer = observer
        self.referral_tracking_agent = referral_tracking_agent or (
            ReferralTrackingAgent(trusted_member_id, self.resilient_services)
        )
        role_tools = _RoleToolRouter(
            trusted_member_id,
            self.insurance_policy_agent,
            self.referral_tracking_agent,
        )
        self.workflow = LangGraphWorkflow(
            checkpoint_path,
            memory,
            reviewer_authorizer,
            intake_parser=self.intake_agent.parse,
            tool_executor=role_tools,
            safety_checker=self.clinical_safety_reviewer.is_clinical_request,
            observer=observer,
        )

    def close(self) -> None:
        self.workflow.close()
        self.resilient_services.close()

    def __enter__(self) -> "CareFlowAgentTeam":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def start_workflow(
        self,
        *,
        session_id: str,
        request_id: str,
        idempotency_key: str,
        request_text: str,
    ) -> WorkflowSnapshot:
        return self.workflow.start_workflow(
            trusted_member_id=self.trusted_member_id,
            session_id=session_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            request_text=request_text,
        )

    def resume_workflow(
        self,
        *,
        session_id: str,
        request_id: str,
        idempotency_key: str,
        reviewer_id: str,
        decision: Literal["approve", "reject"],
    ) -> WorkflowSnapshot:
        return self.workflow.resume_workflow(
            trusted_member_id=self.trusted_member_id,
            session_id=session_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            reviewer_id=reviewer_id,
            decision=decision,
        )

    def get_workflow_snapshot(
        self,
        *,
        session_id: str,
        workflow_id: str,
    ) -> WorkflowSnapshot:
        return self.workflow.get_workflow_snapshot(
            trusted_member_id=self.trusted_member_id,
            session_id=session_id,
            workflow_id=workflow_id,
        )

    def lookup_referral(self, referral_id: str) -> ReferralTrackingRecord | None:
        return self.referral_tracking_agent.lookup_referral(referral_id)


def _record_observation(
    observer: WorkflowObserver,
    stage: str,
    outcome: str,
    request_id: str,
    session_id: str,
    **metadata: Any,
) -> None:
    try:
        observer.record(
            stage=stage,
            outcome=outcome,
            request_id=request_id,
            session_id=session_id,
            metadata=metadata,
        )
    except Exception:
        return
