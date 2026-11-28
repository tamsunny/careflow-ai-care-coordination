"""Offline M7 retry, human checkpoint, and clinical escalation demonstration."""

import json
from pathlib import Path

from .agent import SchedulingInsuranceAgent
from .agents import CareFlowAgentTeam, IntakeAgent
from .memory import SQLiteConversationMemory
from .models import PatientRequest
from .resilience import ResilienceConfiguration, TransientServiceError

MEMBER_ID = "SYN-MEMBER-001"
REVIEWER_ID = "SYN-REVIEWER-M7-DEMO"
DATA_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "data" / "m7_failure_demo"
)


class FailureInjectedTools(SchedulingInsuranceAgent):
    def __init__(self) -> None:
        super().__init__()
        self.eligibility_attempts = 0
        self.referral_calls = 0

    def check_insurance_eligibility(self, request):
        self.eligibility_attempts += 1
        if self.eligibility_attempts == 1:
            raise TransientServiceError("injected first-call failure")
        return super().check_insurance_eligibility(request)

    def route_referral(self, request):
        self.referral_calls += 1
        return super().route_referral(request)


def _offline_intake(_message: str) -> PatientRequest:
    return PatientRequest(request_type="referral", specialty="cardiology")


def main() -> None:
    memory = SQLiteConversationMemory(
        DATA_DIRECTORY / "memory.sqlite3",
        lambda previous, _messages: previous,
        token_counter=lambda text: len(text),
    )
    session_id = memory.create_session(MEMBER_ID)
    tools = FailureInjectedTools()

    def authorize(reviewer_id: str, member_id: str, session: str) -> bool:
        return (
            reviewer_id == REVIEWER_ID
            and member_id == MEMBER_ID
            and session == session_id
        )

    team = CareFlowAgentTeam(
        DATA_DIRECTORY / "workflow.sqlite3",
        memory,
        MEMBER_ID,
        authorize,
        intake_agent=IntakeAgent(parser=_offline_intake),
        tool_executor=tools,
        resilience_config=ResilienceConfiguration(
            max_attempts=2,
            base_delay_seconds=0,
            max_delay_seconds=0,
            timeout_seconds=1,
        ),
    )
    try:
        paused = team.start_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-RETRY-001",
            idempotency_key="m7-demo-retry-001",
            request_text="Please route a cardiology referral.",
        )
        completed = team.resume_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-RETRY-001",
            idempotency_key="m7-demo-retry-001",
            reviewer_id=REVIEWER_ID,
            decision="approve",
        )
        clinical = team.start_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-CLINICAL-001",
            idempotency_key="m7-demo-clinical-001",
            request_text="What exact medication dose should I take?",
        )
    finally:
        team.close()

    print(
        json.dumps(
            {
                "synthetic_only": True,
                "failure_injected": "first eligibility call raises transient error",
                "eligibility_attempts": tools.eligibility_attempts,
                "eligibility_recovered": (
                    paused.eligibility_result.status == "eligible"
                ),
                "approval_checkpoint_reached": paused.interrupted,
                "approved_referral_status": completed.referral_status,
                "referral_calls": tools.referral_calls,
                "clinical_escalation": {
                    "paused_for_review": clinical.interrupted,
                    "reason": clinical.review_reason,
                    "no_referral_created": tools.referral_calls == 1,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
