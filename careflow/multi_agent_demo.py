"""Offline synthetic demonstration of the four M6 roles and M5 checkpoints."""

import json
from pathlib import Path

from .agents import CareFlowAgentTeam, IntakeAgent
from .memory import SQLiteConversationMemory
from .models import PatientRequest

MEMBER_ID = "SYN-MEMBER-001"
REVIEWER_ID = "SYN-REVIEWER-M6-DEMO"
DATA_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "data" / "m6_multi_agent_demo"
)


def _offline_intake(_message: str) -> PatientRequest:
    return PatientRequest(
        request_type="referral",
        specialty="cardiology",
    )


def main() -> None:
    memory = SQLiteConversationMemory(
        DATA_DIRECTORY / "memory.sqlite3",
        lambda previous, _messages: previous,
        token_counter=lambda text: len(text),
    )
    session_id = memory.create_session(MEMBER_ID)

    def authorize_reviewer(reviewer_id: str, member_id: str, session: str) -> bool:
        return (
            reviewer_id == REVIEWER_ID
            and member_id == MEMBER_ID
            and session == session_id
        )

    team = CareFlowAgentTeam(
        DATA_DIRECTORY / "checkpoints.sqlite3",
        memory,
        MEMBER_ID,
        authorize_reviewer,
        intake_agent=IntakeAgent(parser=_offline_intake),
    )
    try:
        administrative_pause = team.start_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M6-DEMO-ADMIN",
            idempotency_key="m6-demo-administrative-001",
            request_text="Please route a cardiology referral.",
        )
        administrative_result = team.resume_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M6-DEMO-ADMIN",
            idempotency_key="m6-demo-administrative-001",
            reviewer_id=REVIEWER_ID,
            decision="approve",
        )

        clinical_pause = team.start_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M6-DEMO-CLINICAL",
            idempotency_key="m6-demo-clinical-001",
            request_text="What exact medication dose should I take?",
        )
        clinical_result = team.resume_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M6-DEMO-CLINICAL",
            idempotency_key="m6-demo-clinical-001",
            reviewer_id=REVIEWER_ID,
            decision="approve",
        )
    finally:
        team.close()

    print(
        json.dumps(
            {
                "synthetic_only": True,
                "agents": [
                    "ClinicalSafetyReviewer",
                    "IntakeAgent",
                    "InsurancePolicyRAGAgent",
                    "ReferralTrackingAgent",
                ],
                "administrative_scenario": {
                    "paused_for_review": administrative_pause.interrupted,
                    "eligibility_status": (
                        administrative_result.eligibility_result.status
                    ),
                    "referral_status": administrative_result.referral_status,
                    "referral_id": (
                        administrative_result.referral_result.referral_id
                    ),
                    "reviewer_id": administrative_result.reviewer_id,
                },
                "clinical_escalation": {
                    "paused_for_review": clinical_pause.interrupted,
                    "reason": clinical_pause.review_reason,
                    "outcome": clinical_result.workflow_status,
                    "referral_status": clinical_result.referral_status,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
