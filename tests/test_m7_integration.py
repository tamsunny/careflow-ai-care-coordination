from careflow.agent import SchedulingInsuranceAgent
from careflow.agents import CareFlowAgentTeam, IntakeAgent
from careflow.memory import SQLiteConversationMemory
from careflow.models import PatientRequest
from careflow.resilience import (
    ResilienceConfiguration,
    TransientServiceError,
)

MEMBER = "SYN-MEMBER-001"
REVIEWER = "SYN-REVIEWER-M7"


def create_memory(path):
    return SQLiteConversationMemory(
        path,
        lambda previous, _messages: previous,
        token_counter=lambda text: len(text),
    )


def parser_agent():
    return IntakeAgent(
        parser=lambda _message: PatientRequest(
            request_type="referral",
            specialty="cardiology",
        )
    )


class UnavailableEligibility(SchedulingInsuranceAgent):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def check_insurance_eligibility(self, request):
        self.calls += 1
        raise TransientServiceError("synthetic outage")


def test_unavailable_eligibility_becomes_checkpointed_human_review(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER)
    tools = UnavailableEligibility()
    team = CareFlowAgentTeam(
        tmp_path / "workflow.sqlite3",
        memory,
        MEMBER,
        lambda reviewer, member, session: reviewer == REVIEWER and member == MEMBER,
        intake_agent=parser_agent(),
        tool_executor=tools,
        resilience_config=ResilienceConfiguration(
            max_attempts=2,
            failure_threshold=1,
            base_delay_seconds=0,
            max_delay_seconds=0,
            timeout_seconds=1,
        ),
    )
    try:
        paused = team.start_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-UNAVAILABLE-001",
            idempotency_key="m7-unavailable-001",
            request_text="Please refer me to cardiology.",
        )
        reviewed = team.resume_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-UNAVAILABLE-001",
            idempotency_key="m7-unavailable-001",
            reviewer_id=REVIEWER,
            decision="approve",
        )
    finally:
        team.close()

    assert tools.calls == 2
    assert paused.interrupted
    assert paused.eligibility_result is None
    assert "eligibility_service_unavailable" in paused.errors
    assert reviewed.workflow_status == "needs_followup"
    assert reviewed.referral_result is None


def test_timed_out_referral_mutation_runs_once_and_duplicate_resume_is_stored(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER)

    class UncertainReferral(SchedulingInsuranceAgent):
        def __init__(self):
            super().__init__()
            self.route_calls = 0

        def route_referral(self, request):
            self.route_calls += 1
            raise TransientServiceError("uncertain mutation response")

    tools = UncertainReferral()
    team = CareFlowAgentTeam(
        tmp_path / "workflow.sqlite3",
        memory,
        MEMBER,
        lambda reviewer, member, session: reviewer == REVIEWER and member == MEMBER,
        intake_agent=parser_agent(),
        tool_executor=tools,
        resilience_config=ResilienceConfiguration(
            max_attempts=5,
            base_delay_seconds=0,
            max_delay_seconds=0,
            timeout_seconds=1,
        ),
    )
    try:
        team.start_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-ONCE-001",
            idempotency_key="m7-once-001",
            request_text="Please refer me to cardiology.",
        )
        failed_action = team.resume_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-ONCE-001",
            idempotency_key="m7-once-001",
            reviewer_id=REVIEWER,
            decision="approve",
        )
        duplicate = team.resume_workflow(
            session_id=session_id,
            request_id="SYN-REQ-M7-ONCE-001",
            idempotency_key="m7-once-001",
            reviewer_id=REVIEWER,
            decision="approve",
        )
    finally:
        team.close()

    assert tools.route_calls == 1
    assert failed_action.workflow_status == "needs_followup"
    assert "referral_service_unavailable" in failed_action.errors
    assert duplicate == failed_action
