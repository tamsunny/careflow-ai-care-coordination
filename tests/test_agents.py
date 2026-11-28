from careflow.agent import SchedulingInsuranceAgent
from careflow.agents import (
    CareFlowAgentTeam,
    ClinicalSafetyReviewer,
    IntakeAgent,
    InsurancePolicyRAGAgent,
    ReferralTrackingAgent,
)
from careflow.memory import SQLiteConversationMemory
from careflow.models import PatientRequest
from careflow.policy_answers import GroundedPolicyAnswerer
from careflow.tool_models import (
    InsuranceEligibilityInput,
)

MEMBER_ONE = "SYN-MEMBER-001"
MEMBER_TWO = "SYN-MEMBER-002"
REVIEWER_ID = "SYN-REVIEWER-M6"


class CountingTools(SchedulingInsuranceAgent):
    def __init__(self):
        super().__init__()
        self.eligibility_calls = 0
        self.referral_calls = 0

    def check_insurance_eligibility(self, request):
        self.eligibility_calls += 1
        return super().check_insurance_eligibility(request)

    def route_referral(self, request):
        self.referral_calls += 1
        return super().route_referral(request)


def create_memory(path):
    return SQLiteConversationMemory(
        path,
        lambda previous, _messages: previous,
        token_counter=lambda text: len(text),
    )


def test_safety_and_intake_agents_escalate_without_calling_parser():
    parser_calls = []
    safety = ClinicalSafetyReviewer()

    def parser(message):
        parser_calls.append(message)
        return PatientRequest(request_type="referral", specialty="cardiology")

    intake = IntakeAgent(parser=parser, safety_reviewer=safety)
    review = safety.review("What exact medication dose should I take?")
    parsed = intake.parse("What exact medication dose should I take?")

    assert review.outcome == "human_review"
    assert "No clinical advice" in safety.response()
    assert parsed.needs_human_review
    assert parser_calls == []


def test_insurance_policy_agent_uses_trusted_tool_and_fails_closed_without_rag():
    tools = CountingTools()
    agent = InsurancePolicyRAGAgent(tools)

    eligibility = agent.check_insurance_eligibility(
        InsuranceEligibilityInput(insurance_id=MEMBER_ONE)
    )
    answer = agent.answer_policy("What does the synthetic policy cover?")
    clinical = agent.answer_policy("What medication should I take?")

    assert eligibility.status == "eligible"
    assert tools.eligibility_calls == 1
    assert answer.status == "insufficient_evidence"
    assert clinical.status == "human_review"


def test_policy_rag_role_uses_existing_index_and_grounded_answerer():
    class EmptyPolicyIndex:
        def __init__(self):
            self.search_arguments = None

        def search(self, query, **kwargs):
            self.search_arguments = (query, kwargs)
            return ()

    index = EmptyPolicyIndex()
    observations = []

    class Observer:
        def record(self, **event):
            observations.append(event)

    answerer = GroundedPolicyAnswerer(
        "mock/no-call",
        completion_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty evidence must fail closed before model call")
        ),
    )
    agent = InsurancePolicyRAGAgent(
        policy_index=index,
        policy_answerer=answerer,
        observer=Observer(),
    )

    result = agent.answer_policy(
        "What is the fictional referral policy?",
        specialty="cardiology",
        plan_id="SYN-PLAN-12",
        version="1.0",
        request_id="SYN-REQ-POLICY-OBSERVE-001",
        session_id="SYN-SESSION-0123456789abcdef0123456789abcdef",
    )

    assert result.status == "insufficient_evidence"
    assert index.search_arguments == (
        "What is the fictional referral policy?",
        {
            "top_k": 3,
            "specialty": "cardiology",
            "plan_id": "SYN-PLAN-12",
            "version": "1.0",
        },
    )
    assert [event["outcome"] for event in observations] == [
        "started",
        "insufficient_evidence",
    ]


def test_referral_tracking_reads_only_the_bound_members_records():
    member_one_tracker = ReferralTrackingAgent(MEMBER_ONE)
    member_two_tracker = ReferralTrackingAgent(MEMBER_TWO)

    own_record = member_one_tracker.lookup_referral("SYN-REF-TRACK-001")
    other_member_record = member_two_tracker.lookup_referral("SYN-REF-TRACK-001")
    unknown_record = member_one_tracker.lookup_referral("SYN-REF-UNKNOWN-001")

    assert own_record is not None
    assert own_record.member_id == MEMBER_ONE
    assert other_member_record is None
    assert unknown_record is None


def test_four_roles_handoff_through_persistent_m5_workflow(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    safety = ClinicalSafetyReviewer()
    intake = IntakeAgent(
        parser=lambda _message: PatientRequest(
            request_type="referral",
            specialty="cardiology",
        ),
        safety_reviewer=safety,
    )
    team = CareFlowAgentTeam(
        tmp_path / "workflow.sqlite3",
        memory,
        MEMBER_ONE,
        lambda reviewer, member, session: reviewer == REVIEWER_ID
        and member == MEMBER_ONE
        and session == session_id,
        intake_agent=intake,
        clinical_safety_reviewer=safety,
        insurance_policy_agent=InsurancePolicyRAGAgent(tools),
        referral_tracking_agent=ReferralTrackingAgent(MEMBER_ONE, tools),
    )

    paused = team.start_workflow(
        session_id=session_id,
        request_id="SYN-REQ-M6-TEAM-001",
        idempotency_key="m6-team-req-001",
        request_text="Please route a cardiology referral.",
    )
    assert paused.workflow_status == "awaiting_human"
    assert paused.eligibility_result.status == "eligible"
    assert tools.eligibility_calls == 1
    assert tools.referral_calls == 0
    duplicate = team.start_workflow(
        session_id=session_id,
        request_id="SYN-REQ-M6-TEAM-001",
        idempotency_key="m6-team-req-001",
        request_text="Please route a cardiology referral.",
    )
    assert duplicate.workflow_id == paused.workflow_id
    assert tools.eligibility_calls == 1

    completed = team.resume_workflow(
        session_id=session_id,
        request_id="SYN-REQ-M6-TEAM-001",
        idempotency_key="m6-team-req-001",
        reviewer_id=REVIEWER_ID,
        decision="approve",
    )
    assert completed.referral_result.status == "routed"
    assert tools.referral_calls == 1
    team.close()
