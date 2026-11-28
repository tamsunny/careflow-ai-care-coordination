from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from careflow.memory import SQLiteConversationMemory
from careflow.models import PatientRequest
from careflow.tool_models import (
    InsuranceEligibilityOutput,
    ReferralRoutingOutput,
)
from careflow.workflow import (
    LangGraphWorkflow,
    WorkflowIdempotencyConflictError,
    WorkflowIdentityError,
    WorkflowInProgressError,
    WorkflowUnauthorizedResumeError,
)

MEMBER_ONE = "SYN-MEMBER-001"
MEMBER_TWO = "SYN-MEMBER-002"
REQUEST_ID = "SYN-REQ-WORKFLOW-001"
IDEMPOTENCY_KEY = "workflow-turn-001"
REVIEWER_ID = "SYN-REVIEWER-DEMO"
REQUEST_TEXT = "Please create a cardiology referral."


class CountingTools:
    def __init__(self, status="eligible"):
        self.eligibility_status = status
        self.eligibility_calls = 0
        self.referral_calls = 0

    def check_insurance_eligibility(self, request):
        self.eligibility_calls += 1
        return InsuranceEligibilityOutput(
            status=self.eligibility_status,
            insurance_id=request.insurance_id,
            eligible=(
                True if self.eligibility_status == "eligible"
                else False if self.eligibility_status == "ineligible"
                else None
            ),
            plan_id="SYN-PLAN-12" if self.eligibility_status == "eligible" else None,
            coverage_end=None,
        )

    def route_referral(self, request):
        self.referral_calls += 1
        return ReferralRoutingOutput(
            status="routed",
            request_id=request.request_id,
            referral_id="SYN-REF-WORKFLOW-001",
            specialty=request.specialty,
            destination="Synthetic Cardiology Clinic",
        )


def create_memory(path):
    return SQLiteConversationMemory(
        path,
        lambda previous, messages: previous,
        token_counter=lambda text: len(text),
    )


def referral_parser(_text):
    return PatientRequest(
        request_type="referral",
        specialty="cardiology",
    )


def authorize_all(reviewer_id, member_id, session_id):
    return reviewer_id == REVIEWER_ID and member_id in {MEMBER_ONE, MEMBER_TWO}


def make_workflow(tmp_path, memory, *, parser=referral_parser, tools=None, authorizer=authorize_all):
    return LangGraphWorkflow(
        tmp_path / "workflow-checkpoints.sqlite3",
        memory,
        authorizer,
        intake_parser=parser,
        tool_executor=tools or CountingTools(),
    )


def start(workflow, session_id, *, request_text=REQUEST_TEXT):
    return workflow.start_workflow(
        trusted_member_id=MEMBER_ONE,
        session_id=session_id,
        request_id=REQUEST_ID,
        idempotency_key=IDEMPOTENCY_KEY,
        request_text=request_text,
    )


def resume(workflow, session_id, *, decision="approve", member_id=MEMBER_ONE):
    return workflow.resume_workflow(
        trusted_member_id=member_id,
        session_id=session_id,
        request_id=REQUEST_ID,
        idempotency_key=IDEMPOTENCY_KEY,
        reviewer_id=REVIEWER_ID,
        decision=decision,
    )


def test_referral_requires_approval_and_resumes_after_restart(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    parser_calls = []

    def parser(text):
        parser_calls.append(text)
        return referral_parser(text)

    workflow = make_workflow(tmp_path, memory, parser=parser, tools=tools)
    paused = start(workflow, session_id)

    assert paused.interrupted
    assert paused.workflow_status == "awaiting_human"
    assert paused.eligibility_result.status == "eligible"
    assert tools.referral_calls == 0
    workflow.close()

    restarted = make_workflow(tmp_path, memory, parser=parser, tools=tools)
    completed = resume(restarted, session_id)

    assert completed.workflow_status == "completed"
    assert completed.referral_status == "routed"
    assert completed.referral_result.referral_id == "SYN-REF-WORKFLOW-001"
    assert completed.reviewer_id == REVIEWER_ID
    assert parser_calls == [REQUEST_TEXT]
    assert tools.eligibility_calls == 1
    assert tools.referral_calls == 1


@pytest.mark.parametrize("eligibility_status", ["ineligible", "not_found"])
def test_missing_or_ineligible_coverage_cannot_route_after_approval(
    tmp_path,
    eligibility_status,
):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools(status=eligibility_status)
    workflow = make_workflow(tmp_path, memory, tools=tools)

    paused = start(workflow, session_id)
    assert paused.eligibility_result.status == eligibility_status
    completed = resume(workflow, session_id)

    assert completed.workflow_status == "needs_followup"
    assert completed.referral_status == "blocked_eligibility"
    assert completed.referral_result is None
    assert tools.referral_calls == 0


def test_clinical_request_bypasses_parser_and_tools_even_after_review(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    parser_calls = []

    def forbidden_parser(text):
        parser_calls.append(text)
        raise AssertionError("clinical request must bypass intake parsing")

    workflow = make_workflow(
        tmp_path,
        memory,
        parser=forbidden_parser,
        tools=tools,
    )
    paused = start(
        workflow,
        session_id,
        request_text="What exact medication dose should I take?",
    )

    assert paused.interrupted
    assert paused.review_reason == "clinical_advice_request"
    assert paused.intake_result is None
    completed = resume(workflow, session_id)

    assert completed.workflow_status == "human_reviewed_no_automation"
    assert completed.referral_status == "not_routed"
    assert parser_calls == []
    assert tools.eligibility_calls == 0
    assert tools.referral_calls == 0


def test_unauthorized_resume_and_cross_member_session_access_are_rejected(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    allowed = False

    def authorizer(reviewer_id, member_id, authorized_session_id):
        return allowed and reviewer_id == REVIEWER_ID

    workflow = make_workflow(tmp_path, memory, authorizer=authorizer)
    start(workflow, session_id)

    with pytest.raises(WorkflowUnauthorizedResumeError):
        resume(workflow, session_id)
    with pytest.raises(WorkflowIdentityError):
        workflow.start_workflow(
            trusted_member_id=MEMBER_TWO,
            session_id=session_id,
            request_id="SYN-REQ-WORKFLOW-002",
            idempotency_key="workflow-turn-other-member",
            request_text=REQUEST_TEXT,
        )

    allowed = True
    assert resume(workflow, session_id).workflow_status == "completed"


def test_model_identity_claim_is_not_used_for_eligibility_or_routing(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    workflow = make_workflow(
        tmp_path,
        memory,
        parser=lambda _text: PatientRequest(
            request_type="referral",
            specialty="cardiology",
            insurance_id=MEMBER_TWO,
        ),
        tools=tools,
    )

    paused = start(workflow, session_id)
    assert paused.review_reason == "intake_identity_mismatch"
    assert paused.eligibility_result is None
    assert tools.eligibility_calls == 0

    completed = resume(workflow, session_id)
    assert completed.workflow_status == "needs_followup"
    assert completed.referral_result is None
    assert tools.referral_calls == 0


def test_missing_specialty_and_intake_failure_do_not_route(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    missing_specialty = make_workflow(
        tmp_path,
        memory,
        parser=lambda _text: PatientRequest(request_type="referral"),
        tools=tools,
    )

    paused = start(missing_specialty, session_id)
    assert paused.intake_result.specialty is None
    assert paused.review_reason == "referral_type_or_specialty_unclear"
    assert tools.eligibility_calls == 0
    reviewed = resume(missing_specialty, session_id)
    assert reviewed.workflow_status == "needs_followup"
    assert tools.referral_calls == 0
    missing_specialty.close()

    def fail_parser(_text):
        raise ValueError("invalid test response")

    failed_parse = make_workflow(
        tmp_path,
        memory,
        parser=fail_parser,
        tools=tools,
    )
    failed = failed_parse.start_workflow(
        trusted_member_id=MEMBER_ONE,
        session_id=session_id,
        request_id="SYN-REQ-WORKFLOW-002",
        idempotency_key="workflow-turn-002",
        request_text="Please refer me to an unspecified specialist.",
    )
    assert failed.workflow_status == "awaiting_human"
    assert failed.errors == ("intake_parse_failed",)
    assert tools.referral_calls == 0


def test_idempotent_start_reuses_checkpoint_and_rejects_conflicting_request(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    parser_calls = []

    def parser(text):
        parser_calls.append(text)
        return referral_parser(text)

    workflow = make_workflow(tmp_path, memory, parser=parser, tools=tools)
    first = start(workflow, session_id)
    duplicate = start(workflow, session_id)

    assert duplicate.workflow_id == first.workflow_id
    assert parser_calls == [REQUEST_TEXT]
    with pytest.raises(WorkflowIdempotencyConflictError):
        start(workflow, session_id, request_text=REQUEST_TEXT + " ")


def test_duplicate_resume_returns_stored_result_without_rerouting(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    tools = CountingTools()
    workflow = make_workflow(tmp_path, memory, tools=tools)
    start(workflow, session_id)

    completed = resume(workflow, session_id, decision="approve")
    duplicate = resume(workflow, session_id, decision="reject")

    assert duplicate == completed
    assert tools.referral_calls == 1


def test_concurrent_duplicate_start_is_rejected_while_first_is_running(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    entered_parser = Event()
    release_parser = Event()

    def blocking_parser(_text):
        entered_parser.set()
        assert release_parser.wait(timeout=5)
        return referral_parser(_text)

    workflow = make_workflow(tmp_path, memory, parser=blocking_parser)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(start, workflow, session_id)
        assert entered_parser.wait(timeout=5)
        with pytest.raises(WorkflowInProgressError):
            start(workflow, session_id)
        release_parser.set()
        paused = first.result(timeout=5)

    assert paused.interrupted
