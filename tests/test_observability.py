import pytest

from careflow.agents import CareFlowAgentTeam, IntakeAgent
from careflow.memory import SQLiteConversationMemory
from careflow.models import PatientRequest
from careflow.observability import LangfuseWorkflowTracer
from careflow.tool_models import ReferralRoutingOutput
from careflow.workflow import WorkflowUnauthorizedResumeError

MEMBER = "SYN-MEMBER-001"
SESSION = "SYN-SESSION-0123456789abcdef0123456789abcdef"
REQUEST = "SYN-REQ-OBSERVE-001"


class FakeSpan:
    def __init__(self, sink, fields):
        self.sink = sink
        self.fields = fields

    def end(self):
        self.sink.append(self.fields)


class FakeTrace:
    def __init__(self, sink, fields):
        self.sink = sink
        self.fields = fields

    def span(self, **kwargs):
        self.fields["span"] = kwargs
        return FakeSpan(self.sink, self.fields)


class FakeLangfuseClient:
    def __init__(self):
        self.traces = []

    def trace(self, **kwargs):
        captured = {"trace": kwargs}
        self.traces.append(captured)
        return FakeTrace(self.traces, captured)


def create_memory(path):
    return SQLiteConversationMemory(
        path,
        lambda previous, _messages: previous,
        token_counter=lambda text: len(text),
    )


def test_langfuse_trace_correlates_synthetic_ids_and_sanitizes_fields():
    client = FakeLangfuseClient()
    tracer = LangfuseWorkflowTracer(client)

    recorded = tracer.record(
        stage="intake",
        outcome="success",
        request_id=REQUEST,
        session_id=SESSION,
        metadata={
            "attempt": 2,
            "reason_code": "eligible",
            "request_text": "synthetic input should not be logged",
            "api_key": "do-not-log",
            "member_id": MEMBER,
        },
    )

    assert recorded is True
    fields = client.traces[0]
    assert fields["trace"]["metadata"] == {
        "synthetic_data": True,
        "request_id": REQUEST,
        "session_id": SESSION,
    }
    assert fields["span"]["metadata"] == {
        "attempt": 2,
        "reason_code": "eligible",
    }
    assert "request_text" not in repr(fields)
    assert "api_key" not in repr(fields)
    assert MEMBER not in repr(fields)


def test_malformed_ids_and_unavailable_langfuse_are_nonfatal():
    assert not LangfuseWorkflowTracer(None).record(
        stage="intake",
        outcome="success",
        request_id="not-synthetic",
        session_id=SESSION,
    )
    assert not LangfuseWorkflowTracer(FakeLangfuseClient()).record(
        stage="intake",
        outcome="success",
        request_id=REQUEST,
        session_id="not-a-session",
    )

    class BrokenClient:
        def trace(self, **_kwargs):
            raise RuntimeError("secret details must not escape")

    assert not LangfuseWorkflowTracer(BrokenClient()).record(
        stage="intake",
        outcome="success",
        request_id=REQUEST,
        session_id=SESSION,
    )


def test_tracing_errors_do_not_change_workflow_outcome_or_resume_auth(tmp_path):
    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER)

    class BrokenObserver:
        def record(self, **_kwargs):
            raise RuntimeError("trace service unavailable")

    class Tools:
        def check_insurance_eligibility(self, request):
            from careflow.agent import SchedulingInsuranceAgent

            return SchedulingInsuranceAgent().check_insurance_eligibility(request)

        def route_referral(self, request):
            return ReferralRoutingOutput(
                status="routed",
                request_id=request.request_id,
                referral_id="SYN-REF-OBSERVE-001",
                specialty=request.specialty,
                destination="Fictional Cardiology Referral Desk",
            )

    reviewer = "SYN-REVIEWER-OBSERVE"
    team = CareFlowAgentTeam(
        tmp_path / "workflow.sqlite3",
        memory,
        MEMBER,
        lambda supplied, member, session: (
            supplied == reviewer and member == MEMBER and session == session_id
        ),
        intake_agent=IntakeAgent(
            parser=lambda _message: PatientRequest(
                request_type="referral",
                specialty="cardiology",
            )
        ),
        tool_executor=Tools(),
        observer=BrokenObserver(),
    )

    paused = team.start_workflow(
        session_id=session_id,
        request_id=REQUEST,
        idempotency_key="obs-idem-001",
        request_text="Please refer me to cardiology.",
    )
    with pytest.raises(WorkflowUnauthorizedResumeError):
        team.resume_workflow(
            session_id=session_id,
            request_id=REQUEST,
            idempotency_key="obs-idem-001",
            reviewer_id="SYN-REVIEWER-UNAUTHORIZED",
            decision="approve",
        )
    completed = team.resume_workflow(
        session_id=session_id,
        request_id=REQUEST,
        idempotency_key="obs-idem-001",
        reviewer_id=reviewer,
        decision="approve",
    )
    team.close()

    assert paused.interrupted
    assert completed.referral_status == "routed"


def test_m5_workflow_records_sanitized_stage_spans(tmp_path):
    from careflow.langfuse_evaluation import LangfuseEvaluationLogger

    memory = create_memory(tmp_path / "memory.sqlite3")
    session_id = memory.create_session(MEMBER)
    client = FakeLangfuseClient()
    tracer = LangfuseEvaluationLogger(client).workflow_tracer()
    reviewer = "SYN-REVIEWER-OBSERVE"

    class Tools:
        def check_insurance_eligibility(self, request):
            from careflow.agent import SchedulingInsuranceAgent

            return SchedulingInsuranceAgent().check_insurance_eligibility(request)

        def route_referral(self, request):
            return ReferralRoutingOutput(
                status="routed",
                request_id=request.request_id,
                referral_id="SYN-REF-OBSERVE-002",
                specialty=request.specialty,
                destination="Fictional Cardiology Referral Desk",
            )

    team = CareFlowAgentTeam(
        tmp_path / "workflow.sqlite3",
        memory,
        MEMBER,
        lambda supplied, member, session: (
            supplied == reviewer and member == MEMBER and session == session_id
        ),
        intake_agent=IntakeAgent(
            parser=lambda _message: PatientRequest(
                request_type="referral",
                specialty="cardiology",
            )
        ),
        tool_executor=Tools(),
        observer=tracer,
    )
    try:
        team.start_workflow(
            session_id=session_id,
            request_id=REQUEST,
            idempotency_key="obs-idem-002",
            request_text="Please refer me to cardiology.",
        )
        team.resume_workflow(
            session_id=session_id,
            request_id=REQUEST,
            idempotency_key="obs-idem-002",
            reviewer_id=reviewer,
            decision="approve",
        )
    finally:
        team.close()

    names = {
        event["span"]["name"]
        for event in client.traces
        if "span" in event
    }
    assert {
        "clinical_safety",
        "intake",
        "eligibility_lookup",
        "human_approval",
        "human_approval_resume",
        "referral_handling",
    }.issubset(names)
    assert all(
        event["trace"]["metadata"]["request_id"] == REQUEST
        for event in client.traces
    )
