import json
from types import SimpleNamespace

import pytest

from careflow.llm_agent import (
    AgentAPIError,
    LiteLLMSchedulingInsuranceAgent,
    POLICY_EVIDENCE_UNAVAILABLE_RESPONSE,
)
from careflow.memory import (
    MemoryOwnershipError,
    SQLiteConversationMemory,
)
from careflow.policy_index import LocalPolicyIndex


MEMBER_ONE = "SYN-MEMBER-001"
MEMBER_TWO = "SYN-MEMBER-002"
EMBEDDING_MODEL = "careflow-m3-demo-fake-v1"
EMBEDDING_TERMS = (
    "appointment",
    "referral",
    "insurance",
    "copay",
    "eligibility",
    "specialist",
    "scheduling",
    "status",
)


class DeterministicFakeEmbedder:
    """Tiny lexical vectors for integration mechanics, not semantic evaluation."""

    def __call__(self, texts):
        return [
            [
                float(text.lower().count(term))
                for term in EMBEDDING_TERMS
            ]
            for text in texts
        ]


class DeterministicFakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **request):
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def model_response(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def appointment_tool_request():
    return model_response(
        tool_calls=[
            SimpleNamespace(
                id="m3-demo-appointment-call",
                type="function",
                function=SimpleNamespace(
                    name="lookup_appointments",
                    arguments='{"specialty":"cardiology"}',
                ),
            )
        ]
    )


def create_memory(path, *, summarizer=None):
    if summarizer is None:
        def summarize(previous, messages):
            older_text = " ".join(
                message.content
                for message in messages
                if message.content is not None
            )
            return " ".join(part for part in (previous, older_text) if part).strip()

        summarizer = summarize
    return SQLiteConversationMemory(
        path,
        summarizer,
        recent_message_limit=6,
        prompt_token_budget=12_000,
        token_counter=lambda value: len(value),
    )


def create_policy_index(path):
    return LocalPolicyIndex(
        DeterministicFakeEmbedder(),
        embedding_model_id=EMBEDDING_MODEL,
        embedding_dimension=len(EMBEDDING_TERMS),
        persist_directory=path,
    )


def test_synthetic_m3_demo_persistence_isolation_policy_and_tools(tmp_path):
    database_path = tmp_path / "synthetic-conversations.sqlite3"
    vector_path = tmp_path / "synthetic-policy-vectors"
    memory = create_memory(database_path)
    index = create_policy_index(vector_path)
    index.index_documents()

    member_one_session = memory.create_session(MEMBER_ONE)
    member_two_session = memory.create_session(MEMBER_TWO)
    memory.store_fact(
        member_two_session,
        MEMBER_TWO,
        {"fact_type": "referral_id", "identifier": "SYN-REF-222"},
    )

    first_llm = DeterministicFakeLLM(
        appointment_tool_request(),
        model_response("Two fictional cardiology appointments are available."),
    )
    first_agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=first_llm,
        memory=memory,
        policy_index=index,
    )
    first_reply = first_agent.respond(
        "Find available synthetic cardiology appointments.",
        trusted_member_id=MEMBER_ONE,
        session_id=member_one_session,
        idempotency_key="demo-appointment-turn-001",
    )

    assert first_reply == "Two fictional cardiology appointments are available."
    first_history = memory.load_history(member_one_session, MEMBER_ONE)
    assert [message.role for message in first_history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert first_history[1].tool_calls[0].id == "m3-demo-appointment-call"
    assert first_history[2].tool_call_id == "m3-demo-appointment-call"

    index.close()
    del first_agent
    reopened_memory = create_memory(database_path)
    reopened_index = create_policy_index(vector_path)
    continuation_llm = DeterministicFakeLLM(
        model_response("This synthetic session has been restored.")
    )
    reopened_agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=continuation_llm,
        memory=reopened_memory,
        policy_index=reopened_index,
    )
    reopened_agent.respond(
        "Continue the earlier appointment conversation.",
        trusted_member_id=MEMBER_ONE,
        session_id=member_one_session,
        idempotency_key="demo-continuation-turn-001",
    )

    context_text = next(
        message["content"]
        for message in continuation_llm.calls[0]["messages"]
        if message["role"] == "user"
        and "current_message" in message["content"]
    )
    restored_context = json.loads(context_text)
    assert any(
        "Two fictional cardiology appointments are available."
        in (message.get("content") or "")
        for message in restored_context["recent_messages"]
    )
    assert "SYN-REF-222" not in context_text
    with pytest.raises(MemoryOwnershipError):
        reopened_memory.load_history(member_two_session, MEMBER_ONE)
    with pytest.raises(MemoryOwnershipError):
        reopened_memory.load_facts(member_two_session, MEMBER_ONE)

    policy_reply = reopened_agent.respond(
        "What is the fictional specialist copay for SYN-PLAN-12?",
        trusted_member_id=MEMBER_ONE,
        session_id=member_one_session,
        idempotency_key="demo-policy-turn-001",
    )
    assert "[SYN-POLICY-INS-001 | Synthetic Specialist Copays]" in policy_reply
    assert "35 dollar specialist visit copay" in policy_reply

    reopened_index.close()


def test_synthetic_m3_demo_low_confidence_policy_fails_closed(tmp_path):
    index = create_policy_index(tmp_path / "low-confidence-vectors")
    index.index_documents()
    completion = DeterministicFakeLLM()
    agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=completion,
        policy_index=index,
        policy_minimum_score=0.35,
    )

    response = agent.respond(
        "What does the fictional handbook say about quantum teleportation coverage?"
    )

    assert response == POLICY_EVIDENCE_UNAVAILABLE_RESPONSE
    assert completion.calls == []
    index.close()


def test_synthetic_m3_demo_failed_llm_call_leaves_history_unchanged(tmp_path):
    memory = create_memory(tmp_path / "failed-turn.sqlite3")
    session_id = memory.create_session(MEMBER_ONE)
    memory.append_messages(
        session_id,
        MEMBER_ONE,
        (
            {"role": "user", "content": "Previously completed synthetic turn."},
            {"role": "assistant", "content": "Previously persisted response."},
        ),
    )
    history_before = memory.load_history(session_id, MEMBER_ONE)
    completion = DeterministicFakeLLM(
        appointment_tool_request(),
        RuntimeError("synthetic model service unavailable"),
    )
    agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=completion,
        memory=memory,
    )

    with pytest.raises(AgentAPIError):
        agent.respond(
            "Find synthetic cardiology appointments.",
            trusted_member_id=MEMBER_ONE,
            session_id=session_id,
            idempotency_key="failed-demo-turn-001",
        )

    assert memory.load_history(session_id, MEMBER_ONE) == history_before
