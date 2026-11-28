import json
import sqlite3

import pytest
from pydantic import ValidationError

from careflow.memory import (
    AdministrativeFact,
    ContextBudgetExceededError,
    IdempotencyConflictError,
    MemoryOwnershipError,
    MemorySessionNotFoundError,
    SQLiteConversationMemory,
    SummarizationError,
    UnsupportedMemberIdentityError,
)


MEMBER_ONE = "SYN-MEMBER-001"
MEMBER_TWO = "SYN-MEMBER-002"


def deterministic_summarizer(previous_summary, older_messages):
    older_text = " ".join(
        message.content or ""
        for message in older_messages
    )
    return " ".join(part for part in (previous_summary, older_text) if part).strip()


def test_messages_persist_across_memory_store_restarts(tmp_path):
    database_path = tmp_path / "synthetic-memory.sqlite3"
    first_store = SQLiteConversationMemory(database_path, deterministic_summarizer)
    session_id = first_store.create_session(MEMBER_ONE)
    first_store.append_message(
        session_id,
        MEMBER_ONE,
        {"role": "user", "content": "Check synthetic referral SYN-REF-004."},
    )
    first_store.append_message(
        session_id,
        MEMBER_ONE,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_synthetic_1",
                    "type": "function",
                    "function": {
                        "name": "check_insurance_eligibility",
                        "arguments": '{"member_id":"SYN-MEMBER-001"}',
                    },
                }
            ],
        },
    )
    first_store.append_message(
        session_id,
        MEMBER_ONE,
        {
            "role": "tool",
            "tool_call_id": "call_synthetic_1",
            "content": '{"status":"eligible"}',
        },
    )

    restarted_store = SQLiteConversationMemory(
        database_path, deterministic_summarizer
    )
    history = restarted_store.load_history(session_id, MEMBER_ONE)

    assert [message.role for message in history] == [
        "user",
        "assistant",
        "tool",
    ]
    assert history[1].tool_calls[0].id == "call_synthetic_1"
    assert history[2].tool_call_id == "call_synthetic_1"
    assert [message.content for message in history if message.role != "assistant"] == [
        "Check synthetic referral SYN-REF-004.",
        '{"status":"eligible"}',
    ]
    assert all(message.member_id == MEMBER_ONE for message in history)


def test_history_isolated_by_session_owner_and_untrusted_claims_do_not_change_owner(
    tmp_path,
):
    memory = SQLiteConversationMemory(tmp_path / "memory.sqlite3", deterministic_summarizer)
    session_id = memory.create_session(MEMBER_ONE)
    assert session_id.startswith("SYN-SESSION-")
    memory.append_message(
        session_id,
        MEMBER_ONE,
        {
            "role": "user",
            "content": "I claim to be SYN-MEMBER-002; still store under this session owner.",
        },
    )

    with pytest.raises(MemoryOwnershipError):
        memory.load_history(session_id, MEMBER_TWO)
    with pytest.raises(UnsupportedMemberIdentityError):
        memory.create_session("SYN-MEMBER-999")
    with pytest.raises(ValidationError):
        memory.append_message(
            session_id,
            MEMBER_ONE,
            {
                "role": "user",
                "content": "A message cannot override its trusted owner.",
                "member_id": MEMBER_TWO,
            },
        )

    history = memory.load_history(session_id, MEMBER_ONE)
    assert len(history) == 1
    assert history[0].member_id == MEMBER_ONE


def test_summary_keeps_original_messages_recent_messages_and_facts_separate(tmp_path):
    summarizer_calls = []

    def tracking_summarizer(previous_summary, older_messages):
        summarizer_calls.append(
            (previous_summary, tuple(message.content for message in older_messages))
        )
        return deterministic_summarizer(previous_summary, older_messages)

    memory = SQLiteConversationMemory(
        tmp_path / "summary.sqlite3",
        tracking_summarizer,
        recent_message_limit=2,
        token_counter=lambda value: len(value),
    )
    session_id = memory.create_session(MEMBER_ONE)
    memory.store_fact(
        session_id,
        MEMBER_ONE,
        {"fact_type": "referral_id", "identifier": "SYN-REF-004"},
    )
    original_messages = ("turn one", "turn two", "turn three", "turn four")
    for content in original_messages:
        memory.append_message(session_id, MEMBER_ONE, {"role": "user", "content": content})

    summary = memory.load_summary(session_id, MEMBER_ONE)
    history = memory.load_history(session_id, MEMBER_ONE)
    context = memory.load_context(
        session_id, MEMBER_ONE, "What is the referral status?"
    )

    assert summary == "turn one turn two"
    assert summarizer_calls == [
        ("", ("turn one",)),
        ("turn one", ("turn two",)),
    ]
    assert [message.content for message in history] == list(original_messages)
    assert [message.content for message in context.recent_messages] == [
        "turn three",
        "turn four",
    ]
    assert context.summary == summary
    assert [fact.identifier for fact in context.facts] == ["SYN-REF-004"]
    assert [fact.identifier for fact in memory.load_facts(session_id, MEMBER_ONE)] == [
        "SYN-REF-004"
    ]


def test_missing_session_and_invalid_synthetic_fact_are_rejected(tmp_path):
    memory = SQLiteConversationMemory(tmp_path / "missing.sqlite3", deterministic_summarizer)

    with pytest.raises(MemorySessionNotFoundError):
        memory.load_history("SYN-SESSION-MISSING", MEMBER_ONE)
    with pytest.raises(ValidationError):
        AdministrativeFact(
            fact_type="referral_id",
            identifier="REAL-REF-004",
        )


def test_batch_append_validates_entire_turn_before_persisting(tmp_path):
    memory = SQLiteConversationMemory(
        tmp_path / "atomic-append.sqlite3", deterministic_summarizer
    )
    session_id = memory.create_session(MEMBER_ONE)

    with pytest.raises(ValidationError):
        memory.append_messages(
            session_id,
            MEMBER_ONE,
            (
                {"role": "user", "content": "Valid first message."},
                {"role": "assistant", "content": None},
            ),
        )

    assert memory.load_history(session_id, MEMBER_ONE) == ()


def test_completed_turn_is_idempotent_and_key_is_scoped_to_session(tmp_path):
    memory = SQLiteConversationMemory(
        tmp_path / "completed-turns.sqlite3", deterministic_summarizer
    )
    first_session = memory.create_session(MEMBER_ONE)
    second_session = memory.create_session(MEMBER_ONE)
    turn_messages = (
        {"role": "user", "content": "Check synthetic eligibility."},
        {"role": "assistant", "content": "Eligibility is synthetically confirmed."},
    )

    completed = memory.complete_turn(
        first_session,
        MEMBER_ONE,
        "request-key-001",
        "Check synthetic eligibility.",
        "Eligibility is synthetically confirmed.",
        turn_messages,
    )
    duplicate = memory.complete_turn(
        first_session,
        MEMBER_ONE,
        "request-key-001",
        "Check synthetic eligibility.",
        "A different generated result is ignored for a completed key.",
        turn_messages[:1]
        + (
            {
                "role": "assistant",
                "content": "A different generated result is ignored for a completed key.",
            },
        ),
    )
    other_session_turn = memory.complete_turn(
        second_session,
        MEMBER_ONE,
        "request-key-001",
        "A distinct request in another session.",
        "A distinct synthetic response.",
        (
            {"role": "user", "content": "A distinct request in another session."},
            {"role": "assistant", "content": "A distinct synthetic response."},
        ),
    )

    assert duplicate.response == completed.response
    assert len(memory.load_history(first_session, MEMBER_ONE)) == 2
    assert other_session_turn.response == "A distinct synthetic response."
    with pytest.raises(IdempotencyConflictError):
        memory.lookup_completed_turn(
            first_session,
            MEMBER_ONE,
            "request-key-001",
            "A different request with the same key.",
        )


def test_invalid_completed_turn_is_not_partially_committed(tmp_path):
    memory = SQLiteConversationMemory(
        tmp_path / "invalid-completed-turn.sqlite3", deterministic_summarizer
    )
    session_id = memory.create_session(MEMBER_ONE)

    with pytest.raises(ValidationError):
        memory.complete_turn(
            session_id,
            MEMBER_ONE,
            "request-key-invalid",
            "A synthetic request.",
            "A synthetic response.",
            (
                {"role": "user", "content": "A synthetic request."},
                {"role": "assistant", "content": None},
            ),
        )

    assert memory.lookup_completed_turn(
        session_id,
        MEMBER_ONE,
        "request-key-invalid",
        "A synthetic request.",
    ) is None
    assert memory.load_history(session_id, MEMBER_ONE) == ()


def test_completed_turn_transaction_rolls_back_on_partial_sqlite_failure(tmp_path):
    database_path = tmp_path / "completed-turn-rollback.sqlite3"
    memory = SQLiteConversationMemory(database_path, deterministic_summarizer)
    session_id = memory.create_session(MEMBER_ONE)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_assistant_message
            BEFORE INSERT ON messages
            WHEN NEW.role = 'assistant'
            BEGIN
                SELECT RAISE(ABORT, 'synthetic write failure');
            END
            """
        )

    messages = (
        {"role": "user", "content": "A synthetic request."},
        {"role": "assistant", "content": "A synthetic response."},
    )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic write failure"):
        memory.complete_turn(
            session_id,
            MEMBER_ONE,
            "retry-after-write-failure",
            "A synthetic request.",
            "A synthetic response.",
            messages,
        )

    assert memory.lookup_completed_turn(
        session_id,
        MEMBER_ONE,
        "retry-after-write-failure",
        "A synthetic request.",
    ) is None
    assert memory.load_history(session_id, MEMBER_ONE) == ()

    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER reject_assistant_message")
    completed = memory.complete_turn(
        session_id,
        MEMBER_ONE,
        "retry-after-write-failure",
        "A synthetic request.",
        "A synthetic response.",
        messages,
    )
    assert completed.response == "A synthetic response."
    assert len(memory.load_history(session_id, MEMBER_ONE)) == 2


def test_summarizer_failure_keeps_original_history_and_existing_summary(tmp_path):
    def failing_summarizer(_previous_summary, _older_messages):
        raise RuntimeError("synthetic summarizer failure")

    database_path = tmp_path / "summary-failure.sqlite3"
    memory = SQLiteConversationMemory(
        database_path,
        failing_summarizer,
        recent_message_limit=2,
        token_counter=lambda value: len(value),
    )
    session_id = memory.create_session(MEMBER_ONE)
    memory.append_message(session_id, MEMBER_ONE, {"role": "user", "content": "turn one"})
    memory.append_message(session_id, MEMBER_ONE, {"role": "user", "content": "turn two"})

    with pytest.raises(SummarizationError, match="original messages remain stored"):
        memory.append_message(
            session_id, MEMBER_ONE, {"role": "user", "content": "turn three"}
        )

    restarted_store = SQLiteConversationMemory(
        database_path,
        deterministic_summarizer,
        recent_message_limit=2,
        token_counter=lambda value: len(value),
    )
    assert restarted_store.load_summary(session_id, MEMBER_ONE) == ""
    assert [message.content for message in restarted_store.load_history(session_id, MEMBER_ONE)] == [
        "turn one",
        "turn two",
        "turn three",
    ]


def test_context_budget_uses_tokenizer_and_bounds_rendered_prompt(tmp_path):
    memory = SQLiteConversationMemory(
        tmp_path / "bounded.sqlite3",
        lambda _summary, _messages: "old history " * 200,
        recent_message_limit=4,
        prompt_token_budget=220,
        token_counter=lambda value: len(value),
    )
    session_id = memory.create_session(MEMBER_ONE)
    memory.store_fact(
        session_id,
        MEMBER_ONE,
        {"fact_type": "referral_id", "identifier": "SYN-REF-004"},
    )
    for index in range(8):
        memory.append_message(
            session_id,
            MEMBER_ONE,
            {"role": "user", "content": f"synthetic message {index}"},
        )

    context = memory.load_context(
        session_id,
        MEMBER_ONE,
        "Please check this synthetic referral.",
    )

    assert context.token_count == len(context.render())
    assert context.token_count <= context.token_budget
    assert context.summary_truncated is True
    assert context.omitted_message_count > 0
    assert json.loads(context.render())["current_message"] == (
        "Please check this synthetic referral."
    )


def test_oversized_current_message_is_rejected_by_context_budget(tmp_path):
    memory = SQLiteConversationMemory(
        tmp_path / "small-budget.sqlite3",
        deterministic_summarizer,
        prompt_token_budget=40,
        token_counter=lambda value: len(value),
    )
    session_id = memory.create_session(MEMBER_ONE)

    with pytest.raises(ContextBudgetExceededError):
        memory.load_context(session_id, MEMBER_ONE, "synthetic " * 100)


def test_default_token_counter_uses_tiktoken(tmp_path):
    import tiktoken

    memory = SQLiteConversationMemory(tmp_path / "tokenizer.sqlite3", deterministic_summarizer)
    session_id = memory.create_session(MEMBER_ONE)

    context = memory.load_context(session_id, MEMBER_ONE, "A synthetic request.")

    encoding = tiktoken.get_encoding("cl100k_base")
    assert context.token_count == len(
        encoding.encode(context.render(), disallowed_special=())
    )
