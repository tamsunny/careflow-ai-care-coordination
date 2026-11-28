"""SQLite-backed synthetic conversation memory for Milestone 3, Phase 1."""

import json
import hashlib
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mock_data import MOCK_MEMBERS

FactType = Literal["referral_id", "appointment_id", "insurance_plan_id"]
MessageRole = Literal["user", "assistant", "tool"]


class ConversationToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    arguments: str = Field(max_length=8192)


class ConversationToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    type: Literal["function"]
    function: ConversationToolFunction


class ConversationMessage(BaseModel):
    """A stored non-system conversation message, including LiteLLM tool metadata."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str | None = Field(default=None, max_length=100_000)
    tool_call_id: str | None = Field(default=None, max_length=200)
    tool_calls: list[ConversationToolCall] | None = Field(default=None, max_length=8)

    @model_validator(mode="after")
    def validate_role_fields(self) -> "ConversationMessage":
        if self.role == "user" and self.content is None:
            raise ValueError("User messages require content")
        if self.role == "tool" and (self.content is None or self.tool_call_id is None):
            raise ValueError("Tool messages require content and a tool-call ID")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("Only assistant messages may contain tool calls")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("Only tool messages may contain a tool-call ID")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("Assistant messages require content or tool calls")
        return self


class StoredMessage(ConversationMessage):
    message_id: int
    session_id: str
    member_id: str
    created_at: datetime

    def to_conversation_message(self) -> ConversationMessage:
        return ConversationMessage.model_validate(
            self.model_dump(
                exclude={"message_id", "session_id", "member_id", "created_at"}
            )
        )


class AdministrativeFact(BaseModel):
    """A synthetic administrative identifier stored independently of summaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fact_type: FactType
    identifier: str = Field(min_length=8, max_length=100)
    note: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def require_matching_synthetic_identifier(self) -> "AdministrativeFact":
        prefix_by_type = {
            "referral_id": "SYN-REF-",
            "appointment_id": "SYN-APPT-",
            "insurance_plan_id": "SYN-PLAN-",
        }
        prefix = prefix_by_type[self.fact_type]
        if not re.fullmatch(re.escape(prefix) + r"[A-Z0-9-]+", self.identifier):
            raise ValueError("Identifier must match its synthetic fact type")
        return self


class StoredFact(AdministrativeFact):
    fact_id: int
    session_id: str
    member_id: str
    stored_at: datetime


class CompletedTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    member_id: str
    idempotency_key: str
    response: str
    completed_at: datetime


class MemoryContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    facts: tuple[StoredFact, ...]
    recent_messages: tuple[StoredMessage, ...]
    current_message: str
    token_count: int
    token_budget: int
    summary_truncated: bool
    omitted_fact_count: int
    omitted_message_count: int

    def render(self) -> str:
        """Render the exact bounded context payload counted by the tokenizer."""
        return _render_context(
            self.summary,
            self.facts,
            self.recent_messages,
            self.current_message,
        )


SummaryFunction = Callable[
    [str, tuple[ConversationMessage, ...]],
    str,
]
TokenCounter = Callable[[str], int]


class MemoryErrorBase(RuntimeError):
    """Base error for the synthetic memory layer."""


class UnsupportedMemberIdentityError(MemoryErrorBase):
    """Member identity is not one of the known synthetic fixture identities."""


class MemorySessionNotFoundError(MemoryErrorBase):
    """The requested synthetic session does not exist."""


class MemoryOwnershipError(MemoryErrorBase):
    """The supplied trusted member identity does not own the session."""


class IdempotencyConflictError(MemoryErrorBase):
    """A completed-turn key was reused for a different request."""


class SummaryRefreshError(MemoryErrorBase):
    """Rolling-summary refresh failed and may be retried separately."""


class SummarizationError(SummaryRefreshError):
    """Summary generation failed; original messages remain intact."""


class SummarizationConflictError(SummaryRefreshError):
    """Another writer changed the summary while it was being generated."""


class ContextBudgetExceededError(MemoryErrorBase):
    """The current message alone exceeds the configured context budget."""


class MemoryConfigurationError(MemoryErrorBase):
    """Memory configuration or tokenizer setup is invalid."""


class _TiktokenCounter:
    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise MemoryConfigurationError(
                "Install the declared tiktoken dependency to count prompt tokens"
            ) from exc
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def __call__(self, value: str) -> int:
        return len(self._encoding.encode(value, disallowed_special=()))


class SQLiteConversationMemory:
    """Persist synthetic messages, summaries, and administrative facts in SQLite.

    ``trusted_member_id`` must come from the hosting application's verified
    context. It is never inferred from message content or an LLM response.
    """

    def __init__(
        self,
        database_path: str | Path,
        summarizer: SummaryFunction,
        *,
        recent_message_limit: int = 6,
        prompt_token_budget: int = 2048,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if not callable(summarizer):
            raise MemoryConfigurationError("A summarizer function is required")
        if (
            isinstance(recent_message_limit, bool)
            or not isinstance(recent_message_limit, int)
            or recent_message_limit < 1
        ):
            raise MemoryConfigurationError(
                "recent_message_limit must be a positive integer"
            )
        if (
            isinstance(prompt_token_budget, bool)
            or not isinstance(prompt_token_budget, int)
            or prompt_token_budget < 1
        ):
            raise MemoryConfigurationError(
                "prompt_token_budget must be a positive integer"
            )

        self.database_path = Path(database_path)
        if str(database_path) == ":memory:":
            raise MemoryConfigurationError(
                "Use a SQLite file path so memory can persist across restarts"
            )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._summarizer = summarizer
        self.recent_message_limit = recent_message_limit
        self.prompt_token_budget = prompt_token_budget
        self._token_counter = (
            _TiktokenCounter() if token_counter is None else token_counter
        )
        self._initialize()

    def create_session(self, trusted_member_id: str) -> str:
        self._validate_member_identity(trusted_member_id)
        session_id = f"SYN-SESSION-{uuid4().hex}"
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (session_id, member_id, created_at) "
                "VALUES (?, ?, ?)",
                (session_id, trusted_member_id, created_at),
            )
            connection.execute(
                "INSERT INTO summaries "
                "(session_id, member_id, summary, summarized_through_message_id, updated_at) "
                "VALUES (?, ?, '', 0, ?)",
                (session_id, trusted_member_id, created_at),
            )
        return session_id

    def append_message(
        self,
        session_id: str,
        trusted_member_id: str,
        message: ConversationMessage | dict[str, Any],
    ) -> StoredMessage:
        return self.append_messages(
            session_id,
            trusted_member_id,
            (message,),
        )[0]

    def append_messages(
        self,
        session_id: str,
        trusted_member_id: str,
        messages: Sequence[ConversationMessage | dict[str, Any]],
    ) -> tuple[StoredMessage, ...]:
        """Store a complete turn atomically before refreshing its rolling summary."""
        self._validate_member_identity(trusted_member_id)
        conversation_messages = tuple(
            ConversationMessage.model_validate(message) for message in messages
        )
        if not conversation_messages:
            raise MemoryConfigurationError("At least one message is required")

        stored_messages = []
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            for conversation_message in conversation_messages:
                created_at = _utc_now()
                cursor = connection.execute(
                    "INSERT INTO messages "
                    "(session_id, member_id, role, message_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        trusted_member_id,
                        conversation_message.role,
                        conversation_message.model_dump_json(exclude_none=True),
                        created_at,
                    ),
                )
                stored_messages.append(
                    _stored_message(
                        conversation_message,
                        cursor.lastrowid,
                        session_id,
                        trusted_member_id,
                        created_at,
                    )
                )

        self._refresh_summary(session_id, trusted_member_id)
        return tuple(stored_messages)

    def lookup_completed_turn(
        self,
        session_id: str,
        trusted_member_id: str,
        idempotency_key: str,
        request_message: str,
    ) -> CompletedTurn | None:
        """Return a prior result, rejecting key reuse for different request text."""
        self._validate_member_identity(trusted_member_id)
        key = _validate_idempotency_key(idempotency_key)
        request_hash = _request_hash(request_message)
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            row = connection.execute(
                "SELECT session_id, member_id, idempotency_key, request_hash, "
                "response, completed_at FROM completed_turns "
                "WHERE session_id = ? AND member_id = ? AND idempotency_key = ?",
                (session_id, trusted_member_id, key),
            ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise IdempotencyConflictError(
                "Idempotency key is already associated with a different request"
            )
        return _completed_turn_from_row(row)

    def complete_turn(
        self,
        session_id: str,
        trusted_member_id: str,
        idempotency_key: str,
        request_message: str,
        response: str,
        messages: Sequence[ConversationMessage | dict[str, Any]],
    ) -> CompletedTurn:
        """Atomically store one completed turn, its response, and its messages."""
        self._validate_member_identity(trusted_member_id)
        key = _validate_idempotency_key(idempotency_key)
        request_hash = _request_hash(request_message)
        if not isinstance(response, str) or not response.strip():
            raise MemoryConfigurationError("A completed turn requires a response")
        conversation_messages = tuple(
            ConversationMessage.model_validate(message) for message in messages
        )
        if (
            not conversation_messages
            or conversation_messages[0].role != "user"
            or conversation_messages[0].content != request_message
            or conversation_messages[-1].role != "assistant"
            or conversation_messages[-1].content != response
        ):
            raise MemoryConfigurationError(
                "Turn messages must begin with the request and end with its response"
            )

        completed_at = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_session_owner(connection, session_id, trusted_member_id)
            existing = connection.execute(
                "SELECT session_id, member_id, idempotency_key, request_hash, "
                "response, completed_at FROM completed_turns "
                "WHERE session_id = ? AND member_id = ? AND idempotency_key = ?",
                (session_id, trusted_member_id, key),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key is already associated with a different request"
                    )
                return _completed_turn_from_row(existing)

            connection.execute(
                "INSERT INTO completed_turns "
                "(session_id, member_id, idempotency_key, request_hash, response, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    trusted_member_id,
                    key,
                    request_hash,
                    response,
                    completed_at,
                ),
            )
            for conversation_message in conversation_messages:
                connection.execute(
                    "INSERT INTO messages "
                    "(session_id, member_id, role, message_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        trusted_member_id,
                        conversation_message.role,
                        conversation_message.model_dump_json(exclude_none=True),
                        completed_at,
                    ),
                )
        return CompletedTurn(
            session_id=session_id,
            member_id=trusted_member_id,
            idempotency_key=key,
            response=response,
            completed_at=completed_at,
        )

    def load_history(
        self, session_id: str, trusted_member_id: str
    ) -> tuple[StoredMessage, ...]:
        self._validate_member_identity(trusted_member_id)
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            rows = connection.execute(
                "SELECT message_id, session_id, member_id, message_json, created_at "
                "FROM messages WHERE session_id = ? AND member_id = ? "
                "ORDER BY message_id",
                (session_id, trusted_member_id),
            ).fetchall()
        return tuple(_stored_message_from_row(row) for row in rows)

    def store_fact(
        self,
        session_id: str,
        trusted_member_id: str,
        fact: AdministrativeFact | dict[str, Any],
    ) -> StoredFact:
        self._validate_member_identity(trusted_member_id)
        validated_fact = AdministrativeFact.model_validate(fact)
        stored_at = _utc_now()
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            connection.execute(
                "INSERT OR IGNORE INTO administrative_facts "
                "(session_id, member_id, fact_type, identifier, note, stored_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    trusted_member_id,
                    validated_fact.fact_type,
                    validated_fact.identifier,
                    validated_fact.note,
                    stored_at,
                ),
            )
            row = connection.execute(
                "SELECT fact_id, session_id, member_id, fact_type, identifier, "
                "note, stored_at FROM administrative_facts "
                "WHERE session_id = ? AND member_id = ? AND fact_type = ? "
                "AND identifier = ?",
                (
                    session_id,
                    trusted_member_id,
                    validated_fact.fact_type,
                    validated_fact.identifier,
                ),
            ).fetchone()
        return _stored_fact_from_row(row)

    def load_facts(
        self, session_id: str, trusted_member_id: str
    ) -> tuple[StoredFact, ...]:
        self._validate_member_identity(trusted_member_id)
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            rows = connection.execute(
                "SELECT fact_id, session_id, member_id, fact_type, identifier, "
                "note, stored_at FROM administrative_facts "
                "WHERE session_id = ? AND member_id = ? "
                "ORDER BY fact_id",
                (session_id, trusted_member_id),
            ).fetchall()
        return tuple(_stored_fact_from_row(row) for row in rows)

    def load_summary(self, session_id: str, trusted_member_id: str) -> str:
        self._validate_member_identity(trusted_member_id)
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            row = connection.execute(
                "SELECT summary FROM summaries WHERE session_id = ? AND member_id = ?",
                (session_id, trusted_member_id),
            ).fetchone()
        if row is None:
            raise MemorySessionNotFoundError("Synthetic session summary was not found")
        return row["summary"]

    def refresh_summary(self, session_id: str, trusted_member_id: str) -> None:
        """Retry rolling-summary maintenance independently of completed turns."""
        self._validate_member_identity(trusted_member_id)
        try:
            self._refresh_summary(session_id, trusted_member_id)
        except SummaryRefreshError:
            raise
        except sqlite3.Error as exc:
            raise SummaryRefreshError(
                "Rolling summary could not be refreshed; retry the refresh separately"
            ) from exc

    def load_context(
        self,
        session_id: str,
        trusted_member_id: str,
        current_message: str,
        *,
        token_budget: int | None = None,
    ) -> MemoryContext:
        self._validate_member_identity(trusted_member_id)
        self._refresh_summary(session_id, trusted_member_id)
        budget = self.prompt_token_budget if token_budget is None else token_budget
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise MemoryConfigurationError("token_budget must be a positive integer")

        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            summary_row = connection.execute(
                "SELECT summary, summarized_through_message_id FROM summaries "
                "WHERE session_id = ? AND member_id = ?",
                (session_id, trusted_member_id),
            ).fetchone()
            if summary_row is None:
                raise MemorySessionNotFoundError(
                    "Synthetic session summary was not found"
                )
            fact_rows = connection.execute(
                "SELECT fact_id, session_id, member_id, fact_type, identifier, "
                "note, stored_at FROM administrative_facts "
                "WHERE session_id = ? AND member_id = ? ORDER BY fact_id",
                (session_id, trusted_member_id),
            ).fetchall()
            message_rows = connection.execute(
                "SELECT message_id, session_id, member_id, message_json, created_at "
                "FROM messages WHERE session_id = ? AND member_id = ? "
                "AND message_id > ? ORDER BY message_id",
                (
                    session_id,
                    trusted_member_id,
                    summary_row["summarized_through_message_id"],
                ),
            ).fetchall()

        summary = summary_row["summary"]
        all_facts = tuple(_stored_fact_from_row(row) for row in fact_rows)
        unsummarized_messages = tuple(
            _stored_message_from_row(row) for row in message_rows
        )
        base_payload = _render_context("", (), (), current_message)
        if self._count_tokens(base_payload) > budget:
            raise ContextBudgetExceededError(
                "Current message exceeds the configured prompt context budget"
            )

        selected_facts_newest_first: list[StoredFact] = []
        omitted_fact_count = 0
        for index, fact in enumerate(reversed(all_facts)):
            trial_facts = tuple(reversed([*selected_facts_newest_first, fact]))
            if self._fits_context(
                "", trial_facts, (), current_message, budget
            ):
                selected_facts_newest_first.append(fact)
            else:
                omitted_fact_count = len(all_facts) - index
                break

        recent_candidates = unsummarized_messages[-self.recent_message_limit :]
        omitted_message_count = len(unsummarized_messages) - len(recent_candidates)
        selected_messages_newest_first: list[StoredMessage] = []
        selected_facts = tuple(reversed(selected_facts_newest_first))
        for index, message in enumerate(reversed(recent_candidates)):
            trial_messages = tuple(reversed([*selected_messages_newest_first, message]))
            if self._fits_context(
                "", selected_facts, trial_messages, current_message, budget
            ):
                selected_messages_newest_first.append(message)
            else:
                omitted_message_count += len(recent_candidates) - index
                break

        selected_messages = tuple(reversed(selected_messages_newest_first))
        fitted_summary = summary
        summary_truncated = False
        if not self._fits_context(
            fitted_summary, selected_facts, selected_messages, current_message, budget
        ):
            fitted_summary = self._fit_summary_suffix(
                summary, selected_facts, selected_messages, current_message, budget
            )
            summary_truncated = fitted_summary != summary

        rendered = _render_context(
            fitted_summary, selected_facts, selected_messages, current_message
        )
        token_count = self._count_tokens(rendered)
        if token_count > budget:
            raise MemoryConfigurationError(
                "Internal prompt context exceeded its configured token budget"
            )
        return MemoryContext(
            summary=fitted_summary,
            facts=selected_facts,
            recent_messages=selected_messages,
            current_message=current_message,
            token_count=token_count,
            token_budget=budget,
            summary_truncated=summary_truncated,
            omitted_fact_count=omitted_fact_count,
            omitted_message_count=omitted_message_count,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS synthetic_members (
                    member_id TEXT PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, member_id),
                    FOREIGN KEY (member_id)
                        REFERENCES synthetic_members (member_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id, member_id)
                        REFERENCES sessions (session_id, member_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS messages_by_session
                    ON messages (session_id, member_id, message_id);
                CREATE TABLE IF NOT EXISTS completed_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    UNIQUE (session_id, member_id, idempotency_key),
                    FOREIGN KEY (session_id, member_id)
                        REFERENCES sessions (session_id, member_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    summarized_through_message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id, member_id)
                        REFERENCES sessions (session_id, member_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS administrative_facts (
                    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    identifier TEXT NOT NULL,
                    note TEXT,
                    stored_at TEXT NOT NULL,
                    UNIQUE (session_id, member_id, fact_type, identifier),
                    FOREIGN KEY (session_id, member_id)
                        REFERENCES sessions (session_id, member_id)
                        ON DELETE CASCADE
                );
                """
            )
            connection.executemany(
                "INSERT OR IGNORE INTO synthetic_members (member_id) VALUES (?)",
                ((member_id,) for member_id in MOCK_MEMBERS),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _validate_member_identity(self, trusted_member_id: str) -> None:
        if (
            not isinstance(trusted_member_id, str)
            or trusted_member_id not in MOCK_MEMBERS
        ):
            raise UnsupportedMemberIdentityError(
                "A supported trusted synthetic member identity is required"
            )

    def _assert_session_owner(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        trusted_member_id: str,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise MemorySessionNotFoundError("Synthetic session was not found")
        row = connection.execute(
            "SELECT member_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise MemorySessionNotFoundError("Synthetic session was not found")
        if row["member_id"] != trusted_member_id:
            raise MemoryOwnershipError(
                "Trusted member identity does not own this synthetic session"
            )

    def _refresh_summary(self, session_id: str, trusted_member_id: str) -> None:
        with self._connect() as connection:
            self._assert_session_owner(connection, session_id, trusted_member_id)
            summary_row = connection.execute(
                "SELECT summary, summarized_through_message_id FROM summaries "
                "WHERE session_id = ? AND member_id = ?",
                (session_id, trusted_member_id),
            ).fetchone()
            if summary_row is None:
                raise MemorySessionNotFoundError(
                    "Synthetic session summary was not found"
                )
            cutoff_row = connection.execute(
                "SELECT message_id FROM messages WHERE session_id = ? AND member_id = ? "
                "ORDER BY message_id DESC LIMIT 1 OFFSET ?",
                (
                    session_id,
                    trusted_member_id,
                    self.recent_message_limit,
                ),
            ).fetchone()
            if cutoff_row is None:
                return
            cutoff_id = cutoff_row["message_id"]
            older_rows = connection.execute(
                "SELECT message_id, session_id, member_id, message_json, created_at "
                "FROM messages WHERE session_id = ? AND member_id = ? "
                "AND message_id > ? AND message_id <= ? ORDER BY message_id",
                (
                    session_id,
                    trusted_member_id,
                    summary_row["summarized_through_message_id"],
                    cutoff_id,
                ),
            ).fetchall()
            previous_summary = summary_row["summary"]
            previous_cutoff = summary_row["summarized_through_message_id"]

        if not older_rows:
            return

        older_messages = tuple(
            _stored_message_from_row(row).to_conversation_message()
            for row in older_rows
        )
        try:
            new_summary = self._summarizer(previous_summary, older_messages)
        except Exception as exc:
            raise SummarizationError(
                "Summary generation failed; original messages remain stored and unchanged"
            ) from exc
        if (
            not isinstance(new_summary, str)
            or not new_summary.strip()
            or len(new_summary) > 20_000
        ):
            raise SummarizationError(
                "Summarizer returned an empty or oversized summary; original messages remain stored"
            )

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE summaries SET summary = ?, summarized_through_message_id = ?, "
                "updated_at = ? WHERE session_id = ? AND member_id = ? "
                "AND summarized_through_message_id = ?",
                (
                    new_summary.strip(),
                    older_rows[-1]["message_id"],
                    _utc_now(),
                    session_id,
                    trusted_member_id,
                    previous_cutoff,
                ),
            )
            if cursor.rowcount != 1:
                raise SummarizationConflictError(
                    "Summary changed concurrently; retry compaction"
                )

    def _count_tokens(self, value: str) -> int:
        count = self._token_counter(value)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise MemoryConfigurationError(
                "Token counter must return a non-negative integer"
            )
        return count

    def _fits_context(
        self,
        summary: str,
        facts: tuple[StoredFact, ...],
        messages: tuple[StoredMessage, ...],
        current_message: str,
        token_budget: int,
    ) -> bool:
        rendered = _render_context(summary, facts, messages, current_message)
        return self._count_tokens(rendered) <= token_budget

    def _fit_summary_suffix(
        self,
        summary: str,
        facts: tuple[StoredFact, ...],
        messages: tuple[StoredMessage, ...],
        current_message: str,
        token_budget: int,
    ) -> str:
        low = 0
        high = len(summary)
        best = ""
        while low <= high:
            length = (low + high) // 2
            candidate = summary[-length:] if length else ""
            if self._fits_context(
                candidate, facts, messages, current_message, token_budget
            ):
                best = candidate
                low = length + 1
            else:
                high = length - 1
        return best


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_idempotency_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise MemoryConfigurationError(
            "idempotency_key must be a non-empty string of at most 200 characters"
        )
    return value.strip()


def _request_hash(request_message: str) -> str:
    if not isinstance(request_message, str) or not request_message.strip():
        raise MemoryConfigurationError("A non-empty request message is required")
    return hashlib.sha256(request_message.encode("utf-8")).hexdigest()


def _completed_turn_from_row(row: sqlite3.Row) -> CompletedTurn:
    return CompletedTurn(
        session_id=row["session_id"],
        member_id=row["member_id"],
        idempotency_key=row["idempotency_key"],
        response=row["response"],
        completed_at=row["completed_at"],
    )


def _stored_message(
    message: ConversationMessage,
    message_id: int,
    session_id: str,
    member_id: str,
    created_at: str,
) -> StoredMessage:
    return StoredMessage.model_validate(
        {
            **message.model_dump(),
            "message_id": message_id,
            "session_id": session_id,
            "member_id": member_id,
            "created_at": created_at,
        }
    )


def _stored_message_from_row(row: sqlite3.Row) -> StoredMessage:
    message = ConversationMessage.model_validate_json(row["message_json"])
    return _stored_message(
        message,
        row["message_id"],
        row["session_id"],
        row["member_id"],
        row["created_at"],
    )


def _stored_fact_from_row(row: sqlite3.Row) -> StoredFact:
    return StoredFact.model_validate(
        {
            "fact_id": row["fact_id"],
            "session_id": row["session_id"],
            "member_id": row["member_id"],
            "fact_type": row["fact_type"],
            "identifier": row["identifier"],
            "note": row["note"],
            "stored_at": row["stored_at"],
        }
    )


def _render_context(
    summary: str,
    facts: tuple[StoredFact, ...],
    messages: tuple[StoredMessage, ...],
    current_message: str,
) -> str:
    payload = {
        "summary": summary,
        "administrative_facts": [
            fact.model_dump(
                mode="json",
                include={"fact_type", "identifier", "note"},
                exclude_none=True,
            )
            for fact in facts
        ],
        "recent_messages": [
            message.model_dump(
                mode="json",
                exclude={
                    "message_id",
                    "session_id",
                    "member_id",
                    "created_at",
                },
                exclude_none=True,
            )
            for message in messages
        ],
        "current_message": current_message,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
