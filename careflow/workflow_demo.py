"""Run the synthetic M5 workflow in separate start and resume processes."""

import argparse
import json
from pathlib import Path
from typing import Literal

from .memory import SQLiteConversationMemory
from .models import PatientRequest
from .workflow import LangGraphWorkflow, WorkflowSnapshot

MEMBER_ID = "SYN-MEMBER-001"
REQUEST_ID = "SYN-REQ-M5-DEMO-001"
IDEMPOTENCY_KEY = "m5-demo-referral-turn-001"
REVIEWER_ID = "SYN-REVIEWER-DEMO"
REQUEST_TEXT = "Please create a synthetic cardiology referral."
DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "m5_workflow_demo"


def _make_memory() -> SQLiteConversationMemory:
    return SQLiteConversationMemory(
        DATA_DIRECTORY / "memory.sqlite3",
        lambda previous, _messages: previous,
        token_counter=lambda text: len(text),
    )


def _parse_intake(_text: str) -> PatientRequest:
    return PatientRequest(
        request_type="referral",
        specialty="cardiology",
    )


def _authorize_reviewer(
    reviewer_id: str,
    member_id: str,
    _session_id: str,
) -> bool:
    return reviewer_id == REVIEWER_ID and member_id == MEMBER_ID


def _make_workflow(memory: SQLiteConversationMemory) -> LangGraphWorkflow:
    return LangGraphWorkflow(
        DATA_DIRECTORY / "checkpoints.sqlite3",
        memory,
        _authorize_reviewer,
        intake_parser=_parse_intake,
    )


def run_start(session_id: str | None) -> WorkflowSnapshot:
    memory = _make_memory()
    session_id = session_id or memory.create_session(MEMBER_ID)
    memory.load_facts(session_id, MEMBER_ID)
    with _make_workflow(memory) as workflow:
        return workflow.start_workflow(
            trusted_member_id=MEMBER_ID,
            session_id=session_id,
            request_id=REQUEST_ID,
            idempotency_key=IDEMPOTENCY_KEY,
            request_text=REQUEST_TEXT,
        )


def run_resume(
    session_id: str,
    reviewer_id: str,
    decision: Literal["approve", "reject"],
) -> WorkflowSnapshot:
    memory = _make_memory()
    with _make_workflow(memory) as workflow:
        return workflow.resume_workflow(
            trusted_member_id=MEMBER_ID,
            session_id=session_id,
            request_id=REQUEST_ID,
            idempotency_key=IDEMPOTENCY_KEY,
            reviewer_id=reviewer_id,
            decision=decision,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic-only CareFlow M5 workflow demonstration."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser(
        "start",
        help="Create a synthetic referral and pause for human approval.",
    )
    start_parser.add_argument(
        "--session-id",
        help="Continue the same synthetic session instead of creating one.",
    )
    resume_parser = commands.add_parser(
        "resume",
        help="Resume a paused workflow as an explicitly supplied reviewer.",
    )
    resume_parser.add_argument("--session-id", required=True)
    resume_parser.add_argument("--reviewer-id", required=True)
    resume_parser.add_argument(
        "--decision",
        choices=("approve", "reject"),
        required=True,
    )
    args = parser.parse_args()

    if args.command == "start":
        result = run_start(args.session_id)
    else:
        result = run_resume(
            args.session_id,
            args.reviewer_id,
            args.decision,
        )
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
