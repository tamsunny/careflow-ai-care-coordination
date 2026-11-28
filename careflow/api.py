"""Offline FastAPI deployment surface for the synthetic CareFlow prototype."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent import SchedulingInsuranceAgent
from .agents import CareFlowAgentTeam, IntakeAgent
from .memory import (
    MemoryOwnershipError,
    MemorySessionNotFoundError,
    SQLiteConversationMemory,
    UnsupportedMemberIdentityError,
)
from .mock_data import MOCK_MEMBERS
from .models import PatientRequest
from .observability import WorkflowObserver
from .resilience import ResilienceConfiguration, ServiceUnavailableError
from .tool_models import ReferralTrackingRecord
from .workflow import (
    WorkflowExecutionError,
    WorkflowIdempotencyConflictError,
    WorkflowIdentityError,
    WorkflowInProgressError,
    WorkflowNotFoundError,
    WorkflowResumeNotAvailableError,
    WorkflowSnapshot,
    WorkflowUnauthorizedResumeError,
)

IdentityAuthenticator = Callable[[str, str | None], bool]
ReviewerAuthenticator = Callable[[str, str, str], bool]
IntakeParser = Callable[[str], PatientRequest]
ToolExecutorFactory = Callable[[], SchedulingInsuranceAgent]

DEV_REVIEWER_ID = "SYN-REVIEWER-DEV"
DEFAULT_RUNTIME_DIR = Path("data") / "m8_runtime"


class IntakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_id: str = Field(
        min_length=12,
        max_length=100,
        pattern=r"^SYN-REQ-[A-Z0-9-]+$",
    )
    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    message: str = Field(min_length=1, max_length=2000)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]


class WorkflowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    request_id: str
    session_id: str
    status: str
    referral_status: str
    approval_state: str
    needs_human_review: bool
    reason_code: str | None
    eligibility_status: str | None
    referral_id: str | None
    errors: list[str]


class IntakeResponse(WorkflowResponse):
    pass


class ReferralResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    referral: ReferralTrackingRecord


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    synthetic_data_only: Literal[True] = True
    authentication_mode: Literal["development_fixture", "host_callback"]


def create_app(
    runtime_dir: str | Path | None = None,
    *,
    identity_authenticator: IdentityAuthenticator | None = None,
    reviewer_authenticator: ReviewerAuthenticator | None = None,
    intake_parser: IntakeParser | None = None,
    tool_executor_factory: ToolExecutorFactory | None = None,
    resilience_config: ResilienceConfiguration | None = None,
    observer: WorkflowObserver | None = None,
) -> FastAPI:
    """Create an isolated app; production hosts must supply auth callbacks."""
    root = Path(runtime_dir or DEFAULT_RUNTIME_DIR).resolve()
    memory: SQLiteConversationMemory | None = None
    teams: dict[str, CareFlowAgentTeam] = {}
    identity_check = identity_authenticator or _development_identity_authenticator
    reviewer_check = reviewer_authenticator or _development_reviewer_authenticator

    def get_memory() -> SQLiteConversationMemory:
        nonlocal memory
        if memory is None:
            root.mkdir(parents=True, exist_ok=True)
            memory = SQLiteConversationMemory(
                root / "memory.sqlite3",
                summarizer=lambda _summary, _messages: "",
                token_counter=lambda text: len(text.split()),
            )
        return memory

    def team_for(member_id: str) -> CareFlowAgentTeam:
        team = teams.get(member_id)
        if team is None:
            parser = intake_parser or deterministic_intake_parser
            team = CareFlowAgentTeam(
                root / f"workflow-{member_id}.sqlite3",
                get_memory(),
                member_id,
                lambda reviewer_id, trusted_member_id, session_id: reviewer_check(
                    reviewer_id,
                    trusted_member_id,
                    session_id,
                ),
                intake_agent=IntakeAgent(parser=parser),
                tool_executor=(
                    tool_executor_factory()
                    if tool_executor_factory is not None
                    else None
                ),
                resilience_config=resilience_config,
                observer=observer,
            )
            teams[member_id] = team
        return team

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for team in teams.values():
            team.close()

    app = FastAPI(
        title="CareFlow Synthetic Prototype API",
        version="8.0.0",
        description=(
            "Fictional administrative workflow demonstration. "
            "Not for clinical use or production authentication."
        ),
        lifespan=lifespan,
    )
    app.state.get_memory = get_memory
    app.state.teams = teams

    @app.exception_handler(Exception)
    async def safe_unexpected_error(_request: Request, _exc: Exception):
        return _json_error(500, "internal_error", "The request could not be completed")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            authentication_mode=(
                "host_callback"
                if identity_authenticator is not None or reviewer_authenticator is not None
                else "development_fixture"
            )
        )

    @app.post("/intake", response_model=IntakeResponse)
    def intake(
        payload: IntakeRequest,
        member_id: Annotated[str | None, Header(alias="X-CareFlow-Member-ID")] = None,
        session_id: Annotated[
            str | None,
            Header(
                alias="X-CareFlow-Session-ID",
                description=(
                    "Omit for a new intake; the API creates and returns a session ID. "
                    "To continue an existing session, use its exact returned ID. "
                    "Unregistered or other-member sessions are rejected."
                ),
            ),
        ] = None,
    ) -> IntakeResponse:
        member = _authenticated_member(member_id, session_id, identity_check)
        session = _get_or_create_session(get_memory(), member, session_id)
        try:
            snapshot = team_for(member).start_workflow(
                session_id=session,
                request_id=payload.request_id,
                idempotency_key=payload.idempotency_key,
                request_text=payload.message,
            )
        except Exception as exc:
            raise _http_error_for(exc) from None
        return _workflow_response(snapshot)

    @app.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
    def get_workflow(
        workflow_id: str,
        member_id: Annotated[str | None, Header(alias="X-CareFlow-Member-ID")] = None,
        session_id: Annotated[str | None, Header(alias="X-CareFlow-Session-ID")] = None,
    ) -> WorkflowResponse:
        session = _require_session(session_id)
        member = _authenticated_member(member_id, session, identity_check)
        try:
            snapshot = team_for(member).get_workflow_snapshot(
                session_id=session,
                workflow_id=workflow_id,
            )
        except Exception as exc:
            raise _http_error_for(exc) from None
        return _workflow_response(snapshot)

    @app.post("/workflows/{workflow_id}/review", response_model=WorkflowResponse)
    def review_workflow(
        workflow_id: str,
        payload: ReviewRequest,
        member_id: Annotated[str | None, Header(alias="X-CareFlow-Member-ID")] = None,
        session_id: Annotated[str | None, Header(alias="X-CareFlow-Session-ID")] = None,
        reviewer_id: Annotated[str | None, Header(alias="X-CareFlow-Reviewer-ID")] = None,
    ) -> WorkflowResponse:
        session = _require_session(session_id)
        member = _authenticated_member(member_id, session, identity_check)
        if reviewer_id is None or not _reviewer_is_authorized(
            reviewer_check, reviewer_id, member, session
        ):
            raise HTTPException(
                status_code=403,
                detail={"code": "reviewer_unauthorized", "message": "Reviewer is not authorized"},
            )
        team = team_for(member)
        try:
            current = team.get_workflow_snapshot(
                session_id=session,
                workflow_id=workflow_id,
            )
            snapshot = team.resume_workflow(
                session_id=session,
                request_id=current.request_id,
                idempotency_key=current.idempotency_key,
                reviewer_id=reviewer_id,
                decision=payload.decision,
            )
        except Exception as exc:
            raise _http_error_for(exc) from None
        return _workflow_response(snapshot)

    @app.get("/referrals/{referral_id}", response_model=ReferralResponse)
    def get_referral(
        referral_id: str,
        member_id: Annotated[str | None, Header(alias="X-CareFlow-Member-ID")] = None,
    ) -> ReferralResponse:
        member = _authenticated_member(member_id, None, identity_check)
        try:
            result = team_for(member).lookup_referral(referral_id)
        except ValidationError:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_referral_id", "message": "Referral ID is invalid"},
            ) from None
        except ServiceUnavailableError:
            raise HTTPException(
                status_code=503,
                detail={"code": "referral_service_unavailable", "message": "Referral lookup is unavailable"},
            ) from None
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "referral_not_found", "message": "Referral was not found"},
            )
        return ReferralResponse(referral=result)

    return app


def deterministic_intake_parser(message: str) -> PatientRequest:
    """Small offline parser for the documented synthetic demonstration."""
    lowered = message.casefold()
    specialties = (
        "cardiology",
        "dermatology",
        "family medicine",
        "pediatrics",
    )
    specialty = next((name for name in specialties if name in lowered), None)
    if specialty == "family medicine":
        specialty = "family_medicine"
    request_type = "referral_request" if "referral" in lowered else "administrative_request"
    return PatientRequest(
        request_type=request_type,
        specialty=specialty,
    )


def _development_identity_authenticator(
    member_id: str,
    session_id: str | None,
) -> bool:
    return member_id in MOCK_MEMBERS and (
        session_id is None
        or (
            session_id.startswith("SYN-SESSION-")
            and len(session_id) <= 64
        )
    )


def _development_reviewer_authenticator(
    reviewer_id: str,
    member_id: str,
    session_id: str,
) -> bool:
    return (
        reviewer_id == DEV_REVIEWER_ID
        and member_id in MOCK_MEMBERS
        and session_id.startswith("SYN-SESSION-")
    )


def _authenticated_member(
    member_id: str | None,
    session_id: str | None,
    authenticator: IdentityAuthenticator,
) -> str:
    try:
        authorized = (
            member_id is not None
            and member_id in MOCK_MEMBERS
            and authenticator(member_id, session_id) is True
        )
    except Exception:
        authorized = False
    if not authorized:
        raise HTTPException(
            status_code=401,
            detail={"code": "identity_unauthorized", "message": "Synthetic identity is not authorized"},
        )
    return member_id


def _reviewer_is_authorized(
    authenticator: ReviewerAuthenticator,
    reviewer_id: str,
    member_id: str,
    session_id: str,
) -> bool:
    try:
        return authenticator(reviewer_id, member_id, session_id) is True
    except Exception:
        return False


def _get_or_create_session(
    memory: SQLiteConversationMemory,
    member_id: str,
    session_id: str | None,
) -> str:
    if session_id is None:
        return memory.create_session(member_id)
    _require_session(session_id)
    try:
        memory.load_facts(session_id, member_id)
    except (MemoryOwnershipError, MemorySessionNotFoundError, UnsupportedMemberIdentityError):
        raise HTTPException(
            status_code=404,
            detail={"code": "session_not_found", "message": "Synthetic session was not found"},
        ) from None
    return session_id


def _require_session(session_id: str | None) -> str:
    if session_id is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "session_required", "message": "A verified session is required"},
        )
    return session_id


def _workflow_response(snapshot: WorkflowSnapshot) -> WorkflowResponse:
    return WorkflowResponse(
        workflow_id=snapshot.workflow_id,
        request_id=snapshot.request_id,
        session_id=snapshot.session_id,
        status=snapshot.workflow_status,
        referral_status=snapshot.referral_status,
        approval_state=snapshot.human_approval_state,
        needs_human_review=(
            snapshot.human_approval_state == "pending"
            or snapshot.workflow_status == "human_reviewed_no_automation"
        ),
        reason_code=snapshot.review_reason,
        eligibility_status=(
            snapshot.eligibility_result.status
            if snapshot.eligibility_result is not None
            else None
        ),
        referral_id=(
            snapshot.referral_result.referral_id
            if snapshot.referral_result is not None
            else None
        ),
        errors=list(snapshot.errors),
    )


def _http_error_for(exc: Exception) -> HTTPException:
    if isinstance(exc, WorkflowUnauthorizedResumeError):
        return HTTPException(403, detail={"code": "reviewer_unauthorized", "message": "Reviewer is not authorized"})
    if isinstance(exc, (WorkflowIdentityError, MemoryOwnershipError, UnsupportedMemberIdentityError)):
        return HTTPException(401, detail={"code": "identity_unauthorized", "message": "Workflow identity is not authorized"})
    if isinstance(exc, WorkflowNotFoundError):
        return HTTPException(404, detail={"code": "workflow_not_found", "message": "Workflow was not found"})
    if isinstance(exc, WorkflowResumeNotAvailableError):
        return HTTPException(409, detail={"code": "review_not_pending", "message": "Workflow is not awaiting review"})
    if isinstance(exc, (WorkflowIdempotencyConflictError, WorkflowInProgressError)):
        return HTTPException(409, detail={"code": "workflow_conflict", "message": "Workflow conflicts with an existing request"})
    if isinstance(exc, WorkflowExecutionError):
        return HTTPException(503, detail={"code": "workflow_unavailable", "message": "Workflow could not be completed"})
    if isinstance(exc, (ValidationError, ValueError)):
        return HTTPException(422, detail={"code": "invalid_workflow_request", "message": "Workflow request is invalid"})
    return HTTPException(500, detail={"code": "internal_error", "message": "The request could not be completed"})


def _json_error(status_code: int, code: str, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
    )


app = create_app()
