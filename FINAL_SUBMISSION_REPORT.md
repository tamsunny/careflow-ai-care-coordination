# CareFlow Capstone Submission Report

## Delivery summary

Milestone 8 is implemented as an offline synthetic FastAPI deployment package and a reproducible 31-case safety/workflow evaluation. It reuses the existing M5 LangGraph workflow, M6 role adapters and mock tools, and M7 resilience/observability hooks. No provider credentials or real patient data were used; no commit or push was performed.

The workflow API preserves owner-scoped SQLite checkpoints, host-authorized review, idempotent repeated review, and clinical non-routing. A workflow lookup API reads only records belonging to the authenticated synthetic member and session. Importing the ASGI app does not create local databases; the separate M8 runtime is created only when the app handles a request.

## Eight milestone overview

| Milestone | Implemented capability | Submission limitation |
|---|---|---|
| M1 | Pydantic intake, provider-independent LiteLLM integration, clinical request escalation | The deterministic safety screen is not clinically validated; no live provider call was made for this final verification. |
| M2 | Typed deterministic synthetic scheduling, eligibility, co-pay, review, and referral tools | No real EHR, payer, appointment, or referral operations. |
| M3 | SQLite interaction memory and local Qdrant policy index | Local demonstration storage, not a production multi-host store. |
| M4 | Fictional PDFs, metadata ingestion, dense/BM25/RRF retrieval, reranking interface, grounded-answer validation, evaluation and optional Langfuse adapter | No real-provider retrieval comparison is included in this report; no retrieval-quality improvement is claimed. |
| M5 | Persistent LangGraph intake → eligibility → review → referral workflow | SQLite checkpointing is single-host; clinical review never enables ordinary referral routing. |
| M6 | Four explicit role adapters and an allowlisted MCP server using the official SDK over stdio | MCP is local stdio, not a hosted network service; mock-only tools. |
| M7 | Optional sanitized tracing, bounded retry/timeout/circuit-breaker controls and fallback behavior | Circuit state is process-local; timeouts cannot cancel already-running synchronous work. |
| M8 | FastAPI endpoints, owner-scoped workflow read/review, offline 31-case labeled evaluation | Development fixture identities are not production authentication; no clinical validation or production safety claim. |

## M8 implementation

The API in `careflow.api` exposes:

- `GET /health`
- `POST /intake`
- `GET /workflows/{workflow_id}`
- `POST /workflows/{workflow_id}/review`
- `GET /referrals/{referral_id}`

OpenAPI is available at `/openapi.json`; Swagger UI is at `/docs`. Intake messages are limited to 2,000 characters, request/response data is Pydantic-validated, and unexpected error details are not returned. Workflow responses omit raw intake text and graph interrupt payloads.

The API uses the existing SQLite-backed workflow with separate M8 runtime files under Git-ignored `data/m8_runtime/`. A session returned by intake can be used to look up or resume the workflow after restarting the service on the same host. Repeated authorized review requests return the existing terminal result. Clinical approval remains non-routable.

The 31 labeled synthetic cases are in `data/evaluation/m8_golden_cases.json`. They cover administrative eligibility and referral, missing eligibility, ambiguous/unsupported specialties, missing and invalid identities, fake clinician and dosage requests, reviewer authorization, duplicate start/review, member/session isolation, referral lookup, policy injection, missing/conflicting evidence, wrong plan/specialty, unsupported answers, transient failure, timeout, and open-circuit fallback. The runner reports synthetic tool-call counts separately from retrieval-quality metrics.

## Verification actually run

- Full Python 3.12 suite: **178 passed** (`.\.venv\Scripts\python.exe -m pytest -q`).
- M8 golden evaluation: **31 passed, 0 failed** (`.\.venv\Scripts\python.exe -m careflow.m8_evaluation`).
- Observed tool calls in the main workflow evaluation: **10 eligibility lookups, 4 referral routes, 2 referral lookups**. Failure-injection cases run in isolated app instances.
- FastAPI smoke test: **`GET /health` returned HTTP 200** with `synthetic_data_only: true` and `authentication_mode: development_fixture`.
- `git diff --check`: passed with no output.
- No live embedding, LLM, reranker, Langfuse, payer, or EHR call was made. Retrieval metrics were not measured by the M8 workflow-safety evaluation.

## Run instructions

Install the required offline dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the golden evaluation and full test suite:

```powershell
.\.venv\Scripts\python.exe -m careflow.m8_evaluation
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Start the local API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn careflow.api:app --host 127.0.0.1 --port 8000
```

The local demo accepts `X-CareFlow-Member-ID: SYN-MEMBER-001` and uses `SYN-REVIEWER-DEV` in `X-CareFlow-Reviewer-ID` for human review. Keep the `session_id` returned from `POST /intake` and send it as `X-CareFlow-Session-ID` on workflow lookup/review. See README for complete PowerShell request examples and the MCP server command.

## Limitations and remaining blockers

- Development identity/reviewer headers are fixtures, not authentication. A real host must supply callbacks backed by an authenticated identity provider, bind the member/session/reviewer claims to that principal, and protect transport.
- The default intake parser is deterministic and recognizes only a short list of synthetic specialties. It is for the offline demonstration, not production language understanding.
- The clinical request detector and policy prompt-injection checks are limited heuristics. Neither the system nor this evaluation is clinically validated, and no production safety guarantee is made.
- EHR, insurance, scheduling, and referral data and actions are deterministic local mocks. Newly routed referrals are not persisted in the fixture lookup records.
- SQLite persistence and M7 circuit state are local/single-host. Timeouts may leave synchronous work running; uncertain external mutation reconciliation is not applicable to the local mocks and is not implemented for real services.
- M4 semantic retrieval quality, cross-encoder quality, and live-provider behavior remain unmeasured here. The 31 M8 cases test synthetic workflow safety and grounded-answer validation mechanics, not retrieval ranking metrics or semantic entailment.
- No credentials, `.env`, or real patient data were accessed or changed. Changes remain uncommitted and unpushed for review.
