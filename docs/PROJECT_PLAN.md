# CareFlow build plan

## Definition of done for M1
- [x] Code contains a Pydantic v2 `PatientRequest` model.
- [x] LiteLLM-based provider-independent call accepts a `model` identifier.
- [x] JSON is validated locally; malformed responses generate explicit exceptions.
- [x] Synthetic sample inputs and mocked tests exist.
- [ ] User installs dependencies and runs tests locally.
- [ ] User confirms one live provider call using synthetic inputs.
- [ ] Instructor/TA verifies M1 against official lab checklist.

## Milestone 2 foundation
- [x] Typed Pydantic contracts and deterministic synthetic fixtures for appointment lookup, insurance eligibility, co-pay calculation, and human-review flagging.
- [x] Unknown records use explicit statuses; invalid input raises validation errors.
- [x] Co-pay arithmetic uses `Decimal` and rounds currency to cents.
- [x] Human-review records are auditable in memory only.
- [x] LiteLLM tool schemas and an allowlisted, bounded tool-call harness validate arguments and preserve tool-call IDs/history.
- [x] Recognized clinical-advice requests bypass LiteLLM and create a local human-review record.
- [ ] Connect to approved mock services and add persistence only when specified; no real booking, financial transaction, or clinical action.

## Milestone 3, Phase 1: persistent interaction memory
- [x] Persist synthetic conversation messages in SQLite with separate session and synthetic member ownership.
- [x] Enforce trusted-member fixture allowlisting and session ownership on history/fact access.
- [x] Keep original messages, rolling summaries, and structured administrative facts in separate records.
- [x] Bound rendered prompt context with a tokenizer-aware budget and injectable summarizer/token counter.
- [x] Keep vector indexing separate from the SQLite interaction-memory component.

## Milestone 3, Phase 2: local policy-handbook index
- [x] Add a small, fictional policy corpus with source IDs and named sections.
- [x] Split source sections into deterministic chunks and store them in persistent local Qdrant.
- [x] Validate embedding dimensions and persist the configured model ID to prevent index/model mismatches.
- [x] Support semantic top-k retrieval with source and section filters; return source metadata and scores.
- [x] Make indexing repeatable and test persistence with injected deterministic embeddings.
- [x] Integrate authorized SQLite history and policy evidence with the LiteLLM agent.
- [ ] Add hybrid retrieval, reranking, PDF ingestion, and full RAG evaluation in M4.

## Milestone 4, Phase 1: synthetic PDF ingestion and metadata
- [x] Add 14 explicitly fictional policy PDFs with a validated source manifest.
- [x] Extract text from text-based PDFs with `pypdf`; preserve section and page boundaries for citations.
- [x] Carry source ID, title, category, version, effective date, synthetic declaration, section, and page range through chunks and Qdrant payloads.
- [x] Detect incompatible index schema, parser/chunk configuration, and corpus changes; preserve legacy M3 index search and require explicit reindexing.
- [x] Test valid, empty, malformed, mismatched, repeat-index, metadata, page-citation, and version-change cases; run the full M1-M3 regression suite.

## Milestone 4, Phase 2: hybrid retrieval and reranking
- [x] Add BM25 lexical retrieval (`rank-bm25`) over indexed Qdrant policy chunks, including exact synthetic policy/plan identifier tokenization.
- [x] Fuse dense and BM25 rankings using deterministic Reciprocal Rank Fusion.
- [x] Add a separately injected cross-encoder scoring interface; do not download or select model weights implicitly.
- [x] Preserve source, version, category, specialty, plan, section, and page metadata through each retrieval mode.
- [x] Apply the same metadata filters before dense and lexical ranking; keep dense, BM25, RRF, and reranker scores separate.
- [x] Add deterministic tests for exact identifiers, near duplicates, filter isolation, score separation, ties, empty results, reranker failures, and incompatible index configurations.
- [ ] OCR/scanned PDF support remains unsupported; retrieval evaluation and Langfuse logging are implemented in Phase 4.
- [x] Add an optional CPU sentence-transformers CrossEncoder provider factory while retaining the injectable scorer interface.

## Milestone 4, Phase 3: grounded policy answers and citation validation
- [x] Add an opt-in LiteLLM-backed `GroundedPolicyAnswerer` with typed answer status, exact supporting quotes, and source/section/page/specialty/plan/version citations.
- [x] Reject fabricated citation chunks or metadata, non-verbatim/unsupported output, missing evidence metadata, mismatched specialty/plan/version scope, and detected policy-version or same-scope text conflicts.
- [x] Treat question and PDF evidence as untrusted input; reject detected instruction-like evidence and clinical-advice requests without generating an answer.
- [x] Integrate the answerer as an explicit optional agent dependency while retaining the existing extractive fallback and M1–M3 safety, memory, idempotency, and tool paths.
- [x] Add deterministic mocked tests for valid and fabricated citations, unsupported amounts, conflicting versions, specialty/plan mismatch, missing evidence, PDF prompt injection, clinical escalation, agent integration, and fallback.
- [ ] Full semantic entailment evaluation, broader adversarial prompt-injection coverage, and live-provider validation remain outstanding.

## Milestone 4, Phase 4: retrieval evaluation and Langfuse logging
- [x] Add a versioned, synthetic question set with expected source/section/page labels and expected answer, insufficient-evidence, or human-review behavior.
- [x] Add a no-index-mutation evaluation API and a CLI that indexes the synthetic PDFs only into the dedicated versioned M4 collection before comparing all four retrieval modes.
- [x] Report Precision@5, Recall@5, MRR@5 (with the prior `mrr` output retained), expected-outcome/abstention accuracy, and per-mode mean, p50, and p95 retrieval latency.
- [x] Add a separate grounded-answer evaluator for citation validity, expected-passage citation, exact quotes, unsupported additions/numbers, and expected refusal/outcome accuracy; semantic entailment remains explicitly unassessed.
- [x] Extend the optional Langfuse adapter to create an evaluation trace and attach sanitized aggregate scores; failures return status without changing the report.
- [x] Add optional, pinned CPU CrossEncoder and Langfuse dependencies in `requirements-m4-evaluation.txt`; credentials remain external environment configuration.
- [x] Add mocked tests for provider adapters, CLI collection selection, trace creation/failure isolation, metric formulas, corpus labels, and retrieval-mode failures.
- [ ] Run and record quality comparisons with a configured embedding model and cross-encoder; deterministic test doubles validate mechanics only and do not establish semantic retrieval quality.

### M4 acceptance CLI configuration
- Install optional tools with `.\.venv\Scripts\python.exe -m pip install -r requirements-m4-evaluation.txt`.
- Set `CAREFLOW_EMBEDDING_MODEL`, `CAREFLOW_EMBEDDING_DIMENSION`, and provider credentials through the process environment or an approved secret manager. Optional `CAREFLOW_EMBEDDING_API_KEY` and `CAREFLOW_EMBEDDING_API_BASE` are supported.
- Run `.\.venv\Scripts\python.exe -m careflow.policy_evaluation_cli`; add `--log-langfuse` only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are securely configured.
- The CLI defaults to `data/qdrant_policy_index_m4_v1/` / `synthetic_policy_handbook_m4_v1`; `--index-version v2` selects a distinct versioned target. It never opens or migrates the M3 collection. Index/corpus/model incompatibility fails closed. Use a new M4 version for changed corpus/model settings.
- No live provider run or quality improvement is claimed until the unchecked comparison item above is completed and its measurements are reviewed.

## Milestone 5: LangGraph workflow and human checkpointing
- [x] Add a typed StateGraph for safety-screened M1 intake, trusted-member eligibility lookup, explicit human review, and deterministic synthetic referral routing.
- [x] Reuse `PatientRequest` and M2 eligibility/referral tool contracts; keep model-provided identity out of authorization and tool selection.
- [x] Require a host-owned member/session identity and existing M3 session ownership check before starting or resuming workflow state.
- [x] Persist graph checkpoints and a uniqueness-enforced workflow idempotency ledger in SQLite; reject conflicting request/key reuse and return stored terminal results for duplicates.
- [x] Interrupt all review cases and require a host-injected reviewer authorization callback on resume. Clinical requests remain non-routable regardless of reviewer decision; missing or ineligible coverage cannot be approved around.
- [x] Add tests for successful approved routing, missing/ineligible eligibility, clinical escalation, unauthorized resume, restart recovery, duplicate start/resume, conflicting keys, member/session isolation, and concurrent duplicate starts.
- [x] Add a two-process synthetic demonstration with exact PowerShell commands in README. Its intake parser is deterministic/offline; no external model, insurer, EHR, or referral service is called.
- [ ] Production identity/reviewer authentication, persistent multi-host checkpointing, operator recovery for abandoned in-progress runs, broader safety evaluation, and live service integrations remain out of scope.

### M5 synthetic demonstration
- Install with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- In the first process run `.\.venv\Scripts\python.exe -m careflow.workflow_demo start` and retain the returned `session_id`.
- In a separate process run `.\.venv\Scripts\python.exe -m careflow.workflow_demo resume --session-id "<returned-session-id>" --reviewer-id SYN-REVIEWER-DEMO --decision approve` (or `reject`).
- Demonstration databases are placed in Git-ignored `data/m5_workflow_demo/`. The demo reviewer ID is a fixture, not an authentication mechanism. Production hosts must authenticate the reviewer and provide a trusted authorizer callback.
- SQLite persistence demonstrates process restart on one local host only. Runs interrupted by a process crash before a completed graph checkpoint are not automatically replayed and require manual recovery; the local safety keyword screen is not comprehensive.

## Milestone 6: four-agent coordination and MCP mock server
- [x] Add explicit `IntakeAgent`, `InsurancePolicyRAGAgent`, `ReferralTrackingAgent`, and `ClinicalSafetyReviewer` role adapters; reuse M1 Pydantic intake, M2 mock tools, M3 policy search, M4 grounded answers, and the M5 LangGraph checkpoint workflow.
- [x] Keep clinical safety ahead of parsing/tools; clinical-advice requests reach human review without diagnosis, treatment, or dosage output.
- [x] Add a local FastMCP server over stdio with a fixed tool allowlist, Pydantic argument validation, trusted host-bound member context, and owner-scoped referral lookup.
- [x] Deny MCP state-changing mock actions by default and require an application-supplied authorization callback. Do not accept reviewer identity from tool arguments.
- [x] Add deterministic tests for handoffs, policy insufficiency, member isolation, clinical refusal, invalid/unknown tools, denied actions, unavailable mock tools, and duplicate requests.
- [x] Add an offline four-role synthetic demo and an MCP protocol/stdio verification path; document exact commands in README.
- [ ] Production identity-provider integration, multi-tenant MCP authorization, external EHR connectivity, real referral persistence, and provider-backed multi-agent execution remain out of scope.

### M6 demo and MCP server
- Install the SDK and project dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Run the fully offline role-coordination and clinical-escalation demo with `.\.venv\Scripts\python.exe -m careflow.multi_agent_demo`. It calls the role adapters and M5 graph in-process and does not connect through MCP.
- Launch the actual local MCP stdio server from a compatible MCP host with `.\.venv\Scripts\python.exe -m careflow.mcp_server`. The optional `--allow-synthetic-mutations` switch enables only the synthetic local route/review tools; it is not production authorization.
- The MCP server implements real SDK `tools/list` and `tools/call` over stdio. It is not an HTTP service; production hosts must bind member identity and authorization callbacks to authenticated application context.

## Milestone 7: observability and fault tolerance
- [x] Add an optional Langfuse v2-compatible workflow tracer and reuse the configured M4 evaluation client; trace intake, eligibility, policy retrieval, referral, clinical safety, human approval and resume.
- [x] Correlate with validated synthetic request/session identifiers and allowlisted scalar metadata only. Tracing is nonfatal and records neither raw request/answer/policy content nor reviewer/member IDs or credentials.
- [x] Add configurable bounded exponential retries, per-call timeouts, max concurrent calls, and a thread-safe closed/open/half-open circuit breaker for synthetic service calls.
- [x] Retry only transient `TimeoutError`/`ConnectionError` reads; do not retry validation/authorization failures or referral/review mutations.
- [x] Map exhausted eligibility lookup to a persistent human-review checkpoint; map transient MCP dependency failures to explicit `unavailable` results; prevent duplicate workflow route execution by returning stored idempotent workflow outcomes.
- [x] Add deterministic tests for backoff, timeout, circuit state changes, concurrent calls, trace sanitization/failure, unavailable-service fallback, and mutation non-retry/duplicate resume.
- [x] Add an offline first-call failure-injection demonstration and exact PowerShell command in README.
- [ ] Distributed circuit state, hard cancellation of blocking dependencies, reconciliation of uncertain external mutations, production metrics/alerting, and exhaustive service-policy coverage are not included.

### M7 demo and optional tracing
- Run `.\.venv\Scripts\python.exe -m careflow.m7_failure_demo` for the offline transient eligibility failure, retry, approved referral, and clinical escalation scenario.
- To enable spans in an application, install optional Langfuse support using `.\.venv\Scripts\python.exe -m pip install -r requirements-m4-evaluation.txt`; construct the v2-compatible client from the host's approved secrets manager and pass `LangfuseEvaluationLogger(client).workflow_tracer()` to `CareFlowAgentTeam(observer=...)`.
- Per-call timeouts return while a synchronous worker may still be running. Retry-safe reads may run concurrently after a timeout; mutations are never retried and may need manual reconciliation if their response timed out. The local circuit breaker has process-local state only.

## Milestone 8: final synthetic safety evaluation and FastAPI

- [x] Add an offline FastAPI app factory integrating the existing M5 workflow, M6 role adapters and M7 resilience/tracing.
- [x] Expose `GET /health`, `POST /intake`, `GET /workflows/{workflow_id}`, `POST /workflows/{workflow_id}/review`, and owner-scoped `GET /referrals/{referral_id}` with Pydantic request/response contracts, bounded intake text, explicit error codes, and OpenAPI docs.
- [x] Enforce trusted synthetic member/session identity and host-side reviewer authorization. Approval is accepted only from the reviewer header/host callback, never the request body or LLM. Clinical requests remain non-routable after approval.
- [x] Keep checkpoint and memory data in a separate ignored `data/m8_runtime/` database set; verify API restart/resume using the same workflow identity, duplicate review behavior, and session isolation.
- [x] Add 31 synthetic golden cases in `data/evaluation/m8_golden_cases.json`, covering administrative eligibility/referral, missing eligibility, invalid and missing identities, ambiguous and unsupported specialty, fake clinician/dosing, authorization, duplicate operations, policy refusal/injection/conflict/plan/specialty scope, transient failure, timeout, and open-circuit fallback.
- [x] Add an offline evaluation CLI that validates outcome, authorization, isolation, idempotency, tool calls, clinical non-routing, and grounded-policy answer/refusal behavior. Retrieval-quality metrics remain a separate M4 evaluation and are not represented as measured here.
- [x] Document install, server, endpoint, review, MCP, demo, evaluation, and verification commands. Add `FINAL_SUBMISSION_REPORT.md` with only the executed results and explicit limitations.
- [ ] Production authentication/authorization, external EHR/insurance/referral services, multi-host checkpointing, clinically validated safety controls, and provider-backed retrieval quality are out of scope.

### M8 local run and evaluation
- Install with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Run the offline labeled suite with `.\.venv\Scripts\python.exe -m careflow.m8_evaluation`; expected output is 31 cases, each marked pass/fail, plus synthetic tool-call counts. This suite uses local deterministic fixtures and fake model output; it performs no paid provider calls.
- Start the HTTP service with `.\.venv\Scripts\python.exe -m uvicorn careflow.api:app --host 127.0.0.1 --port 8000`. Health, intake, workflow lookup/review, referral lookup, and OpenAPI are documented in README.
- Local fixture headers are for the demo only. A deployment host must configure both identity and reviewer callbacks and bind member/session/reviewer claims to its authenticated principal. Never expose the development reviewer fixture as production authentication.
- A workflow's SQLite checkpoint is durable on one host. To resume, restart using the same runtime directory, authorized synthetic member, and owned session ID. Clinical approvals do not route referrals. Duplicate review returns the stored terminal outcome.
- Run `.\.venv\Scripts\python.exe -m pytest -q`, `.\.venv\Scripts\python.exe -m careflow.m8_evaluation`, the documented FastAPI `TestClient` smoke command, and `git diff --check`.
- Workflow-safety counts do not measure semantic policy retrieval accuracy, clinical effectiveness, or safety. No real patient/clinic/payer data is used.

## Four mandatory datasets (begin early)
- `data/policies/`: fourteen synthetic policy PDFs with a versioned manifest; `careflow/policy_corpus.py` retains the smaller M3 example corpus.
- `data/intake/`: diverse synthetic intake records (starter examples only).
- `data/mock_api/`: synthetic patients, plans, appointments, referrals (M2 source fixtures are in `careflow/mock_data.py`; expanded dataset not yet created).
- `data/policies/evaluation_cases.json`: ten synthetic M4 retrieval/grounded-answer cases. `data/evaluation/m8_golden_cases.json`: 31 M8 workflow and safety cases.

## Architecture decisions
- Python 3.11+, Pydantic 2, LiteLLM, pytest for M1.
- Add mock tools for M2, Qdrant for M3, RAG and reranking for M4, LangGraph for M5, MCP for M6, Langfuse tracing and bounded fault handling for M7, FastAPI for M8.
- Never use real PHI, real payer IDs, real clinic secrets or actual clinical treatment recommendations.
- Keep each milestone independently demonstrable and testable.

## Human-in-the-loop and safety
- Clinical questions are escalated, not answered with diagnoses or dosages.
- M5 referral routing is not executed until an explicitly authorized host-side reviewer resumes its checkpoint; a clinical escalation can never be approved into ordinary routing.
- Model-only urgency categorization is not a safe clinical decision; decisions require qualified staff.
- M1 keyword marker is for demo only, not a safety guarantee. M8's synthetic adversarial cases exercise selected safeguards, not production clinical validation or a safety guarantee.
