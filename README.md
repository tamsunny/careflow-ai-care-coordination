# CareFlow — AI Care Coordination Assistant

**IIT Hyderabad Applied AI Professional Certification — Capstone starter**

CareFlow is a **healthcare-operations prototype**, not a medical device. It parses synthetic outpatient intake messages and will progressively add scheduling tools, insurance/policy RAG, referral tracking, human oversight, reliability and evaluation.

> **Strict data rule:** Only synthetic/de-identified test data. Never commit real patient names, medical records, insurance numbers, credentials, or confidential clinic documents. No diagnostic, dosing or treatment advice.

## Current scope

**Milestone M1 (starter implementation)**
- `PatientRequest` Pydantic v2 model (symptoms, urgency flag, insurance ID).
- Provider-independent `litellm.completion()` wrapper.
- Validated structured JSON parse and explicit parsing failure.
- Conservative deterministic flagging for requests explicitly seeking diagnosis, treatment, or dosing advice. This is a **prototype safety check**, not a clinically validated triage system; the model's urgency flag must not be used to decide actual urgency.
- Unit tests using a fake LLM response (no API calls or credentials needed).

**Milestone M2 (synthetic tool foundation)**
- `SchedulingInsuranceAgent` exposes typed Pydantic inputs and outputs for appointment lookup, eligibility lookup, mock co-pay calculation, and staff-review flagging.
- `LiteLLMSchedulingInsuranceAgent` advertises those four tools to LiteLLM and can select/invoke them; an explicit allowlist and Pydantic validation gate every model-proposed call.
- Tool implementations remain independent of the model and run against small fictional fixtures. Unknown tools, invalid arguments, tool errors, and tool-round exhaustion do not count as successful operations.
- Appointment lookup only returns available slots; it does not book appointments. Eligibility and co-pay results are fictional and do not initiate financial transactions or make clinical decisions.
- Human-review records are kept in memory for the lifetime of the agent instance and are not sent to real staff or persisted.
- A deterministic keyword/pattern screen routes recognized clinical-advice requests directly to a local human-review record before calling LiteLLM. This demo screen is not exhaustive or clinically validated.
- Unknown records have explicit `not_found` statuses. Invalid inputs raise Pydantic validation errors; unexpected technical errors are not converted to record-not-found responses.

**Milestone M3 Phase 1 (persistent interaction memory)**
- `SQLiteConversationMemory` persists synthetic user, assistant, and tool messages, scoped by a generated session ID and an explicitly supplied trusted synthetic member ID.
- History loads require both identifiers and enforce the session owner. Member IDs must match the synthetic fixture allowlist; identity is never inferred from message text or model output.
- Original messages remain in SQLite. Rolling summaries and typed administrative facts (for example `SYN-REF-004`) are stored separately.
- Prompt context includes a tokenizer-counted budget using `tiktoken` by default, with injectable token counters and summarizers for offline tests.
- SQLite database files and sidecars are excluded by `.gitignore`. This phase does not add external identity/authentication.

**Milestone M3 Phase 2 (synthetic policy index)**
- `LocalPolicyIndex` chunks a small source-controlled fictional handbook by section and indexes it in persistent local Qdrant; returned results include source ID, section, text, and similarity score.
- The default corpus is in `careflow/policy_corpus.py`. Search supports configurable `top_k` and exact source/section filters.
- A caller-supplied embedder and model ID are used consistently; vector dimensions are checked, and a local manifest prevents reopening an index with a different model configuration.
- Indexing synchronizes the collection to the supplied corpus using deterministic point IDs. Local vector data is stored under the Git-ignored `data/qdrant_policy_index/`.
- Policy indexing is separate from SQLite conversation memory. This phase does not add hybrid search, reranking, PDF processing, or full RAG evaluation. Tests inject deterministic embeddings and require no API credentials.

**M3 integration**
- The LiteLLM agent can receive `memory` and `policy_index` instances. When memory is enabled, each `respond` call requires host-supplied `trusted_member_id`, `session_id`, and a stable `idempotency_key`; only authorized history and facts are loaded, and successful turns are committed atomically.
- Policy passages are placed in an explicitly untrusted user-context message with source and section metadata. Policy-specific answers are extractive, cited passages (not model-generated paraphrases) and fail closed if no passage meets the configurable similarity threshold (default `0.35`); tune this threshold for the selected embedding model.
- Recognized clinical-advice requests bypass model calls and policy retrieval, create a local human-review record, and persist the synthetic turn when memory is enabled.
- A completed key returns its stored response without another model or tool call; reusing that key for different request text is rejected. In-process concurrent retries are serialized by key. Separate processes do not reserve in-flight keys, so simultaneous cross-process requests could both call the model/tools before SQLite accepts only one completed turn.
- Summary refresh runs separately after turn commit. A refresh error is logged without request contents and does not change the successful response; retry it with `memory.refresh_summary(...)`. A retry after a completed turn is idempotent.
- This remains a prototype: host identity verification is external, local Qdrant is single-process storage, and the threshold/summarizer/embedding model require deployment-specific validation.

**Milestone M4 Phase 1 (synthetic PDF ingestion and metadata)**
- `data/policies/` contains 14 fictional, text-based policy PDFs and a validating `manifest.json`; documents cover referrals, pre-authorization, synthetic co-pay schedules, administrative escalation/intake, and specialty handbooks.
- `careflow.policy_ingestion` validates the manifest and embedded PDF identifiers, extracts sections with page numbers using `pypdf`, and returns typed synthetic documents. Scanned PDFs/OCR are not supported.
- Policy chunks and Qdrant payloads include source ID, category, version, effective date, section, synthetic-data marker, and page citation. Run `.\.venv\Scripts\python.exe -m careflow.policy_ingestion` to validate and extract the local corpus without API credentials.
- The versioned index manifest records embedding settings, parser/chunking/index schema versions, and a canonical corpus fingerprint. An incompatible M3 index remains searchable but cannot be re-indexed implicitly. Create a separate versioned collection or persistent directory for the M4 corpus; corpus changes to an M4 collection require `index_documents(..., reindex=True)`.
- Phase 1 does not add generated grounded answers or retrieval evaluation. The existing caller-supplied embedder remains required for Qdrant indexing; the ingestion command itself does not create embeddings.

Install dependencies and validate/extract all 14 PDFs locally with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m careflow.policy_ingestion
```

For Qdrant indexing, pass `load_policy_documents()` to `LocalPolicyIndex.index_documents()` with the application's configured embedder and `PARSER_VERSION`. Use a new M4 collection name (for example `synthetic_policy_handbook_m4`) and an ignored persistence directory such as `data/qdrant_policy_index_m4/`; do not point an M4 reindex at the existing M3 collection unless an explicit migration is intended.

**Milestone M4 Phase 2 (hybrid retrieval and reranking)**
- `LocalPolicyIndex.search()` remains pure dense retrieval; `search_bm25()` computes BM25 over locally indexed chunks. `PolicyHybridRetriever` exposes `search_dense()`, `search_bm25()`, `search_hybrid()` (RRF, default `rrf_k=60`), and `search_hybrid_reranked()` independently.
- Source, section, category, specialty, synthetic plan ID, and document version filters are applied to both dense Qdrant search and lexical BM25 candidate collection.
- `PolicySearchResult` keeps `dense_score`, `bm25_score`, `rrf_score`, and `reranker_score` separate. Legacy `score` remains a dense-score alias; the M3 agent's `0.35` threshold is applied to dense scores only.
- `CrossEncoderReranker` retains the injectable `(query, passage)` scorer interface. `careflow.policy_providers.create_cpu_cross_encoder_reranker()` loads the configurable sentence-transformers CrossEncoder on CPU; its default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Weights are loaded/downloaded only when the acceptance CLI is run. Tests use deterministic fake scores.
- BM25 currently scrolls the local Qdrant payload and rebuilds its small lexical corpus per query. This is appropriate for the synthetic handbook, not a large production corpus. The agent still retrieves with dense search by default; hybrid retrieval is available through the separate evaluation/retrieval API.

**Milestone M4 Phase 3 (grounded policy answers)**
- `GroundedPolicyAnswerer` uses the provider-independent `litellm.completion()` interface and accepts retrieved policy passages. Each answer includes a status, verbatim supporting quote(s), and citations carrying source, section, page, specialty, plan and version metadata.
- A response is accepted only when every quote is an exact substring of retrieved text, the answer exactly reproduces those quotes, and all citation metadata matches the cited chunk. Fabricated chunk IDs, unsupported additions (including numbers), malformed schemas, scope mismatches, missing page metadata, and detected version/text conflicts fail closed.
- The question and policy text are sent as untrusted data; obvious instruction-like PDF content is rejected before the model call. This is defense in depth, not a guarantee against every prompt injection or a semantic entailment proof.
- Integration is opt-in: pass `policy_answerer=...` alongside the M4 `policy_index` when constructing `LiteLLMSchedulingInsuranceAgent`. Without it, the existing extractive response is unchanged. If model output cannot be validated, the agent falls back to the existing extractive passages; missing, inapplicable, or contradictory evidence yields an insufficient-evidence response. Clinical-advice requests still bypass retrieval and model generation and use M1/M2 human review handling.
- This is synthetic-only. Exact-quote and metadata validation do not establish that a real policy is authentic, current, or applicable; specialty recognition, contradiction detection, and clinical-language detection are limited deterministic checks. No live provider or clinical safety validation is included.

**Milestone M4 Phase 4 (retrieval evaluation and Langfuse logging)**
- `data/policies/evaluation_cases.json` contains ten synthetic cases with expected source/section/page labels, two insufficient-evidence cases, and a clinical human-review case. Tests check labels against the ingested PDF corpus.
- `run_policy_retrieval_evaluation()` compares dense, BM25, RRF hybrid, and hybrid-plus-reranking. `python -m careflow.policy_evaluation_cli` loads the 14 PDFs, indexes them in the dedicated `synthetic_policy_handbook_m4_v1` collection, and evaluates the ten synthetic cases. The report includes Precision@5, Recall@5, MRR@5 (the existing `mrr` field remains as a compatibility alias), and mean/p50/p95 latency. Two insufficient-evidence cases and one human-review case are reported separately; human review skips retrieval.
- `run_grounded_answer_evaluation()` separately checks citation metadata, expected passage citations, verbatim quote matching, unsupported answer additions/numeric values, and expected refusal/outcome status. Citation correctness and quote matching are distinct metrics; semantic entailment is explicitly reported as `not_assessed`. A missing/insufficient answer on an answerable case is an outcome failure, not an unsupported-claim finding.
- `LangfuseEvaluationLogger` optionally creates a retrieval/answer evaluation trace and attaches aggregate scores (hashed dataset version, case count, mode, metric). It never sends questions, answers, passages, member IDs, or credentials. The CLI's `--log-langfuse` uses `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optional `LANGFUSE_HOST`; trace/score failures are returned as a sanitized status and do not change the retrieval report.
- Meaningful quality comparisons require the caller's configured production embedding model and cross-encoder. Deterministic test embeddings/reranker validate evaluation mechanics only; they do not establish semantic quality. Timing values also depend on local hardware and model/provider setup.

### Reproducible M4 retrieval acceptance run

The CLI defaults to the separate `data/qdrant_policy_index_m4_v1/` directory and `synthetic_policy_handbook_m4_v1` collection. It explicitly indexes `load_policy_documents()` and does not use or migrate the M3 collection. Repeated runs with the same corpus and embedding configuration validate stored point IDs and skip re-embedding/upserting. Corpus/pipeline mismatches fail under the existing manifest safeguards; there is deliberately no implicit destructive reindex option. For a changed corpus or embedding model, pass a new version such as `--index-version v2`; this selects a distinct M4 collection and persistence directory.

Install the optional CPU reranker and Langfuse dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-m4-evaluation.txt
```

Set the real provider model and its exact vector dimension in the process environment. LiteLLM's provider-native credential variables can be used, or set `CAREFLOW_EMBEDDING_API_KEY`; `CAREFLOW_EMBEDDING_API_BASE` is optional for compatible endpoints. Example model settings (the API key is intentionally not shown):

```powershell
$env:CAREFLOW_EMBEDDING_MODEL = "openai/text-embedding-3-small"
$env:CAREFLOW_EMBEDDING_DIMENSION = "1536"
.\.venv\Scripts\python.exe -m careflow.policy_evaluation_cli
```

Use a fresh M4 index version when changing model/corpus settings:

```powershell
.\.venv\Scripts\python.exe -m careflow.policy_evaluation_cli --index-version v2
```

The CPU CrossEncoder defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`; override with `CAREFLOW_CROSS_ENCODER_MODEL` or point it to an already cached/local model. Its weights may be downloaded on first use. To also create a Langfuse trace and attach aggregate metrics, securely populate `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (and optionally `LANGFUSE_HOST`) in the process environment, then run:

```powershell
.\.venv\Scripts\python.exe -m careflow.policy_evaluation_cli --log-langfuse
```

The JSON output contains one retrieval report with all four modes. Each mode evaluates seven answerable and two insufficient-evidence cases (nine retrieval queries; the clinical case is skipped), including Precision@5, Recall@5, MRR@5, and latency. Optional Langfuse output reports `success`, a generated trace ID, and the score count or a sanitized error code. Run this only when authorized: indexing and queries send synthetic policy text/questions to the configured embedding service and may incur provider costs. This repository does not run or claim a live-provider quality comparison automatically.

Example opt-in agent setup:

```python
from careflow.llm_agent import LiteLLMSchedulingInsuranceAgent
from careflow.policy_answers import GroundedPolicyAnswerer

answerer = GroundedPolicyAnswerer(model="your-configured-model")
agent = LiteLLMSchedulingInsuranceAgent(
    model="your-configured-model",
    policy_index=policy_index,
    policy_answerer=answerer,
)
```

The same four retrieval modes remain directly callable from application code:

```python
from careflow.policy_models import PolicySearchFilters
from careflow.policy_retrieval import PolicyHybridRetriever

retriever = PolicyHybridRetriever(index, reranker=reranker)
filters = PolicySearchFilters(
    specialty="cardiology",
    plan_id="SYN-PLAN-12",
    version="1.0",
)
dense = retriever.search_dense("cardiology SYN-PLAN-12 copay", filters=filters)
lexical = retriever.search_bm25("cardiology SYN-PLAN-12 copay", filters=filters)
hybrid = retriever.search_hybrid("cardiology SYN-PLAN-12 copay", filters=filters)
reranked = retriever.search_hybrid_reranked(
    "cardiology SYN-PLAN-12 copay", filters=filters
)

for mode, results in (
    ("dense", dense),
    ("BM25", lexical),
    ("RRF", hybrid),
    ("RRF + reranker", reranked),
):
    print(mode, [
        (
            result.source_id,
            result.section_name,
            result.page_start,
            result.dense_score,
            result.bm25_score,
            result.rrf_score,
            result.reranker_score,
        )
        for result in results
    ])
```

Run the end-to-end synthetic verification with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_m3_end_to_end.py
```

The test uses a deterministic fake embedder and fake LLM responses with real local SQLite and Qdrant persistence. It verifies integration mechanics only; fake embeddings do **not** establish semantic retrieval quality. No live embedding-provider or live-agent smoke test is included.

```python
agent = LiteLLMSchedulingInsuranceAgent(
    model="your-configured-model",
    memory=memory,
    policy_index=policy_index,
)
reply = agent.respond(
    "Check this synthetic request.",
    trusted_member_id="SYN-MEMBER-001",
    session_id=session_id,  # supplied by the host application
    idempotency_key=host_generated_request_id,  # reuse for retries of this logical turn
)
```

Example appointment lookup (no API key required):

```python
from careflow.agent import SchedulingInsuranceAgent
from careflow.tool_models import AppointmentLookupInput

agent = SchedulingInsuranceAgent()
result = agent.lookup_appointments(
    AppointmentLookupInput(specialty="cardiology")
)
print(result.model_dump_json(indent=2))
```

For model-selected tool calls, configure the provider key in the process environment as described below, then:

```python
from careflow.llm_agent import LiteLLMSchedulingInsuranceAgent

agent = LiteLLMSchedulingInsuranceAgent(model="groq/your-available-model-id")
print(agent.respond("Find available synthetic cardiology appointments."))
```

The LiteLLM harness keeps tool-call IDs and tool messages in its in-memory conversation history and permits at most four tool-call rounds by default.

**Milestone M5 (LangGraph intake → eligibility → referral)**
- `careflow.workflow.LangGraphWorkflow` builds a typed LangGraph `StateGraph` that runs the M1 clinical-advice screen before intake parsing, checks eligibility with the trusted host-side synthetic member ID, and requires a separately authorized human decision before deterministic referral routing.
- Workflow identity includes the host-supplied member ID, an existing owned SQLite memory session, request ID, idempotency key and request text. The workflow registry rejects conflicting key/request reuse and returns stored results for duplicate completed starts/resumes. Model-parsed member IDs never select whose eligibility is checked.
- LangGraph `interrupt()` checkpoints are persisted with `SqliteSaver` in a local SQLite database. A host-supplied `reviewer_authorizer(reviewer_id, member_id, session_id)` must explicitly return `True` on resume; human approval cannot override a clinical escalation or missing/ineligible coverage.
- `careflow.workflow_demo` demonstrates a synthetic cardiology referral across two separate process invocations. It uses a deterministic intake parser and M2 mock eligibility/referral tools, so it makes no LLM/API calls. Local memory and checkpoint databases are under `data/m5_workflow_demo/` and are Git-ignored.
- Install and run the demo in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m careflow.workflow_demo start
```

Copy the `session_id` from the paused result, then in a **new process** explicitly approve or reject:

```powershell
.\.venv\Scripts\python.exe -m careflow.workflow_demo resume --session-id "SYN-SESSION-<copied-id>" --reviewer-id SYN-REVIEWER-DEMO --decision approve
```

Replace `approve` with `reject` to stop the fictional route. This demo's reviewer ID is a fixed synthetic fixture, not authentication. Real hosts must authenticate the reviewer and inject an authorizer bound to that trusted identity; they must never trust an LLM-generated decision. `langgraph-checkpoint-sqlite` 3.x is required for the LangGraph checkpoint 4.x API.
- This is a local synthetic prototype, not a workflow service: member identities are restricted by the existing M3 fixture allowlist, clinical-advice detection is a limited keyword/pattern guard, mock operations do not contact an EHR or insurer, and SQLite checkpointing is intended for a single local application host. A process crash before a workflow reaches a persisted interrupt/terminal checkpoint leaves that idempotency key blocked for manual operational recovery; no automated replay or recovery UI is included.

**Milestone M6 (four explicit roles and MCP mock server)**
- `careflow.agents` defines `IntakeAgent`, `InsurancePolicyRAGAgent`, `ReferralTrackingAgent`, and `ClinicalSafetyReviewer`. `CareFlowAgentTeam` wires those role adapters into the persistent M5 LangGraph: the existing `PatientRequest` schema and mock eligibility/referral tools are reused, and policy answers can use injected M3 `LocalPolicyIndex` plus M4 `GroundedPolicyAnswerer`. If policy retrieval is not configured, policy questions return insufficient evidence; clinical questions are escalated without an advice-generation call.
- The M5 workflow still owns safety gating, member/session isolation, idempotency, persistent review interrupts, and reviewer authorization. M6 does not accept LLM-provided reviewer identities.
- `careflow.mcp_server` uses the official Python MCP SDK (`FastMCP`) and serves real MCP `tools/list` and `tools/call` over **stdio**. Its allowlisted tools provide appointment lookup, eligibility for the host-bound synthetic member, mock co-pay calculation, referral routing, owner-scoped referral lookup, and local human-review records. Each input is validated with the existing Pydantic tool models; unknown tools and extra arguments are rejected.
- MCP mutations (`route_referral` and `flag_for_human`) are denied by default. A host must inject `authorize_action(action, trusted_member_id)`; tool arguments never contain a reviewer identity or trusted member override. The CLI's `--allow-synthetic-mutations` is only a local fixture switch, not production authentication. Start the server from an MCP client/host:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m careflow.mcp_server
```

For the local synthetic fixture only, optionally permit mocked mutations:

```powershell
.\.venv\Scripts\python.exe -m careflow.mcp_server --member-id SYN-MEMBER-001 --allow-synthetic-mutations
```

- Run the offline four-role intake → eligibility → authorized referral demo and clinical escalation:

```powershell
.\.venv\Scripts\python.exe -m careflow.multi_agent_demo
```

The demo injects a deterministic parser and uses local M2–M5 mock tools/checkpoints; it makes no provider calls. It runs the role handoffs **in-process** and does not connect to the MCP server. The separate MCP server uses the actual MCP protocol; its tool list/call behavior is covered through the SDK protocol transport and a local stdio client smoke test. This is not a network-hosted MCP service, production identity provider, real EHR, or multi-agent LLM deployment.

**Milestone M7 (optional tracing and bounded fault tolerance)**
- `LangfuseWorkflowTracer` adds optional spans for clinical safety, intake, eligibility, policy retrieval, referral handling, human approval, and resume. Supply a configured Langfuse v2-compatible client, or obtain the tracer from the existing `LangfuseEvaluationLogger.workflow_tracer()`. Without a client, tracing is a no-op. Traces correlate only validated synthetic request/session IDs and use allowlisted status/count metadata; request text, answers, policy passages, reviewer/member identities and credential fields are excluded. Trace/client errors never escape into authorization or workflow execution.
- Retry, timeout, concurrency and circuit settings are validated in `ResilienceConfiguration`. Transient `TimeoutError`/`ConnectionError` reads (eligibility, appointment/co-pay lookups) retry with bounded exponential backoff; validation, authorization and other non-transient errors do not retry. The circuit breaker is thread-safe and supports open/half-open probing. M6’s MCP service returns a structured `unavailable` result for exhausted transient failures; the M5 graph converts eligibility outages into a persistent human-review checkpoint and does not route without confirmed eligibility.
- Referral routing and human-review creation are non-retryable mutations even after a timeout, because a timed-out synchronous call may still finish. The persistent M5 workflow ledger returns its stored terminal result on duplicate start/resume, preventing a second visible workflow action. A direct, manually repeated MCP human-review call has no durable idempotency ledger; callers must not automatically retry it.
- Run the deterministic offline failure-injection demonstration:

```powershell
.\.venv\Scripts\python.exe -m careflow.m7_failure_demo
```

It injects a transient failure on the first synthetic eligibility lookup, verifies bounded retry and authorized referral approval, and separately confirms clinical escalation never routes. No API/provider calls are made. To enable Langfuse spans in an application, install the optional client with `.\.venv\Scripts\python.exe -m pip install -r requirements-m4-evaluation.txt`, create the client through the host's approved secret-management path, then pass `LangfuseEvaluationLogger(client).workflow_tracer()` as `observer` to `CareFlowAgentTeam`. No credentials are needed for tests or the demo.
- Limitations: timeouts cannot forcibly stop a synchronous operation already running in a worker thread. Safe reads may continue while a retry runs; mutations are deliberately called once and can complete after the caller received an unavailable result, so manual reconciliation may be needed. Circuit state is in-process and resets on restart. The retryable exception classes and local safety detector are intentionally narrow; this is not production circuit-breaker/service supervision.

**Milestone M8 (FastAPI deployment and final synthetic evaluation)**
- `careflow.api:create_app()` exposes `GET /health`, `POST /intake`, `GET /workflows/{workflow_id}`, `POST /workflows/{workflow_id}/review`, and `GET /referrals/{referral_id}`. OpenAPI is available at `/openapi.json` and Swagger UI at `/docs`.
- Workflow lookup is owner-scoped by authenticated synthetic member and existing SQLite session. Responses omit raw request text and interrupt payloads. Intake is capped at 2,000 characters; Pydantic validates IDs and rejects unexpected fields. Known authorization, ownership, idempotency, review, and service errors use explicit HTTP codes.
- The default local server uses only the `SYN-MEMBER-001/002` fixtures and `SYN-REVIEWER-DEV`; these header values are development fixtures, **not authentication**. A production host must supply `identity_authenticator(member_id, session_id)` and `reviewer_authenticator(reviewer_id, member_id, session_id)` callbacks backed by its trusted identity provider. The API never accepts reviewer identity or approval in the JSON body.
- To start a new intake, send `X-CareFlow-Member-ID` and **omit** `X-CareFlow-Session-ID`; the API creates a registered session and returns its ID. Reuse that exact returned ID to continue the session. A guessed/unregistered value such as `SYN-SESSION-DEMO-001` is rejected with `404 session_not_found`; session IDs are not created from caller-provided strings.
- The workflow checkpoint and interaction memory persist under ignored `data/m8_runtime/`. Use the same authorized member/session identity after restarting the process to resume a pending review. Referral review is idempotent; clinical cases remain non-routable even if a reviewer approves.
- `data/evaluation/m8_golden_cases.json` contains 31 labeled, synthetic cases. `careflow.m8_evaluation` executes the API with in-process mock tools and tests policy answers with deterministic fake completions. It checks workflow state, authorization/isolation, idempotency, tool calls, policy evidence/refusal, missing eligibility, transient failure, timeout, and open-circuit fallback. The output explicitly separates these checks from retrieval-quality metrics; no retrieval quality is measured by M8.
- Install and run offline:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m careflow.m8_evaluation
```

- Start the local API (no provider key required):

```powershell
.\.venv\Scripts\python.exe -m uvicorn careflow.api:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal, submit a synthetic administrative request and retain the generated session ID:

```powershell
$headers = @{ "X-CareFlow-Member-ID" = "SYN-MEMBER-001" }
$body = @{
  request_id = "SYN-REQ-API-001"
  idempotency_key = "api-demo-001"
  message = "Please create a cardiology referral."
} | ConvertTo-Json
$start = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/intake -Headers $headers -ContentType "application/json" -Body $body
$start
```

The initial status is `awaiting_human`; eligibility is checked, but referral routing waits for review. In the local demo only, approve through the documented fixture reviewer:

```powershell
$reviewHeaders = @{
  "X-CareFlow-Member-ID" = "SYN-MEMBER-001"
  "X-CareFlow-Session-ID" = $start.session_id
  "X-CareFlow-Reviewer-ID" = "SYN-REVIEWER-DEV"
}
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/workflows/$($start.workflow_id)/review" -Headers $reviewHeaders -ContentType "application/json" -Body '{"decision":"approve"}'
```

The same session can be used after restarting the server because the checkpoint is local SQLite. Clinical requests still return `human_reviewed_no_automation` with `not_routed`, regardless of approval. Run all validation with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m careflow.m8_evaluation
.\.venv\Scripts\python.exe -c "from fastapi.testclient import TestClient; from careflow.api import create_app; app=create_app('data/m8_smoke_runtime'); c=TestClient(app); r=c.get('/health'); print(r.status_code, r.json()); c.close()"
git diff --check
```

- Limitations: development fixture headers are not production authentication; a host must bind identity and review callbacks to authenticated context and secure transport. The deterministic parser supports only a short list of synthetic specialties. Local SQLite checkpoints and process-local circuit state are single-host demonstration components. No real EHR, payer, clinical validation, production safety guarantee, or live-provider retrieval measurement is included. PDF retrieval-quality metrics remain the separate M4 evaluation.

## Quickstart

Requires Python **3.11+**.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Set the appropriate provider key in your local environment, e.g. `GROQ_API_KEY` for Groq. **Never commit `.env`**.

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-key"
python -m careflow.cli --model "groq/your-available-model-id" --message "I would like to schedule a cardiology follow-up. My insurance ID is SYN-PLAN-12."

# macOS/Linux
# export GROQ_API_KEY="your-key"
# python -m careflow.cli --model "groq/your-available-model-id" --message "..."
```

Model identifiers and structured-output support vary by provider. This starter requests JSON text in its messages and validates locally with Pydantic. Never assume a provider will follow the shape. Check actual availability in your account.

## Architecture (target, built incrementally)

```
Synthetic request -> Intake (M1)
                       -> Approved mock EHR / eligibility / co-pay tools (M2)
                       -> SQLite interaction memory (M3 Phase 1)
                       -> Qdrant policy index (M3 Phase 2)
                       -> Hybrid RAG + eval (M4)
                       -> LangGraph HITL workflow (M5)
                       -> Four agents + MCP mock services (M6)
                       -> Langfuse traces + fallback/retry/circuit breaker (M7)
                       -> Safety eval + FastAPI deploy package (M8)
```

## Capstone checkpoints (from problem statement)

| Milestone | Delivery | Status |
|---|---|---|
| M1 | LiteLLM + Pydantic structured patient intake | Starter included; live API smoke test pending |
| M2 | Mock EHR appointment, eligibility, copay, human flag tools | Typed in-memory foundation included; no real integrations |
| M3 | Session/long-term memory and Qdrant handbook index | Phases 1 and 2 included; RAG pending |
| M4 | Hybrid policy PDF RAG + rerank + groundedness baseline | Phases 1–4 implementation complete; provider-backed retrieval quality remains unmeasured |
| M5 | LangGraph checkpointed intake → eligibility → referral | Synthetic local workflow and restartable human checkpoint implemented; host authentication remains external |
| M6 | Intake, Insurance/Policy, Referral, Clinical Safety agents + MCP | Four roles and official local stdio MCP mock server implemented; production host authorization remains external |
| M7 | Langfuse + bounded retries, fallbacks, circuit breaker | Sanitized optional spans and bounded local resilience implemented; production supervision/reconciliation remains limited |
| M8 | Golden evaluation + clinical-advice guardrails + FastAPI | Implemented; 31 synthetic cases pass locally; production authentication and clinical validation remain out of scope |

## Data to prepare early

1. **Clinic corpus:** Fourteen fictional policy PDFs with intentionally different synthetic terms are included under `data/policies/`; `careflow/policy_corpus.py` remains the small M3 in-code corpus.
2. **Intake messages:** synthetic phone, email, web and SMS examples, including ambiguity and prohibited clinical-advice requests.
3. **Mock API tables:** synthetic patients, plans, eligibility, appointments and referrals; M2 currently uses small source-controlled fixtures in `careflow/mock_data.py`. Expand only with synthetic records; MITRE Synthea may supply synthetic base records subject to its license.
4. **Golden evaluation:** 31 cases in `data/evaluation/m8_golden_cases.json`, covering workflow, authorization, idempotency, clinical escalation, policy evidence, and service-failure fallback.

The course document mentions a companion generator command, `python generate.py --domain careflow`, but **that generator is not included in this starter**. Obtain the official generator and Data Sourcing Guide from the course LMS, or build a compatible alternative if permitted.

## GitHub submission readiness

- Commit source, tests, safe examples and README.
- Verify `.gitignore` excludes `.env`, `.venv`, patient data and keys.
- Show tests passing in the README or pull request.
- Add demo screenshots and evaluation results as milestones finish.
- Confirm the **actual current deadline, repository visibility and submission form** in IIT's LMS. The provided capstone sheet mentions a 15–16 August final weekend, which may belong to an earlier schedule; this document alone does not establish the current submission deadline.

## License / usage

Educational prototype. Before importing third-party data, review that dataset's license and permitted redistribution. No real PHI.
