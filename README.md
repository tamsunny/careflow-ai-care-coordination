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
| M4 | Hybrid policy PDF RAG + rerank + groundedness baseline | Pending |
| M5 | LangGraph checkpointed intake → eligibility → referral | Pending |
| M6 | Intake, Insurance/Policy, Referral, Clinical Safety agents + MCP | Pending |
| M7 | Langfuse + bounded retries, fallbacks, circuit breaker | Pending |
| M8 | 20-case golden evaluation + clinical-advice guardrails + FastAPI | Pending |

## Data to prepare early

1. **Clinic corpus:** The starter includes three fictional policy documents in `careflow/policy_corpus.py`; expand toward ~14 documents with intentionally different synthetic terms.
2. **Intake messages:** synthetic phone, email, web and SMS examples, including ambiguity and prohibited clinical-advice requests.
3. **Mock API tables:** synthetic patients, plans, eligibility, appointments and referrals; M2 currently uses small source-controlled fixtures in `careflow/mock_data.py`. Expand only with synthetic records; MITRE Synthea may supply synthetic base records subject to its license.
4. **Golden evaluation:** 20 cases, with strong clinical refusal/escalation coverage, including a fake clinician requesting dosing advice.

The course document mentions a companion generator command, `python generate.py --domain careflow`, but **that generator is not included in this starter**. Obtain the official generator and Data Sourcing Guide from the course LMS, or build a compatible alternative if permitted.

## GitHub submission readiness

- Commit source, tests, safe examples and README.
- Verify `.gitignore` excludes `.env`, `.venv`, patient data and keys.
- Show tests passing in the README or pull request.
- Add demo screenshots and evaluation results as milestones finish.
- Confirm the **actual current deadline, repository visibility and submission form** in IIT's LMS. The provided capstone sheet mentions a 15–16 August final weekend, which may belong to an earlier schedule; this document alone does not establish the current submission deadline.

## License / usage

Educational prototype. Before importing third-party data, review that dataset's license and permitted redistribution. No real PHI.
