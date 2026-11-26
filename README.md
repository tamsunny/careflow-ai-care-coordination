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
                       -> Memory + Qdrant policies (M3)
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
| M2 | Mock EHR appointment, eligibility, copay, human flag tools | Pending |
| M3 | Session/long-term memory and Qdrant handbook index | Pending |
| M4 | Hybrid policy PDF RAG + rerank + groundedness baseline | Pending |
| M5 | LangGraph checkpointed intake → eligibility → referral | Pending |
| M6 | Intake, Insurance/Policy, Referral, Clinical Safety agents + MCP | Pending |
| M7 | Langfuse + bounded retries, fallbacks, circuit breaker | Pending |
| M8 | 20-case golden evaluation + clinical-advice guardrails + FastAPI | Pending |

## Data to prepare early

1. **Clinic corpus:** ~14 fictional policy documents with intentionally different per-specialty numbers.
2. **Intake messages:** synthetic phone, email, web and SMS examples, including ambiguity and prohibited clinical-advice requests.
3. **Mock API tables:** synthetic patients, plans, eligibility, appointments and referrals; MITRE Synthea may supply synthetic base records.
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
