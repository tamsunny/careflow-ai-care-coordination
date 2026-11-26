# CareFlow build plan

## Definition of done for M1
- [x] Code contains a Pydantic v2 `PatientRequest` model.
- [x] LiteLLM-based provider-independent call accepts a `model` identifier.
- [x] JSON is validated locally; malformed responses generate explicit exceptions.
- [x] Synthetic sample inputs and mocked tests exist.
- [ ] User installs dependencies and runs tests locally.
- [ ] User confirms one live provider call using synthetic inputs.
- [ ] Instructor/TA verifies M1 against official lab checklist.

## Four mandatory datasets (begin early)
- `docs/policies/`: ~14 synthetic clinic policy documents (not yet created).
- `data/intake/`: diverse synthetic intake records (starter examples only).
- `data/mock_api/`: synthetic patients, plans, appointments, referrals (not yet created).
- `eval/`: 20 golden cases incl. diagnosis/dosing refusal (not yet created).

## Architecture decisions
- Python 3.11+, Pydantic 2, LiteLLM, pytest for M1.
- Add mock tools for M2, Qdrant for M3, RAG and reranking for M4, LangGraph for M5, MCP for M6, Langfuse for M7, FastAPI for M8.
- Never use real PHI, real payer IDs, real clinic secrets or actual clinical treatment recommendations.
- Keep each milestone independently demonstrable and testable.

## Human-in-the-loop and safety
- Clinical questions are escalated, not answered with diagnoses or dosages.
- Model-only urgency categorization is not a safe clinical decision; decisions require qualified staff.
- M1 keyword marker is for demo only, not a safety guarantee. Production-style fail-closed escalation and adversarial evaluations belong in M8.
