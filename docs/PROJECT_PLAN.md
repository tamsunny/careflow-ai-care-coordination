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

## Four mandatory datasets (begin early)
- `careflow/policy_corpus.py`: three small synthetic policy documents; expand toward the target handbook corpus as needed.
- `data/intake/`: diverse synthetic intake records (starter examples only).
- `data/mock_api/`: synthetic patients, plans, appointments, referrals (M2 source fixtures are in `careflow/mock_data.py`; expanded dataset not yet created).
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
