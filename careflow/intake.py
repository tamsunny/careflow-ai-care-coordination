"""Milestone 1: LiteLLM wrapper + local Pydantic validation.

No patient records should enter this prototype. Only synthetic examples.
"""
import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from .models import PatientRequest

SYSTEM = """You parse SYNTHETIC outpatient administrative intake messages.
Return only one JSON object with keys:
request_type (string), specialty (string or null), symptoms (list of strings),
urgency_flag (boolean), insurance_id (string or null), needs_human_review
(boolean), reason_for_review (string or null).
Extract only what was provided; do not invent IDs or diagnoses.
If the user seeks diagnosis, medication dosing or treatment advice, set
needs_human_review=true. This service must not supply clinical advice.
If information is missing, use null/empty list and human review if needed.
"""

# Demo-only keywords: not exhaustive, clinically validated, or sufficient for triage.
CLINICAL_ADVICE_MARKERS = (
    "diagnose", "diagnosis", "what disease", "what medication",
    "what dose", "dosage", "dose should", "prescribe", "treatment advice",
)


class IntakeParsingError(ValueError):
    """Output could not be parsed into a validated PatientRequest."""


def parse_patient_message(
    message: str,
    model: str,
    completion_fn: Callable[..., Any] | None = None,
) -> PatientRequest:
    if not message.strip():
        raise ValueError("Intake message must not be empty")
    if completion_fn is None:
        from litellm import completion
        completion_fn = completion

    response = completion_fn(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content
    if not isinstance(raw, str):
        raise IntakeParsingError("LLM returned no text")
    try:
        parsed = PatientRequest.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise IntakeParsingError("LLM output failed JSON/schema validation") from exc

    # Illustrative defense in depth: override model refusal classification for
    # explicitly clinical-advice requests. Not a comprehensive clinical guardrail.
    if any(marker in message.lower() for marker in CLINICAL_ADVICE_MARKERS):
        parsed.needs_human_review = True
        parsed.reason_for_review = "Clinical advice requested; hand off to qualified staff"
    return parsed
