"""Milestone 1: LiteLLM wrapper + local Pydantic validation.

No patient records should enter this prototype. Only synthetic examples.
"""
import json
import re
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
CLINICAL_ADVICE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:dose|dosage)\b",
        r"\b(?:diagnos(?:e|is|ing)|prescrib(?:e|ed|ing|tion))\b",
        r"\bwhat\s+(?:medication|medicine|drug)\b",
        r"\b(?:medical|treatment)\s+advice\b",
        r"\b(?:should|can|could|may)\s+i\s+(?:take|start|stop|increase|decrease|change|mix)\b",
        r"\bwhat\s+can\s+i\s+(?:take|use)\b",
        r"\bis\s+it\s+safe\s+to\s+(?:take|use|stop|start)\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\b(?:do|could|might)\s+i\s+have\b",
        r"\bwhat\s+does\s+.+\s+mean\b",
        r"\bwhat(?:'s| is)\s+wrong\s+with\s+me\b",
        r"\bwhat\s+could\s+be\s+causing\b",
        r"\bhow\s+(?:do|can|should)\s+i\s+treat\b",
    )
)


def is_clinical_advice_request(message: str) -> bool:
    """Conservatively identify common clinical-advice requests for escalation."""
    normalized = message.casefold()
    return any(marker in normalized for marker in CLINICAL_ADVICE_MARKERS) or any(
        pattern.search(normalized) for pattern in CLINICAL_ADVICE_PATTERNS
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
    if is_clinical_advice_request(message):
        parsed.needs_human_review = True
        parsed.reason_for_review = "Clinical advice requested; hand off to qualified staff"
    return parsed
