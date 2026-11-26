from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientRequest(BaseModel):
    """Structured interpretation of *synthetic* clinic intake text."""

    model_config = ConfigDict(extra="forbid")
    request_type: str = Field(description="administrative request category")
    specialty: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    urgency_flag: bool = False
    insurance_id: str | None = None
    needs_human_review: bool = False
    reason_for_review: str | None = None

    @field_validator("insurance_id")
    @classmethod
    def normalize_insurance_id(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()
