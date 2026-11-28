"""Typed contracts for the synthetic Milestone 2 coordination tools."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppointmentLookupInput(ToolInput):
    specialty: str = Field(min_length=1, max_length=80)
    requested_date: date | None = None


class AppointmentSlot(ToolOutput):
    appointment_id: str
    specialty: str
    starts_at: datetime
    location: str


class AppointmentLookupOutput(ToolOutput):
    status: Literal["available", "not_found"]
    appointments: list[AppointmentSlot]


class InsuranceEligibilityInput(ToolInput):
    insurance_id: str = Field(min_length=1, max_length=100)


class InsuranceEligibilityOutput(ToolOutput):
    status: Literal["eligible", "ineligible", "not_found"]
    insurance_id: str
    eligible: bool | None
    plan_id: str | None
    coverage_end: date | None


class ReferralRoutingInput(ToolInput):
    request_id: str = Field(pattern=r"^SYN-REQ-[A-Z0-9-]+$")
    specialty: str = Field(min_length=1, max_length=80)


class ReferralRoutingOutput(ToolOutput):
    status: Literal["routed", "not_found"]
    request_id: str
    referral_id: str | None
    specialty: str
    destination: str | None


class ReferralLookupInput(ToolInput):
    referral_id: str = Field(pattern=r"^SYN-REF-[A-Z0-9-]+$")


class ReferralTrackingRecord(ToolOutput):
    referral_id: str = Field(pattern=r"^SYN-REF-[A-Z0-9-]+$")
    request_id: str = Field(pattern=r"^SYN-REQ-[A-Z0-9-]+$")
    member_id: str = Field(pattern=r"^SYN-MEMBER-[A-Z0-9-]+$")
    referral_status: Literal["pending", "scheduled", "completed", "cancelled"]
    specialty: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=120)


ServiceCategory = Literal["primary_care", "specialist_visit", "urgent_care"]


class CopayCalculationInput(ToolInput):
    plan_id: str = Field(min_length=1, max_length=100)
    service_category: ServiceCategory
    allowed_amount: Decimal = Field(
        ge=0, max_digits=12, decimal_places=2
    )


class CopayCalculationOutput(ToolOutput):
    status: Literal["calculated", "plan_not_found"]
    plan_id: str
    service_category: ServiceCategory
    allowed_amount: Decimal
    flat_copay: Decimal | None
    coinsurance_amount: Decimal | None
    copay_amount: Decimal | None
    currency: Literal["USD"] = "USD"


ReviewCategory = Literal["administrative", "safety_concern", "other"]


class HumanReviewInput(ToolInput):
    request_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)
    category: ReviewCategory


class HumanReviewRecord(ToolOutput):
    review_id: str
    request_id: str
    reason: str
    category: ReviewCategory
    status: Literal["pending"] = "pending"
    created_at: datetime
