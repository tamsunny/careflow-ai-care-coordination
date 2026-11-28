"""Direct, deterministic execution of synthetic Milestone 2 tools.

This agent does not call an LLM or execute model-generated tool requests.
"""

import hashlib
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from uuid import uuid4

from .mock_data import (
    MOCK_APPOINTMENTS,
    MOCK_MEMBERS,
    MOCK_PLANS,
    MOCK_REFERRALS,
    MOCK_REFERRAL_ROUTES,
)
from .tool_models import (
    AppointmentLookupInput,
    AppointmentLookupOutput,
    CopayCalculationInput,
    CopayCalculationOutput,
    HumanReviewInput,
    HumanReviewRecord,
    InsuranceEligibilityInput,
    InsuranceEligibilityOutput,
    ReferralLookupInput,
    ReferralRoutingInput,
    ReferralRoutingOutput,
    ReferralTrackingRecord,
)


class SchedulingInsuranceAgent:
    """Expose typed mock scheduling, insurance, copay, and review tools."""

    def __init__(self) -> None:
        self._review_records: list[HumanReviewRecord] = []

    @property
    def review_records(self) -> tuple[HumanReviewRecord, ...]:
        """Return copies of the in-memory audit records."""
        return tuple(record.model_copy(deep=True) for record in self._review_records)

    def lookup_appointments(
        self, request: AppointmentLookupInput
    ) -> AppointmentLookupOutput:
        specialty = request.specialty.casefold()
        appointments = [
            appointment.model_copy(deep=True)
            for appointment in MOCK_APPOINTMENTS
            if appointment.specialty.casefold() == specialty
            and (
                request.requested_date is None
                or appointment.starts_at.date() == request.requested_date
            )
        ]
        return AppointmentLookupOutput(
            status="available" if appointments else "not_found",
            appointments=appointments,
        )

    def check_insurance_eligibility(
        self, request: InsuranceEligibilityInput
    ) -> InsuranceEligibilityOutput:
        member = MOCK_MEMBERS.get(request.insurance_id)
        if member is None:
            return InsuranceEligibilityOutput(
                status="not_found",
                insurance_id=request.insurance_id,
                eligible=None,
                plan_id=None,
                coverage_end=None,
            )

        return InsuranceEligibilityOutput(
            status="eligible" if member.eligible else "ineligible",
            insurance_id=request.insurance_id,
            eligible=member.eligible,
            plan_id=member.plan_id,
            coverage_end=member.coverage_end,
        )

    def route_referral(
        self, request: ReferralRoutingInput
    ) -> ReferralRoutingOutput:
        """Return a deterministic fictional route without contacting a clinic."""
        specialty = request.specialty.casefold()
        destination = MOCK_REFERRAL_ROUTES.get(specialty)
        if destination is None:
            return ReferralRoutingOutput(
                status="not_found",
                request_id=request.request_id,
                referral_id=None,
                specialty=specialty,
                destination=None,
            )
        referral_suffix = hashlib.sha256(
            request.request_id.encode("utf-8")
        ).hexdigest()[:12].upper()
        return ReferralRoutingOutput(
            status="routed",
            request_id=request.request_id,
            referral_id=f"SYN-REF-{referral_suffix}",
            specialty=specialty,
            destination=destination,
        )

    def lookup_referral(
        self,
        request: ReferralLookupInput,
        trusted_member_id: str,
    ) -> ReferralTrackingRecord | None:
        if trusted_member_id not in MOCK_MEMBERS:
            raise ValueError("A supported trusted synthetic member is required")
        record = MOCK_REFERRALS.get(request.referral_id)
        if record is None or record.member_id != trusted_member_id:
            return None
        return record.model_copy(deep=True)

    def calculate_copay(
        self, request: CopayCalculationInput
    ) -> CopayCalculationOutput:
        plan = MOCK_PLANS.get(request.plan_id)
        if plan is None:
            return CopayCalculationOutput(
                status="plan_not_found",
                plan_id=request.plan_id,
                service_category=request.service_category,
                allowed_amount=request.allowed_amount,
                flat_copay=None,
                coinsurance_amount=None,
                copay_amount=None,
            )

        flat_copay = plan.fixed_copay_for(request.service_category)
        coinsurance_amount = (request.allowed_amount * plan.coinsurance_rate).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        copay_amount = (flat_copay + coinsurance_amount).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return CopayCalculationOutput(
            status="calculated",
            plan_id=request.plan_id,
            service_category=request.service_category,
            allowed_amount=request.allowed_amount,
            flat_copay=flat_copay,
            coinsurance_amount=coinsurance_amount,
            copay_amount=copay_amount,
        )

    def flag_for_human(self, request: HumanReviewInput) -> HumanReviewRecord:
        record = HumanReviewRecord(
            review_id=f"SYN-REVIEW-{uuid4().hex}",
            request_id=request.request_id,
            reason=request.reason,
            category=request.category,
            created_at=datetime.now(timezone.utc),
        )
        self._review_records.append(record.model_copy(deep=True))
        return record
