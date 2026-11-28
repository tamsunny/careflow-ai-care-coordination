from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from careflow.agent import SchedulingInsuranceAgent
from careflow.tool_models import (
    AppointmentLookupInput,
    CopayCalculationInput,
    HumanReviewInput,
    InsuranceEligibilityInput,
)


def test_lookup_appointments_returns_fictional_matching_slots():
    agent = SchedulingInsuranceAgent()

    result = agent.lookup_appointments(
        AppointmentLookupInput(
            specialty="Cardiology",
            requested_date=date(2026, 12, 10),
        )
    )

    assert result.status == "available"
    assert len(result.appointments) == 1
    assert result.appointments[0].appointment_id == "SYN-APPT-101"
    assert result.appointments[0].specialty == "cardiology"


def test_lookup_appointments_distinguishes_unavailable_records():
    agent = SchedulingInsuranceAgent()

    result = agent.lookup_appointments(
        AppointmentLookupInput(specialty="neurology")
    )

    assert result.status == "not_found"
    assert result.appointments == []


def test_invalid_tool_input_is_rejected():
    with pytest.raises(ValidationError):
        AppointmentLookupInput(specialty=" ")
    with pytest.raises(ValidationError):
        CopayCalculationInput(
            plan_id="SYN-PLAN-12",
            service_category="specialist_visit",
            allowed_amount=Decimal("-1.00"),
        )
    with pytest.raises(ValidationError):
        CopayCalculationInput(
            plan_id="SYN-PLAN-12",
            service_category="specialist_visit",
            allowed_amount=Decimal("10.001"),
        )


def test_check_insurance_eligibility_returns_eligible_synthetic_record():
    agent = SchedulingInsuranceAgent()

    result = agent.check_insurance_eligibility(
        InsuranceEligibilityInput(insurance_id=" SYN-MEMBER-001 ")
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.plan_id == "SYN-PLAN-12"


def test_check_insurance_eligibility_distinguishes_unknown_and_ineligible():
    agent = SchedulingInsuranceAgent()

    unknown = agent.check_insurance_eligibility(
        InsuranceEligibilityInput(insurance_id="SYN-MEMBER-999")
    )
    inactive = agent.check_insurance_eligibility(
        InsuranceEligibilityInput(insurance_id="SYN-MEMBER-002")
    )

    assert unknown.status == "not_found"
    assert unknown.eligible is None
    assert inactive.status == "ineligible"
    assert inactive.eligible is False


def test_calculate_copay_uses_mock_rules_and_decimal_currency():
    agent = SchedulingInsuranceAgent()

    result = agent.calculate_copay(
        CopayCalculationInput(
            plan_id="SYN-PLAN-12",
            service_category="specialist_visit",
            allowed_amount=Decimal("120.00"),
        )
    )

    assert result.status == "calculated"
    assert result.flat_copay == Decimal("35.00")
    assert result.coinsurance_amount == Decimal("12.00")
    assert result.copay_amount == Decimal("47.00")
    assert result.currency == "USD"


def test_calculate_copay_rounds_coinsurance_half_up_to_cents():
    agent = SchedulingInsuranceAgent()

    result = agent.calculate_copay(
        CopayCalculationInput(
            plan_id="SYN-PLAN-12",
            service_category="primary_care",
            allowed_amount=Decimal("10.05"),
        )
    )

    assert result.coinsurance_amount == Decimal("1.01")
    assert result.copay_amount == Decimal("21.01")


def test_calculate_copay_reports_unknown_plan_without_calculating():
    agent = SchedulingInsuranceAgent()

    result = agent.calculate_copay(
        CopayCalculationInput(
            plan_id="SYN-PLAN-999",
            service_category="specialist_visit",
            allowed_amount=Decimal("120.00"),
        )
    )

    assert result.status == "plan_not_found"
    assert result.copay_amount is None


def test_flag_for_human_creates_auditable_pending_record():
    agent = SchedulingInsuranceAgent()

    record = agent.flag_for_human(
        HumanReviewInput(
            request_id="SYN-REQ-001",
            reason="Synthetic request needs staff follow-up",
            category="administrative",
        )
    )

    assert record.review_id.startswith("SYN-REVIEW-")
    assert record.request_id == "SYN-REQ-001"
    assert record.status == "pending"
    assert record.created_at.tzinfo is not None
    assert agent.review_records == (record,)
