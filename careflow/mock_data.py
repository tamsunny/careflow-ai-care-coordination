"""Fictional, deterministic records for the Milestone 2 prototype."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

from .tool_models import AppointmentSlot, ReferralTrackingRecord, ServiceCategory


@dataclass(frozen=True)
class MockMember:
    plan_id: str
    eligible: bool
    coverage_end: date


@dataclass(frozen=True)
class MockPlan:
    fixed_copays: tuple[tuple[ServiceCategory, Decimal], ...]
    coinsurance_rate: Decimal

    def fixed_copay_for(self, service_category: ServiceCategory) -> Decimal:
        return dict(self.fixed_copays)[service_category]


MOCK_APPOINTMENTS = (
    AppointmentSlot(
        appointment_id="SYN-APPT-101",
        specialty="cardiology",
        starts_at=datetime(2026, 12, 10, 9, 0, tzinfo=timezone.utc),
        location="Fictional North Clinic - Room A",
    ),
    AppointmentSlot(
        appointment_id="SYN-APPT-102",
        specialty="cardiology",
        starts_at=datetime(2026, 12, 12, 14, 30, tzinfo=timezone.utc),
        location="Fictional North Clinic - Room B",
    ),
    AppointmentSlot(
        appointment_id="SYN-APPT-201",
        specialty="dermatology",
        starts_at=datetime(2026, 12, 11, 11, 0, tzinfo=timezone.utc),
        location="Fictional South Clinic - Room C",
    ),
)

MOCK_MEMBERS = {
    "SYN-MEMBER-001": MockMember(
        plan_id="SYN-PLAN-12",
        eligible=True,
        coverage_end=date(2030, 12, 31),
    ),
    "SYN-MEMBER-002": MockMember(
        plan_id="SYN-PLAN-24",
        eligible=False,
        coverage_end=date(2025, 12, 31),
    ),
}

MOCK_PLANS = {
    "SYN-PLAN-12": MockPlan(
        fixed_copays=(
            ("primary_care", Decimal("20.00")),
            ("specialist_visit", Decimal("35.00")),
            ("urgent_care", Decimal("50.00")),
        ),
        coinsurance_rate=Decimal("0.10"),
    ),
    "SYN-PLAN-24": MockPlan(
        fixed_copays=(
            ("primary_care", Decimal("15.00")),
            ("specialist_visit", Decimal("30.00")),
            ("urgent_care", Decimal("45.00")),
        ),
        coinsurance_rate=Decimal("0.20"),
    ),
}

MOCK_REFERRAL_ROUTES = {
    "cardiology": "Fictional Cardiology Referral Desk",
    "dermatology": "Fictional Dermatology Referral Desk",
    "family_medicine": "Fictional Family Medicine Referral Desk",
    "pediatrics": "Fictional Pediatrics Referral Desk",
}

MOCK_REFERRALS = {
    "SYN-REF-TRACK-001": ReferralTrackingRecord(
        referral_id="SYN-REF-TRACK-001",
        request_id="SYN-REQ-TRACK-001",
        member_id="SYN-MEMBER-001",
        referral_status="pending",
        specialty="cardiology",
        destination=MOCK_REFERRAL_ROUTES["cardiology"],
    ),
    "SYN-REF-TRACK-002": ReferralTrackingRecord(
        referral_id="SYN-REF-TRACK-002",
        request_id="SYN-REQ-TRACK-002",
        member_id="SYN-MEMBER-002",
        referral_status="completed",
        specialty="dermatology",
        destination=MOCK_REFERRAL_ROUTES["dermatology"],
    ),
}
