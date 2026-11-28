from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from careflow.agent import SchedulingInsuranceAgent
from careflow.resilience import (
    CircuitBreaker,
    CircuitState,
    ResilienceConfiguration,
    ResilientMockServices,
    ServiceUnavailableError,
    TransientServiceError,
)
from careflow.tool_models import (
    InsuranceEligibilityInput,
    ReferralRoutingInput,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_retry_transient_read_with_bounded_exponential_backoff():
    class FlakyTools(SchedulingInsuranceAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def check_insurance_eligibility(self, request):
            self.calls += 1
            if self.calls < 3:
                raise TransientServiceError("synthetic transient")
            return super().check_insurance_eligibility(request)

    delays = []
    tools = FlakyTools()
    resilient = ResilientMockServices(
        tools,
        ResilienceConfiguration(
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.02,
            timeout_seconds=1,
        ),
        sleep=delays.append,
    )
    try:
        result = resilient.check_insurance_eligibility(
            InsuranceEligibilityInput(insurance_id="SYN-MEMBER-001")
        )
    finally:
        resilient.close()

    assert result.status == "eligible"
    assert tools.calls == 3
    assert delays == [0.01, 0.02]


@pytest.mark.parametrize("failure", [PermissionError, ValueError])
def test_validation_and_authorization_failures_are_not_retried(failure):
    class InvalidTools(SchedulingInsuranceAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def check_insurance_eligibility(self, request):
            self.calls += 1
            raise failure("nontransient")

    tools = InvalidTools()
    resilient = ResilientMockServices(
        tools,
        ResilienceConfiguration(max_attempts=4, timeout_seconds=1),
        sleep=lambda _delay: pytest.fail("nontransient errors must not retry"),
    )
    try:
        with pytest.raises(failure):
            resilient.check_insurance_eligibility(
                InsuranceEligibilityInput(insurance_id="SYN-MEMBER-001")
            )
    finally:
        resilient.close()
    assert tools.calls == 1


def test_exhausted_transient_failures_open_breaker_and_block_calls():
    class AlwaysDown(SchedulingInsuranceAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def check_insurance_eligibility(self, request):
            self.calls += 1
            raise ConnectionError("synthetic outage")

    tools = AlwaysDown()
    resilient = ResilientMockServices(
        tools,
        ResilienceConfiguration(
            max_attempts=1,
            failure_threshold=1,
            timeout_seconds=1,
        ),
    )
    try:
        request = InsuranceEligibilityInput(insurance_id="SYN-MEMBER-001")
        with pytest.raises(ServiceUnavailableError):
            resilient.check_insurance_eligibility(request)
        assert resilient.circuit_state == CircuitState.OPEN
        with pytest.raises(ServiceUnavailableError) as error:
            resilient.check_insurance_eligibility(request)
        assert error.value.reason == "circuit_open"
        assert tools.calls == 1
    finally:
        resilient.close()


def test_service_timeout_is_bounded_and_opens_breaker():
    release_service = Event()

    class SlowTools(SchedulingInsuranceAgent):
        def check_insurance_eligibility(self, request):
            release_service.wait(timeout=2)
            return super().check_insurance_eligibility(request)

    resilient = ResilientMockServices(
        SlowTools(),
        ResilienceConfiguration(
            max_attempts=1,
            failure_threshold=1,
            timeout_seconds=0.01,
            max_concurrent_calls=1,
        ),
    )
    try:
        with pytest.raises(ServiceUnavailableError) as error:
            resilient.check_insurance_eligibility(
                InsuranceEligibilityInput(insurance_id="SYN-MEMBER-001")
            )
        assert error.value.reason == "timeout"
        assert resilient.circuit_state == CircuitState.OPEN
    finally:
        release_service.set()
        resilient.close()


def test_circuit_half_open_allows_one_probe_then_closes_on_success():
    clock = FakeClock()
    breaker = CircuitBreaker(1, 2, clock=clock)

    assert breaker.acquire()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert not breaker.acquire()

    clock.advance(2)
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.acquire()
    assert not breaker.acquire()
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_concurrent_safe_reads_complete_without_duplicate_state_changes():
    class CountingTools(SchedulingInsuranceAgent):
        def __init__(self):
            super().__init__()
            self.lock = Lock()
            self.calls = 0

        def check_insurance_eligibility(self, request):
            with self.lock:
                self.calls += 1
            return super().check_insurance_eligibility(request)

    tools = CountingTools()
    resilient = ResilientMockServices(
        tools,
        ResilienceConfiguration(
            max_attempts=2,
            max_concurrent_calls=4,
            timeout_seconds=2,
        ),
    )
    request = InsuranceEligibilityInput(insurance_id="SYN-MEMBER-001")
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _index: resilient.check_insurance_eligibility(request),
                    range(4),
                )
            )
    finally:
        resilient.close()

    assert len(results) == 4
    assert all(result.status == "eligible" for result in results)
    assert tools.calls == 4


def test_non_idempotent_referral_mutation_is_never_retried():
    class FailingMutation(SchedulingInsuranceAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def route_referral(self, request):
            self.calls += 1
            raise TransientServiceError("uncertain mutation response")

    tools = FailingMutation()
    resilient = ResilientMockServices(
        tools,
        ResilienceConfiguration(max_attempts=5, timeout_seconds=1),
        sleep=lambda _delay: pytest.fail("mutations must never be retried"),
    )
    try:
        with pytest.raises(ServiceUnavailableError):
            resilient.route_referral(
                ReferralRoutingInput(
                    request_id="SYN-REQ-ONCE-001",
                    specialty="cardiology",
                )
            )
    finally:
        resilient.close()
    assert tools.calls == 1
