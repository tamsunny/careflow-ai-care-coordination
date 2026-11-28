"""Bounded reliability controls for local synthetic service adapters."""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from enum import Enum
from threading import BoundedSemaphore, Lock
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResilienceConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=2, ge=1, le=5)
    base_delay_seconds: float = Field(default=0.05, ge=0, le=5)
    max_delay_seconds: float = Field(default=0.5, ge=0, le=30)
    timeout_seconds: float = Field(default=2.0, gt=0, le=60)
    failure_threshold: int = Field(default=3, ge=1, le=20)
    circuit_reset_seconds: float = Field(default=5.0, gt=0, le=300)
    max_concurrent_calls: int = Field(default=8, ge=1, le=64)

    @model_validator(mode="after")
    def validate_delay_bounds(self) -> "ResilienceConfiguration":
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        return self


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ServiceUnavailableError(RuntimeError):
    """A synthetic dependency timed out, failed transiently, or is circuit-open."""

    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"Synthetic service unavailable: {operation} ({reason})")


class TransientServiceError(ConnectionError):
    """Explicit marker for a retryable synthetic dependency failure."""


class CircuitBreaker:
    """Thread-safe closed/open/half-open breaker with one half-open probe."""

    def __init__(
        self,
        failure_threshold: int,
        reset_timeout: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or reset_timeout <= 0:
            raise ValueError("Circuit threshold and reset timeout must be positive")
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._clock = clock
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._refresh_open_state()
            return self._state

    def acquire(self) -> bool:
        with self._lock:
            self._refresh_open_state()
            if self._state == CircuitState.OPEN:
                return False
            if self._state == CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._open()
                return
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._open()

    def record_neutral(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._open()

    def _refresh_open_state(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self._reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False


class ResilientMockServices:
    """Wrap existing synthetic tools with timeouts, retries, and a circuit.

    Only explicitly transient reads (eligibility and appointment lookup) retry.
    Referral routing and human-review creation are mutations and execute at
    most once, even when the caller supplied an idempotency-like request ID.
    """

    def __init__(
        self,
        service: Any,
        config: ResilienceConfiguration | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.config = config or ResilienceConfiguration()
        self._service = service
        self._sleep = sleep
        self._executor = executor or ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_calls,
            thread_name_prefix="careflow-mock-service",
        )
        self._owns_executor = executor is None
        self._slots = BoundedSemaphore(self.config.max_concurrent_calls)
        self._breaker = CircuitBreaker(
            self.config.failure_threshold,
            self.config.circuit_reset_seconds,
            clock=clock,
        )

    @property
    def circuit_state(self) -> CircuitState:
        return self._breaker.state

    def lookup_appointments(self, request: Any) -> Any:
        return self._call(
            "lookup_appointments",
            lambda: self._service.lookup_appointments(request),
            retry_safe=True,
        )

    def check_insurance_eligibility(self, request: Any) -> Any:
        return self._call(
            "check_insurance_eligibility",
            lambda: self._service.check_insurance_eligibility(request),
            retry_safe=True,
        )

    def calculate_copay(self, request: Any) -> Any:
        return self._call(
            "calculate_copay",
            lambda: self._service.calculate_copay(request),
            retry_safe=True,
        )

    def route_referral(self, request: Any) -> Any:
        return self._call(
            "route_referral",
            lambda: self._service.route_referral(request),
            retry_safe=False,
        )

    def flag_for_human(self, request: Any) -> Any:
        return self._call(
            "flag_for_human",
            lambda: self._service.flag_for_human(request),
            retry_safe=False,
        )

    def lookup_referral(self, request: Any, trusted_member_id: str) -> Any:
        return self._call(
            "lookup_referral",
            lambda: self._service.lookup_referral(request, trusted_member_id),
            retry_safe=True,
        )

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _call(
        self,
        operation: str,
        call: Callable[[], Any],
        *,
        retry_safe: bool,
    ) -> Any:
        if not self._breaker.acquire():
            raise ServiceUnavailableError(operation, "circuit_open")
        attempts = self.config.max_attempts if retry_safe else 1
        for attempt in range(attempts):
            if not self._slots.acquire(timeout=self.config.timeout_seconds):
                self._breaker.record_failure()
                raise ServiceUnavailableError(operation, "capacity_timeout")
            try:
                future = self._executor.submit(call)
            except Exception:
                self._slots.release()
                self._breaker.record_neutral()
                raise
            future.add_done_callback(lambda _future: self._slots.release())
            try:
                result = future.result(timeout=self.config.timeout_seconds)
            except FutureTimeout:
                future.cancel()
                error: Exception = TimeoutError("service_timeout")
            except Exception as exc:
                error = exc
            else:
                self._breaker.record_success()
                return result

            if not _is_transient(error):
                self._breaker.record_neutral()
                raise error
            if attempt + 1 >= attempts:
                self._breaker.record_failure()
                reason = "timeout" if isinstance(error, TimeoutError) else "transient_failure"
                raise ServiceUnavailableError(operation, reason) from None
            delay = min(
                self.config.base_delay_seconds * (2**attempt),
                self.config.max_delay_seconds,
            )
            self._sleep(delay)
        raise ServiceUnavailableError(operation, "retry_exhausted")

    def __enter__(self) -> "ResilientMockServices":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _is_transient(error: Exception) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError))
