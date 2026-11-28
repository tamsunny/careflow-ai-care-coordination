"""Official MCP stdio server exposing allowlisted synthetic CareFlow tools."""

import argparse
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict

from .agent import SchedulingInsuranceAgent
from .agents import ReferralTrackingAgent
from .mock_data import MOCK_MEMBERS
from .resilience import (
    ResilienceConfiguration,
    ResilientMockServices,
    ServiceUnavailableError,
)
from .tool_models import (
    AppointmentLookupInput,
    CopayCalculationInput,
    HumanReviewInput,
    InsuranceEligibilityInput,
    InsuranceEligibilityOutput,
    ReferralLookupInput,
    ReferralRoutingInput,
    ServiceCategory,
)

MCP_TOOL_ALLOWLIST = frozenset(
    {
        "lookup_appointments",
        "check_insurance_eligibility",
        "calculate_copay",
        "route_referral",
        "lookup_referral",
        "flag_for_human",
    }
)
ActionAuthorizer = Callable[[str, str], bool]


class EmptyToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolAuthorizationError(PermissionError):
    """The host did not authorize a state-changing mock tool."""


class MCPToolNotAllowedError(ValueError):
    """The requested tool is not in the application allowlist."""


class SyntheticMCPToolService:
    """Bind MCP tool calls to a host-trusted synthetic member identity."""

    def __init__(
        self,
        trusted_member_id: str,
        *,
        authorize_action: ActionAuthorizer | None = None,
        tool_executor: SchedulingInsuranceAgent | None = None,
        resilience_config: ResilienceConfiguration | None = None,
    ) -> None:
        if trusted_member_id not in MOCK_MEMBERS:
            raise ValueError("A supported trusted synthetic member is required")
        if authorize_action is not None and not callable(authorize_action):
            raise ValueError("authorize_action must be callable")
        self.trusted_member_id = trusted_member_id
        self._authorize_action = authorize_action or (lambda _action, _member: False)
        self._raw_tool_executor = tool_executor or SchedulingInsuranceAgent()
        self._tool_executor = ResilientMockServices(
            self._raw_tool_executor,
            resilience_config,
        )
        self._referral_tracker = ReferralTrackingAgent(
            trusted_member_id,
            self._tool_executor,
        )
        self._dispatch = {
            "lookup_appointments": self._lookup_appointments,
            "check_insurance_eligibility": self._check_eligibility,
            "calculate_copay": self._calculate_copay,
            "route_referral": self._route_referral,
            "lookup_referral": self._lookup_referral,
            "flag_for_human": self._flag_for_human,
        }

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if name not in MCP_TOOL_ALLOWLIST:
            raise MCPToolNotAllowedError("Tool is not in the CareFlow allowlist")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        try:
            return self._dispatch[name](arguments)
        except ServiceUnavailableError as exc:
            return {
                "status": "unavailable",
                "operation": exc.operation,
                "reason": exc.reason,
            }

    def close(self) -> None:
        self._tool_executor.close()

    def _lookup_appointments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = AppointmentLookupInput.model_validate(arguments)
        result = self._tool_executor.lookup_appointments(request)
        return result.model_dump(mode="json")

    def _check_eligibility(self, arguments: dict[str, Any]) -> dict[str, Any]:
        EmptyToolInput.model_validate(arguments)
        result = self._tool_executor.check_insurance_eligibility(
            InsuranceEligibilityInput(insurance_id=self.trusted_member_id)
        )
        return InsuranceEligibilityOutput.model_validate(result).model_dump(
            mode="json"
        )

    def _calculate_copay(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = CopayCalculationInput.model_validate(arguments)
        result = self._tool_executor.calculate_copay(request)
        return result.model_dump(mode="json")

    def _route_referral(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = ReferralRoutingInput.model_validate(arguments)
        self._require_authorization("route_referral")
        result = self._referral_tracker.route_referral(request)
        return result.model_dump(mode="json")

    def _lookup_referral(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = ReferralLookupInput.model_validate(arguments)
        record = self._tool_executor.lookup_referral(
            request,
            self.trusted_member_id,
        )
        if record is None:
            return {
                "status": "not_found",
                "referral_id": request.referral_id,
            }
        return {
            "status": "found",
            "record": record.model_dump(mode="json"),
        }

    def _flag_for_human(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = HumanReviewInput.model_validate(arguments)
        self._require_authorization("flag_for_human")
        result = self._tool_executor.flag_for_human(request)
        return result.model_dump(mode="json")

    def _require_authorization(self, action: str) -> None:
        try:
            authorized = self._authorize_action(action, self.trusted_member_id)
        except Exception:
            authorized = False
        if authorized is not True:
            raise ToolAuthorizationError(
                f"Host authorization is required for {action}"
            )


def create_mcp_server(
    service: SyntheticMCPToolService,
) -> FastMCP:
    """Create an MCP protocol server with validated, allowlisted tool schemas."""
    server = FastMCP(
        name="CareFlow Synthetic Mock Tools",
        instructions=(
            "Fictional local scheduling, eligibility and referral tools only. "
            "Never provide clinical advice. State-changing tools require "
            "application-side authorization."
        ),
    )

    @server.tool(description="Look up available fictional appointment slots.")
    def lookup_appointments(
        specialty: str,
        requested_date: date | None = None,
    ) -> dict[str, Any]:
        return service.call_tool(
            "lookup_appointments",
            {"specialty": specialty, "requested_date": requested_date},
        )

    @server.tool(
        description=(
            "Check eligibility for the host-bound synthetic member; "
            "the caller cannot choose another member."
        )
    )
    def check_insurance_eligibility() -> dict[str, Any]:
        return service.call_tool("check_insurance_eligibility", {})

    @server.tool(description="Calculate a fictional co-pay; no transaction occurs.")
    def calculate_copay(
        plan_id: str,
        service_category: ServiceCategory,
        allowed_amount: Decimal,
    ) -> dict[str, Any]:
        return service.call_tool(
            "calculate_copay",
            {
                "plan_id": plan_id,
                "service_category": service_category,
                "allowed_amount": allowed_amount,
            },
        )

    @server.tool(
        description=(
            "Create a deterministic synthetic referral route. "
            "Application authorization is required."
        )
    )
    def route_referral(request_id: str, specialty: str) -> dict[str, Any]:
        return service.call_tool(
            "route_referral",
            {"request_id": request_id, "specialty": specialty},
        )

    @server.tool(
        description="Read a synthetic referral record owned by the host member."
    )
    def lookup_referral(referral_id: str) -> dict[str, Any]:
        return service.call_tool("lookup_referral", {"referral_id": referral_id})

    @server.tool(
        description=(
            "Create a local synthetic human-review record. "
            "Application authorization is required."
        )
    )
    def flag_for_human(
        request_id: str,
        reason: str,
        category: str,
    ) -> dict[str, Any]:
        return service.call_tool(
            "flag_for_human",
            {
                "request_id": request_id,
                "reason": reason,
                "category": category,
            },
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local CareFlow synthetic MCP stdio server."
    )
    parser.add_argument(
        "--member-id",
        choices=tuple(MOCK_MEMBERS),
        default="SYN-MEMBER-001",
        help="Synthetic member identity bound by the launching host process.",
    )
    parser.add_argument(
        "--allow-synthetic-mutations",
        action="store_true",
        help="Enable mocked referral/review actions for this local demo process.",
    )
    args = parser.parse_args()
    service = SyntheticMCPToolService(
        args.member_id,
        authorize_action=(
            lambda _action, _member: True
            if args.allow_synthetic_mutations
            else False
        ),
    )
    create_mcp_server(service).run(transport="stdio")


if __name__ == "__main__":
    main()
