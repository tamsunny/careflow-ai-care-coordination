import asyncio

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from careflow.agent import SchedulingInsuranceAgent
from careflow.mcp_server import (
    MCP_TOOL_ALLOWLIST,
    MCPToolNotAllowedError,
    SyntheticMCPToolService,
    ToolAuthorizationError,
    create_mcp_server,
)
from careflow.resilience import ResilienceConfiguration

MEMBER_ONE = "SYN-MEMBER-001"


def test_mcp_service_validates_allowlist_and_host_bound_identity():
    service = SyntheticMCPToolService(MEMBER_ONE)

    appointments = service.call_tool(
        "lookup_appointments",
        {"specialty": "cardiology"},
    )
    eligibility = service.call_tool("check_insurance_eligibility", {})

    assert appointments["status"] == "available"
    assert eligibility["insurance_id"] == MEMBER_ONE
    assert set(MCP_TOOL_ALLOWLIST) == {
        "lookup_appointments",
        "check_insurance_eligibility",
        "calculate_copay",
        "route_referral",
        "lookup_referral",
        "flag_for_human",
    }
    with pytest.raises(ValidationError):
        service.call_tool(
            "lookup_appointments",
            {"specialty": " ", "unexpected": True},
        )
    with pytest.raises(ValidationError):
        service.call_tool(
            "check_insurance_eligibility",
            {"insurance_id": "SYN-MEMBER-002"},
        )
    with pytest.raises(MCPToolNotAllowedError):
        service.call_tool("delete_patient", {})


def test_mutating_tools_require_host_authorization_and_repeat_deterministically():
    denied = SyntheticMCPToolService(MEMBER_ONE)
    route_args = {
        "request_id": "SYN-REQ-MCP-ROUTE-001",
        "specialty": "cardiology",
    }
    review_args = {
        "request_id": "SYN-REQ-MCP-ROUTE-001",
        "reason": "Synthetic test review",
        "category": "administrative",
    }

    with pytest.raises(ToolAuthorizationError):
        denied.call_tool("route_referral", route_args)
    with pytest.raises(ToolAuthorizationError):
        denied.call_tool("flag_for_human", review_args)

    authorized = SyntheticMCPToolService(
        MEMBER_ONE,
        authorize_action=lambda action, member: (
            action in {"route_referral", "flag_for_human"} and member == MEMBER_ONE
        ),
    )
    first = authorized.call_tool("route_referral", route_args)
    retry = authorized.call_tool("route_referral", route_args)
    review = authorized.call_tool("flag_for_human", review_args)

    assert first == retry
    assert first["status"] == "routed"
    assert review["status"] == "pending"


def test_referral_lookup_is_scoped_to_the_host_member():
    member_one = SyntheticMCPToolService(MEMBER_ONE)
    member_two = SyntheticMCPToolService("SYN-MEMBER-002")

    own = member_one.call_tool(
        "lookup_referral",
        {"referral_id": "SYN-REF-TRACK-001"},
    )
    other_owner = member_two.call_tool(
        "lookup_referral",
        {"referral_id": "SYN-REF-TRACK-001"},
    )

    assert own["status"] == "found"
    assert own["record"]["member_id"] == MEMBER_ONE
    assert other_owner == {
        "status": "not_found",
        "referral_id": "SYN-REF-TRACK-001",
    }
    with pytest.raises(ValidationError):
        member_one.call_tool(
            "lookup_referral",
            {"referral_id": "REF-1"},
        )


def test_unavailable_mock_ehr_service_fails_explicitly():
    class UnavailableTools(SchedulingInsuranceAgent):
        def lookup_appointments(self, request):
            raise RuntimeError("mock service unavailable")

    service = SyntheticMCPToolService(
        MEMBER_ONE,
        tool_executor=UnavailableTools(),
    )
    with pytest.raises(RuntimeError, match="mock service unavailable"):
        service.call_tool(
            "lookup_appointments",
            {"specialty": "cardiology"},
        )


def test_transient_mcp_service_failure_returns_unavailable_status():
    class DownTools(SchedulingInsuranceAgent):
        def lookup_appointments(self, request):
            raise ConnectionError("synthetic outage")

    service = SyntheticMCPToolService(
        MEMBER_ONE,
        tool_executor=DownTools(),
        resilience_config=ResilienceConfiguration(
            max_attempts=1,
            failure_threshold=1,
            timeout_seconds=1,
        ),
    )
    result = service.call_tool(
        "lookup_appointments",
        {"specialty": "cardiology"},
    )
    service.close()

    assert result == {
        "status": "unavailable",
        "operation": "lookup_appointments",
        "reason": "transient_failure",
    }


def test_fastmcp_exposes_real_tool_list_and_call_protocol():
    service = SyntheticMCPToolService(MEMBER_ONE)
    server = create_mcp_server(service)

    async def protocol_round_trip():
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            result = await client.call_tool(
                "lookup_appointments",
                {"specialty": "cardiology"},
            )
            return listed, result

    listed, result = asyncio.run(protocol_round_trip())
    names = {tool.name for tool in listed.tools}

    assert names == MCP_TOOL_ALLOWLIST
    assert result.isError is False
    assert "available" in result.content[0].text
