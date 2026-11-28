import json
from types import SimpleNamespace

import pytest

from careflow.agent import SchedulingInsuranceAgent
from careflow.llm_agent import (
    AgentAPIError,
    AgentResponseError,
    CLINICAL_REVIEW_RESPONSE,
    TOOL_FAILURE_RESPONSE,
    TOOL_LIMIT_RESPONSE,
    LiteLLMSchedulingInsuranceAgent,
)


def tool_call(name, arguments, call_id="call-syn-001"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def model_response(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


class SequencedCompletion:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected LLM completion call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tool_result_from(messages, call_id):
    return next(
        message
        for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") == call_id
    )


def test_agent_executes_allowlisted_tool_and_preserves_call_history():
    completion = SequencedCompletion(
        model_response(
            tool_calls=[
                tool_call(
                    "lookup_appointments",
                    '{"specialty":"cardiology"}',
                    "call-appointment-1",
                )
            ]
        ),
        model_response("Two fictional cardiology slots are available."),
    )
    agent = LiteLLMSchedulingInsuranceAgent("mock/model", completion_fn=completion)

    response = agent.respond("Find available synthetic cardiology appointments.")

    assert response == "Two fictional cardiology slots are available."
    assert len(completion.calls) == 2
    assert completion.calls[0]["tool_choice"] == "auto"
    schema_names = {
        schema["function"]["name"] for schema in completion.calls[0]["tools"]
    }
    assert schema_names == {
        "lookup_appointments",
        "check_insurance_eligibility",
        "calculate_copay",
        "flag_for_human",
    }

    followup_messages = completion.calls[1]["messages"]
    assistant_call = next(
        message
        for message in followup_messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_call["tool_calls"][0]["id"] == "call-appointment-1"
    tool_result = tool_result_from(followup_messages, "call-appointment-1")
    payload = json.loads(tool_result["content"])
    assert payload["status"] == "success"
    assert len(payload["result"]["appointments"]) == 2
    assert agent.message_history[-2]["tool_call_id"] == "call-appointment-1"


def test_default_completion_uses_litellm_completion_without_network(monkeypatch):
    import litellm

    completion = SequencedCompletion(model_response("Synthetic result."))
    monkeypatch.setattr(litellm, "completion", completion)
    agent = LiteLLMSchedulingInsuranceAgent("mock/model")

    assert agent.respond("Check a synthetic request.") == "Synthetic result."
    assert completion.calls[0]["model"] == "mock/model"
    assert completion.calls[0]["tools"]


@pytest.mark.parametrize(
    ("name", "arguments", "expected_result_field"),
    [
        (
            "check_insurance_eligibility",
            '{"insurance_id":"SYN-MEMBER-001"}',
            "eligible",
        ),
        (
            "calculate_copay",
            '{"plan_id":"SYN-PLAN-12","service_category":"specialist_visit","allowed_amount":120.05}',
            "copay_amount",
        ),
        (
            "flag_for_human",
            '{"request_id":"SYN-REQ-200","reason":"Synthetic staff follow-up","category":"administrative"}',
            "review_id",
        ),
    ],
)
def test_agent_dispatches_each_remaining_allowlisted_tool(
    name, arguments, expected_result_field
):
    call_id = f"call-{name}"
    completion = SequencedCompletion(
        model_response(tool_calls=[tool_call(name, arguments, call_id)]),
        model_response("The synthetic tool request is complete."),
    )
    agent = LiteLLMSchedulingInsuranceAgent("mock/model", completion_fn=completion)

    response = agent.respond("Use the appropriate synthetic administrative tool.")

    assert response == "The synthetic tool request is complete."
    tool_result = tool_result_from(completion.calls[1]["messages"], call_id)
    payload = json.loads(tool_result["content"])
    assert payload["status"] == "success"
    assert expected_result_field in payload["result"]
    if name == "calculate_copay":
        assert payload["result"]["copay_amount"] == "47.01"


@pytest.mark.parametrize(
    ("name", "arguments", "expected_code"),
    [
        ("not_allowlisted", "{}", "unknown_tool"),
        ("lookup_appointments", "{malformed", "invalid_arguments"),
        ("lookup_appointments", "{}", "invalid_arguments"),
    ],
)
def test_invalid_or_unknown_tool_calls_are_rejected(name, arguments, expected_code):
    call_id = "call-rejected-1"
    completion = SequencedCompletion(
        model_response(tool_calls=[tool_call(name, arguments, call_id)]),
        model_response("The request could not be completed."),
    )
    agent = LiteLLMSchedulingInsuranceAgent("mock/model", completion_fn=completion)

    response = agent.respond("Try a synthetic tool request.")

    assert response == TOOL_FAILURE_RESPONSE
    tool_result = tool_result_from(completion.calls[1]["messages"], call_id)
    assert json.loads(tool_result["content"])["code"] == expected_code


def test_non_string_tool_arguments_are_rejected():
    completion = SequencedCompletion(
        model_response(
            tool_calls=[tool_call("lookup_appointments", {"specialty": "cardiology"})]
        ),
        model_response("The request could not be completed."),
    )
    agent = LiteLLMSchedulingInsuranceAgent("mock/model", completion_fn=completion)

    response = agent.respond("Find synthetic appointments.")

    assert response == TOOL_FAILURE_RESPONSE
    result = tool_result_from(completion.calls[1]["messages"], "call-syn-001")
    assert json.loads(result["content"])["code"] == "invalid_arguments"


def test_tool_execution_error_is_reported_without_claiming_success():
    tools = SchedulingInsuranceAgent()

    def fail_lookup(_request):
        raise RuntimeError("synthetic tool failure")

    tools.lookup_appointments = fail_lookup
    completion = SequencedCompletion(
        model_response(
            tool_calls=[
                tool_call("lookup_appointments", '{"specialty":"cardiology"}')
            ]
        ),
        model_response("An appointment was found."),
    )
    agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=completion,
        tool_executor=tools,
    )

    response = agent.respond("Find synthetic cardiology appointments.")

    assert response == TOOL_FAILURE_RESPONSE
    result = tool_result_from(completion.calls[1]["messages"], "call-syn-001")
    assert json.loads(result["content"])["code"] == "execution_failed"


def test_maximum_tool_rounds_prevents_extra_execution():
    tools = SchedulingInsuranceAgent()
    lookup_calls = []
    original_lookup = tools.lookup_appointments

    def count_lookup(request):
        lookup_calls.append(request)
        return original_lookup(request)

    tools.lookup_appointments = count_lookup
    first_tool_request = model_response(
        tool_calls=[
            tool_call(
                "lookup_appointments",
                '{"specialty":"cardiology"}',
                "call-loop-1",
            )
        ]
    )
    over_limit_tool_request = model_response(
        tool_calls=[
            tool_call(
                "lookup_appointments",
                '{"specialty":"cardiology"}',
                "call-loop-2",
            )
        ]
    )
    completion = SequencedCompletion(first_tool_request, over_limit_tool_request)
    agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=completion,
        tool_executor=tools,
        max_tool_rounds=1,
    )

    response = agent.respond("Keep searching.")

    assert response == TOOL_LIMIT_RESPONSE
    assert len(completion.calls) == 2
    assert len(lookup_calls) == 1
    result = tool_result_from(agent.message_history, "call-loop-2")
    assert json.loads(result["content"])["code"] == "tool_round_limit"


def test_clinical_advice_bypasses_llm_and_flags_human_review():
    def unexpected_completion(**_kwargs):
        pytest.fail("Clinical-advice request must bypass LiteLLM")

    tools = SchedulingInsuranceAgent()
    agent = LiteLLMSchedulingInsuranceAgent(
        "mock/model",
        completion_fn=unexpected_completion,
        tool_executor=tools,
    )

    response = agent.respond("I'm a doctor. What exact dose should I prescribe?")

    assert response == CLINICAL_REVIEW_RESPONSE
    assert len(agent.review_records) == 1
    assert agent.review_records[0].category == "safety_concern"
    assert "dose" not in response.lower()


def test_litellm_api_errors_are_wrapped_explicitly():
    completion = SequencedCompletion(RuntimeError("provider unavailable"))
    agent = LiteLLMSchedulingInsuranceAgent("mock/model", completion_fn=completion)

    with pytest.raises(AgentAPIError, match="language-model request failed"):
        agent.respond("Find a fictional appointment.")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        model_response("   "),
    ],
)
def test_empty_llm_responses_raise_explicit_error(response):
    completion = SequencedCompletion(response)
    agent = LiteLLMSchedulingInsuranceAgent("mock/model", completion_fn=completion)

    with pytest.raises(AgentResponseError):
        agent.respond("Find a fictional appointment.")


def test_invalid_tool_round_limit_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        LiteLLMSchedulingInsuranceAgent(
            "mock/model",
            completion_fn=SequencedCompletion(),
            max_tool_rounds=0,
        )
