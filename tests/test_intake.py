from types import SimpleNamespace
import pytest
from careflow.intake import parse_patient_message, IntakeParsingError


def fake_completion(content):
    def call(*, model, messages, temperature):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return call


def test_valid_intake_parses():
    json_text = '{"request_type":"schedule","specialty":"cardiology","symptoms":[],"urgency_flag":false,"insurance_id":" SYN-12 ","needs_human_review":false,"reason_for_review":null}'
    result = parse_patient_message("Book my cardiology follow-up, plan SYN-12", "mock/model", fake_completion(json_text))
    assert result.insurance_id == "SYN-12"
    assert result.request_type == "schedule"


def test_invalid_json_rejected():
    with pytest.raises(IntakeParsingError):
        parse_patient_message("Book appointment", "mock/model", fake_completion("not json"))


def test_missing_fields_rejected():
    with pytest.raises(IntakeParsingError):
        parse_patient_message("Book appointment", "mock/model", fake_completion('{"request_type":33,"urgency_flag":"not_bool"}'))


def test_clinical_advice_flagged_even_when_model_misses_it():
    text = '{"request_type":"question","specialty":null,"symptoms":[],"urgency_flag":false,"insurance_id":null,"needs_human_review":false,"reason_for_review":null}'
    result = parse_patient_message("What dose should I take?", "mock/model", fake_completion(text))
    assert result.needs_human_review is True


def test_blank_message_rejected():
    with pytest.raises(ValueError):
        parse_patient_message(" ", "mock/model", fake_completion("{}"))
