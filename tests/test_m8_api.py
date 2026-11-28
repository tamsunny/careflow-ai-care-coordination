from fastapi.testclient import TestClient

from careflow.api import DEV_REVIEWER_ID, create_app
from careflow.m8_evaluation import run_evaluation


def _intake(client, *, message="Please create a cardiology referral.", request_id="SYN-REQ-API-001"):
    return client.post(
        "/intake",
        headers={"X-CareFlow-Member-ID": "SYN-MEMBER-001"},
        json={
            "request_id": request_id,
            "idempotency_key": request_id,
            "message": message,
        },
    )


def test_health_openapi_and_admin_review_flow(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["authentication_mode"] == "development_fixture"
        assert client.get("/openapi.json").status_code == 200

        started = _intake(client)
        assert started.status_code == 200
        initial = started.json()
        assert initial["status"] == "awaiting_human"
        assert initial["eligibility_status"] == "eligible"
        assert initial["referral_status"] == "awaiting_human_approval"

        headers = {
            "X-CareFlow-Member-ID": "SYN-MEMBER-001",
            "X-CareFlow-Session-ID": initial["session_id"],
        }
        assert client.get(
            f"/workflows/{initial['workflow_id']}", headers=headers
        ).json()["workflow_id"] == initial["workflow_id"]

        denied = client.post(
            f"/workflows/{initial['workflow_id']}/review",
            headers={**headers, "X-CareFlow-Reviewer-ID": "SYN-REVIEWER-FAKE"},
            json={"decision": "approve"},
        )
        assert denied.status_code == 403

        approved = client.post(
            f"/workflows/{initial['workflow_id']}/review",
            headers={**headers, "X-CareFlow-Reviewer-ID": DEV_REVIEWER_ID},
            json={"decision": "approve"},
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "completed"
        assert approved.json()["referral_status"] == "routed"
        assert approved.json()["referral_id"].startswith("SYN-REF-")

        duplicate = client.post(
            f"/workflows/{initial['workflow_id']}/review",
            headers={**headers, "X-CareFlow-Reviewer-ID": DEV_REVIEWER_ID},
            json={"decision": "approve"},
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["referral_id"] == approved.json()["referral_id"]


def test_new_intake_creates_registered_session_and_rejects_invented_session(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        intake_operation = client.get("/openapi.json").json()["paths"]["/intake"]["post"]
        session_parameter = next(
            parameter
            for parameter in intake_operation["parameters"]
            if parameter["name"] == "X-CareFlow-Session-ID"
        )
        assert session_parameter["required"] is False
        assert "omit for a new intake" in session_parameter["description"].casefold()

        created = _intake(client, request_id="SYN-REQ-API-NEW-SESSION")
        assert created.status_code == 200
        created_body = created.json()
        assert created_body["session_id"].startswith("SYN-SESSION-")

        resumed_context = client.get(
            f"/workflows/{created_body['workflow_id']}",
            headers={
                "X-CareFlow-Member-ID": "SYN-MEMBER-001",
                "X-CareFlow-Session-ID": created_body["session_id"],
            },
        )
        assert resumed_context.status_code == 200

        invented = client.post(
            "/intake",
            headers={
                "X-CareFlow-Member-ID": "SYN-MEMBER-001",
                "X-CareFlow-Session-ID": "SYN-SESSION-DEMO-001",
            },
            json={
                "request_id": "SYN-REQ-API-INVENTED-SESSION",
                "idempotency_key": "invented-session",
                "message": "Please create a cardiology referral.",
            },
        )
        assert invented.status_code == 404
        assert invented.json()["detail"]["code"] == "session_not_found"


def test_clinical_review_never_routes_even_after_approval(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        started = _intake(
            client,
            message="What exact medication dose should I take?",
            request_id="SYN-REQ-API-CLINICAL",
        ).json()
        assert started["status"] == "awaiting_human"
        assert started["referral_status"] == "not_routed"

        result = client.post(
            f"/workflows/{started['workflow_id']}/review",
            headers={
                "X-CareFlow-Member-ID": "SYN-MEMBER-001",
                "X-CareFlow-Session-ID": started["session_id"],
                "X-CareFlow-Reviewer-ID": DEV_REVIEWER_ID,
            },
            json={"decision": "approve"},
        )
        assert result.status_code == 200
        assert result.json()["status"] == "human_reviewed_no_automation"
        assert result.json()["referral_status"] == "not_routed"
        assert result.json()["referral_id"] is None


def test_checkpoint_survives_app_restart_and_owner_isolation(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        started = _intake(client, request_id="SYN-REQ-API-RESTART").json()

    restarted_app = create_app(tmp_path)
    with TestClient(restarted_app) as client:
        reviewed = client.post(
            f"/workflows/{started['workflow_id']}/review",
            headers={
                "X-CareFlow-Member-ID": "SYN-MEMBER-001",
                "X-CareFlow-Session-ID": started["session_id"],
                "X-CareFlow-Reviewer-ID": DEV_REVIEWER_ID,
            },
            json={"decision": "approve"},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "completed"

        other_session = client.post(
            "/intake",
            headers={"X-CareFlow-Member-ID": "SYN-MEMBER-001"},
            json={
                "request_id": "SYN-REQ-API-OTHER",
                "idempotency_key": "other-session",
                "message": "Please create a cardiology referral.",
            },
        ).json()["session_id"]
        hidden = client.get(
            f"/workflows/{started['workflow_id']}",
            headers={
                "X-CareFlow-Member-ID": "SYN-MEMBER-001",
                "X-CareFlow-Session-ID": other_session,
            },
        )
        assert hidden.status_code == 404


def test_auth_validation_and_referral_lookup(tmp_path):
    app = create_app(
        tmp_path,
        identity_authenticator=lambda member_id, _session_id: member_id == "SYN-MEMBER-001",
        reviewer_authenticator=lambda _reviewer, _member, _session: False,
    )
    with TestClient(app) as client:
        assert client.post("/intake", json={}).status_code == 422
        assert client.post(
            "/intake",
            headers={"X-CareFlow-Member-ID": "SYN-MEMBER-002"},
            json={
                "request_id": "SYN-REQ-API-AUTH",
                "idempotency_key": "auth",
                "message": "Please create a cardiology referral.",
            },
        ).status_code == 401
        assert client.get(
            "/referrals/SYN-REF-TRACK-001",
            headers={"X-CareFlow-Member-ID": "SYN-MEMBER-001"},
        ).status_code == 200
        assert client.get(
            "/referrals/SYN-REF-TRACK-002",
            headers={"X-CareFlow-Member-ID": "SYN-MEMBER-001"},
        ).status_code == 404
        assert client.post(
            "/intake",
            headers={"X-CareFlow-Member-ID": "SYN-MEMBER-001"},
            json={
                "request_id": "SYN-REQ-API-EXTRA",
                "idempotency_key": "extra",
                "message": "Please create a cardiology referral.",
                "reviewer_id": "SYN-REVIEWER-DEV",
            },
        ).status_code == 422


def test_full_labeled_m8_evaluation_passes():
    report = run_evaluation()
    assert report["case_count"] >= 20
    assert report["failed"] == 0, [
        result for result in report["results"] if not result["passed"]
    ]
    assert report["retrieval_quality_metrics"] == "not_measured_by_workflow_safety_evaluation"
