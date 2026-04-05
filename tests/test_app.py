from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from access_governance_env.server.web import reset_demo_session, run_baseline_demo
from server.app import app


def test_http_smoke_routes():
    client = TestClient(app)

    empty_reset_response = client.post("/reset")
    assert empty_reset_response.status_code == 200
    empty_reset_payload = empty_reset_response.json()
    assert empty_reset_payload["observation"]["difficulty"] == "medium"
    assert empty_reset_payload["info"]["step_count"] == 0

    reset_response = client.post("/reset", json={"seed": 5})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert "observation" in reset_payload
    assert "info" in reset_payload
    assert reset_payload["info"]["step_count"] == 0

    step_response = client.post(
        "/step",
        json={"action": {"kind": "inspect_policy_rules"}},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["reward"] == 0.0
    assert step_payload["info"]["step_count"] == 1

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert "step_count" in state_response.json()

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    tasks_payload = tasks_response.json()
    assert [task["id"] for task in tasks_payload["tasks"]] == ["easy", "medium", "hard"]
    assert all("description" in task for task in tasks_payload["tasks"])
    assert tasks_payload["action_schema"]["title"] == "AccessGovernanceAction"
    assert set(tasks_payload["action_schema"]["required"]) == {"kind"}
    assert set(tasks_payload["action_schema"]["properties"]["kind"]["enum"]) == {
        "inspect_requester_profile",
        "inspect_resource_metadata",
        "inspect_policy_rules",
        "inspect_access_history",
        "inspect_business_context",
        "approve",
        "escalate",
        "deny",
    }

    baseline_response = client.get("/baseline")
    assert baseline_response.status_code == 200
    baseline_payload = baseline_response.json()
    assert set(baseline_payload) == {"easy", "medium", "hard"}
    assert all(0.0 <= score <= 1.0 for score in baseline_payload.values())

    grader_response = client.get("/grader")
    assert grader_response.status_code == 200
    assert grader_response.json()["done"] is False
    assert grader_response.json()["score"] is None

    metadata_response = client.get("/metadata")
    assert metadata_response.status_code == 200
    assert metadata_response.json()["name"] == "access_governance_env"

    schema_response = client.get("/schema")
    assert schema_response.status_code == 200
    schema_payload = schema_response.json()
    assert "action" in schema_payload
    assert "observation" in schema_payload
    assert "state" in schema_payload

    mcp_response = client.post("/mcp", json={})
    assert mcp_response.status_code == 200
    assert mcp_response.json()["jsonrpc"] == "2.0"

    final_step_response = client.post(
        "/step",
        json={"action": {"kind": "approve"}},
    )
    assert final_step_response.status_code == 200
    assert final_step_response.json()["done"] is True

    final_grader_response = client.get("/grader")
    assert final_grader_response.status_code == 200
    assert final_grader_response.json()["done"] is True
    assert final_grader_response.json()["score"] is not None


def test_web_and_baseline_smoke():
    client = TestClient(app)

    web_response = client.get("/web")
    assert web_response.status_code == 200
    assert "Access Governance Review Console" in web_response.text

    session, *_ = reset_demo_session("medium", "7")
    session, _, _, _, score = run_baseline_demo(session, "medium", "7")

    assert session is not None
    assert score["final_reward"] >= 0.0
    assert score["gold_decision"] in {"approve", "escalate", "deny"}


def test_run_baseline_script_smoke():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "run_baseline.py", "--json"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert set(payload) == {"easy", "medium", "hard"}
