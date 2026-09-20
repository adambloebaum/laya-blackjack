import pytest
from fastapi.testclient import TestClient

from blackjack.engine import Game, Rules
from blackjack.server import app

client = TestClient(app)


def new_table(**kwargs):
    response = client.post("/api/sessions", json={"samples": 32, **kwargs})
    assert response.status_code == 200
    return response.json()


def test_sessions_are_isolated_and_stale_writes_rejected():
    a = new_table(seed=42)
    b = new_table(seed=42)
    response = client.post(f"/api/sessions/{a['id']}/step", json={"revision": 0, "action": "stand"})
    assert response.status_code == 200
    assert response.json()["revision"] == 1
    assert client.get(f"/api/sessions/{b['id']}").json()["state"] == b["state"]
    assert client.post(f"/api/sessions/{a['id']}/step", json={"revision": 0}).status_code == 409


def test_real_model_required_for_laya_policy():
    from blackjack import server

    old = server.policy.agent
    server.policy.agent = None
    try:
        table = new_table(seed=42)
        response = client.post(f"/api/sessions/{table['id']}/step", json={"revision": 0, "policy": "laya"})
        assert response.status_code == 409
        assert client.get(f"/api/sessions/{table['id']}").json()["revision"] == 0
    finally:
        server.policy.agent = old


def test_invalid_config_and_not_found():
    assert client.post("/api/sessions", json={"players": 8}).status_code == 422
    assert client.post("/api/sessions", json={"samples": 100000}).status_code == 422
    assert client.get("/api/sessions/missing").status_code == 404
    assert client.get("/api/health").json()["ok"]
    assert client.get("/").status_code == 200


def test_overnight_status_survives_server_restart_and_marks_stale_heartbeat(monkeypatch, tmp_path):
    import json
    import time

    from blackjack import server

    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    assert client.get("/api/overnight").json() == {"available": False}
    path = tmp_path / "overnight" / "example" / "status.json"
    path.parent.mkdir(parents=True)
    run = {"status": "running", "stage": "generation", "updated_unix": time.time()}
    path.write_text(json.dumps(run))
    assert client.get("/api/overnight").json()["status"] == "running"
    assert client.post("/api/jobs", json={"kind": "benchmark"}).status_code == 409
    run["updated_unix"] -= 120
    path.write_text(json.dumps(run))
    assert client.get("/api/overnight").json()["status"] == "unresponsive"
    run["status"] = "complete"
    path.write_text(json.dumps(run))
    assert client.get("/api/overnight").json()["status"] == "complete"
    assert server.latest_checkpoint() is None


def test_export_reconstructs_entire_session():
    table = new_table(seed=71, players=7)
    for _ in range(30):
        response = client.post(
            f"/api/sessions/{table['id']}/step", json={"revision": table["revision"], "policy": "basic"}
        )
        assert response.status_code == 200
        table.update(response.json())
    replay = client.get(f"/api/sessions/{table['id']}/export").json()
    game = Game(Rules(**replay["rules"]), replay["seed"])
    for action in replay["actions"]:
        game.deal() if action == "deal" else game.step(action)
    assert game.observation() == replay["state"]
    assert game.history == replay["history"]


@pytest.mark.parametrize("full_model", [False, True])
def test_training_job_runs_selected_mode_and_publishes_report(monkeypatch, tmp_path, full_model):
    from blackjack import server

    monkeypatch.setattr(server, "ARTIFACTS", tmp_path)
    calls = []

    def process(command, **kwargs):
        calls.append(command)
        if "train" in command:
            report = tmp_path / "runs" / "test-job" / "model" / "training_report.json"
            report.parent.mkdir()
            report.write_text('{"test": {"teacher_agreement": 0.8}}')

    monkeypatch.setattr(server.subprocess, "run", process)
    server.jobs["test-job"] = {"status": "running"}
    try:
        server.run_job("test-job", server.JobRequest(kind="train", full_model=full_model))
        assert server.jobs["test-job"]["status"] == "complete"
        assert len(calls) == 2
        assert ("--full-model" in calls[1]) == full_model
        assert server.jobs["test-job"]["result"]["test"]["teacher_agreement"] == 0.8
    finally:
        server.jobs.pop("test-job")
