"""Local, single-process lab server. Each browser gets an isolated, versioned table."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .engine import Game, Rules, basic_action, tablemate_action
from .model import BASE_MODEL, LayaPolicy
from .reference import analyze

app = FastAPI(title="Laya Blackjack Laboratory", version="0.1.0")
STATIC = Path(__file__).parent / "static"
ARTIFACTS = Path("artifacts")
policy = LayaPolicy()
model_lock = threading.Lock()
model_version = 0
sessions_lock = threading.Lock()
job_lock = threading.Lock()
jobs: dict[str, dict] = {}


class TableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    players: int = Field(default=3, ge=1, le=7)
    decks: Literal[1, 2, 4, 6, 8] = 6
    hit_soft_17: bool = False
    double_after_split: bool = True
    surrender: bool = True
    blackjack_payout: Literal[1.2, 1.5] = 1.5
    penetration: float = Field(default=0.75, ge=0.25, le=0.85)
    max_hands: int = Field(default=4, ge=2, le=4)
    tablemate_policy: Literal["basic", "random", "conservative"] = "basic"
    seed: int = Field(default=42, ge=0, le=2147483647)
    samples: int = Field(default=256, ge=32, le=2048)


class StepRequest(BaseModel):
    revision: int
    action: Literal["auto", "deal", "stand", "hit", "double", "split", "surrender"] = "auto"
    policy: Literal["reference", "laya", "basic"] = "reference"


class LoadRequest(BaseModel):
    source: Literal["base", "trained"] = "trained"


class JobRequest(BaseModel):
    kind: Literal["train", "benchmark"]
    states: int = Field(default=500, ge=50, le=20000)
    samples: int = Field(default=256, ge=32, le=2048)
    epochs: int = Field(default=3, ge=1, le=20)
    rounds: int = Field(default=200, ge=25, le=10000)
    players: int = Field(default=3, ge=1, le=7)


@dataclass
class Session:
    game: Game
    samples: int
    revision: int = 0
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    trace: list[str] = field(default_factory=lambda: ["deal"])
    cached: dict | None = None
    cache_model_version: int = -1


sessions: dict[str, Session] = {}


def get_session(sid: str):
    with sessions_lock:
        session = sessions.get(sid)
        if not session:
            raise HTTPException(404, "Table expired. Create a new table.")
        session.touched = time.monotonic()
        return session


def latest_checkpoint():
    reports = list(ARTIFACTS.glob("checkpoints/*/training_report.json")) + list(
        ARTIFACTS.glob("runs/*/model/training_report.json")
    )
    return str(max(reports, key=lambda p: p.stat().st_mtime).parent) if reports else None


def snapshot(session: Session):
    if session.cached and session.cache_model_version == model_version:
        return session.cached
    obs = session.game.observation()
    ref = analyze(obs, session.samples)
    with model_lock:
        try:
            inference = policy.predict(obs)
        except Exception as exc:
            inference = {"available": False, "reason": f"Inference failed: {exc}"}
        status = policy.status()
    session.cached = {
        "state": obs,
        "reference": ref,
        "inference": inference,
        "model": status,
        "revision": session.revision,
        "history": session.game.history,
        "events": session.game.events[-30:],
    }
    session.cache_model_version = model_version
    return session.cached


@app.get("/api/health")
def health():
    return {"ok": True, "version": "0.1.0"}


@app.post("/api/sessions")
def create_session(config: TableConfig):
    rules = Rules(**config.model_dump(exclude={"seed", "samples"}))
    game = Game(rules, config.seed)
    game.deal()
    session = Session(game, config.samples)
    with sessions_lock:
        stale = [sid for sid, s in sessions.items() if time.monotonic() - s.touched > 7200]
        for sid in stale:
            sessions.pop(sid)
        if len(sessions) >= 50:
            raise HTTPException(
                429, "Too many active tables; restart the local server or wait for idle tables to expire."
            )
        sid = uuid.uuid4().hex
        sessions[sid] = session
    with session.lock:
        return {"id": sid, **snapshot(session)}


@app.get("/api/sessions/{sid}")
def state(sid: str):
    session = get_session(sid)
    with session.lock:
        return snapshot(session)


@app.post("/api/sessions/{sid}/step")
def step(sid: str, request: StepRequest):
    session = get_session(sid)
    with session.lock:
        if request.revision != session.revision:
            raise HTTPException(409, "Table changed; refresh before taking another action.")
        g = session.game
        action = request.action
        if action == "auto":
            if g.phase != "playing":
                action = "deal"
            elif g.active[0] != 0:
                action = tablemate_action(g)
            elif request.policy == "basic":
                action = basic_action(g.observation())
            elif request.policy == "reference":
                action = snapshot(session)["reference"]["recommendation"]
            else:
                inference = snapshot(session)["inference"]
                if not inference["available"]:
                    raise HTTPException(409, inference["reason"])
                action = inference["answers"]["action"]["choice"]
        try:
            if action == "deal":
                g.deal()
            else:
                g.step(action)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        session.trace.append(action)
        session.revision += 1
        session.cached = None
        return snapshot(session)


@app.get("/api/sessions/{sid}/export")
def export(sid: str):
    session = get_session(sid)
    with session.lock:
        return {
            "format": "laya-blackjack-replay-v1",
            "rules": asdict(session.game.rules),
            "seed": session.game.seed,
            "actions": session.trace.copy(),
            "history": session.game.history.copy(),
            "state": session.game.observation(),
        }


@app.get("/api/model")
def model_status():
    return {**policy.status(), "checkpoint_available": latest_checkpoint() is not None}


@app.post("/api/model/load")
def load_model(request: LoadRequest):
    global model_version
    source = BASE_MODEL if request.source == "base" else latest_checkpoint()
    if source is None:
        raise HTTPException(404, "No trained checkpoint yet. Run a training experiment first.")
    with model_lock:
        try:
            policy.load(source)
            model_version += 1
        except Exception as exc:
            policy.error = str(exc)
            raise HTTPException(
                503, f"Could not load Laya: {exc}. Install dependencies with uv sync --extra model."
            ) from exc
    return policy.status()


def run_job(jid: str, request: JobRequest):
    directory = ARTIFACTS / "runs" / jid
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "run.log"
    try:
        base = [sys.executable, "-u", "-m", "blackjack.cli"]
        if request.kind == "train":
            commands = [
                base
                + [
                    "generate",
                    "--output",
                    str(directory / "data"),
                    "--states",
                    str(request.states),
                    "--samples",
                    str(request.samples),
                ],
                base
                + [
                    "train",
                    "--dataset",
                    str(directory / "data"),
                    "--output",
                    str(directory / "model"),
                    "--epochs",
                    str(request.epochs),
                ],
            ]
            report = directory / "model" / "training_report.json"
        else:
            command = base + [
                "benchmark",
                "--output",
                str(directory / "benchmark.json"),
                "--rounds",
                str(request.rounds),
                "--players",
                str(request.players),
                "--samples",
                str(request.samples),
            ]
            checkpoint = latest_checkpoint()
            if checkpoint:
                command += ["--model-path", checkpoint]
            commands = [command]
            report = directory / "benchmark.json"
        with log.open("w") as f:
            for command in commands:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
        with job_lock:
            jobs[jid].update(status="complete", result=json.loads(report.read_text()))
    except Exception as exc:
        with job_lock:
            jobs[jid].update(status="failed", error=str(exc))


@app.post("/api/jobs")
def create_job(request: JobRequest):
    if request.kind == "train":
        from importlib.util import find_spec

        if find_spec("laya") is None:
            raise HTTPException(409, "Install model dependencies: uv sync --extra model")
    with job_lock:
        if any(j["status"] == "running" for j in jobs.values()):
            raise HTTPException(409, "An experiment is already running.")
        jid = uuid.uuid4().hex[:12]
        jobs[jid] = {"id": jid, "kind": request.kind, "status": "running", "config": request.model_dump()}
    threading.Thread(target=run_job, args=(jid, request), daemon=True).start()
    return jobs[jid].copy()


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    with job_lock:
        if jid not in jobs:
            raise HTTPException(404, "Unknown experiment.")
        result = jobs[jid].copy()
    log = ARTIFACTS / "runs" / jid / "run.log"
    result["log"] = log.read_text(errors="replace")[-6000:] if log.exists() else "Starting…"
    return result


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
