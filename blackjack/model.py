"""Laya boundary: public state only, typed outputs, no silent reference fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .engine import value
from .reference import DEALER_LABELS

BASE_MODEL = "convaiinnovations/laya"
BASE_REVISION = os.getenv("LAYA_REVISION", "1c5edc17a7acd8701df6fc341c0d179f1c62c982")
ACTION_DESCRIPTIONS = {
    "stand": "keep this hand",
    "hit": "draw one card",
    "double": "double wager and draw once",
    "split": "split the pair into two hands",
    "surrender": "give up half the wager",
}


def model_state(obs: dict) -> str:
    seat, hi = obs["active"]
    hand = obs["players"][seat][hi]
    # Compact and deterministic: critical state comes first; no seed or shoe order.
    state = {
        "hand": [value(c) for c in hand["cards"]],
        "total": hand["total"],
        "soft": hand["soft"],
        "dealer_up": value(obs["dealer"][0]),
        "dealer_peek": "no blackjack",
        "legal": obs["legal_actions"],
        "unseen_A_to_10": obs["unseen_counts"],
        "rules": obs["rules"],
        "active": obs["active"],
        "split": hand["from_split"],
        "table": [
            [{"cards": [value(c) for c in h["cards"]], "bet": h["bet"], "status": h["status"]} for h in hands]
            for hands in obs["players"]
        ],
    }
    return json.dumps(state, separators=(",", ":"))


def questions(obs: dict) -> dict:
    return {
        "action": {
            "type": "choice",
            "instructions": "Which legal blackjack action maximizes expected net return?",
            "criteria": {a: ACTION_DESCRIPTIONS[a] for a in obs["legal_actions"]},
        },
        "hit_bust": {"type": "noul", "instructions": "Will the active hand bust if it draws one card now?"},
        "dealer": {
            "type": "choice",
            "instructions": "Dealer final total if no more player cards are drawn?",
            "criteria": {k: k for k in DEALER_LABELS},
        },
    }


def resolve_checkpoint(source: str, revision: str = BASE_REVISION) -> str:
    p = Path(source)
    if p.is_dir():
        return str(p.resolve())
    if source.startswith((".", "/", "artifacts/")):
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    from huggingface_hub import snapshot_download

    # SDK root loading fetches bundled multilingual checkpoints too. Limit explicitly.
    return snapshot_download(
        source,
        revision=revision,
        allow_patterns=[
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/*",
            "tokenizer/*",
        ],
    )


def load_agent(source: str = BASE_MODEL, device: str | None = None):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import laya
    import torch

    torch.set_num_threads(int(os.getenv("LAYA_CPU_THREADS", "4")))
    agent = laya.load(resolve_checkpoint(source), device=device or os.getenv("LAYA_DEVICE"))
    agent.cfg["max_len"] = 1024
    agent.cfg["head_max_len"] = 256
    agent.model.encoder.config.reference_compile = False
    return agent


def ensure_fits(agent, state: str, qs: dict):
    """Reject silent state truncation, including crowded split tables."""
    from laya.common import build_sequence

    tokens = agent.tok(state, add_special_tokens=False)["input_ids"]
    for q in qs.values():
        internal = agent._to_internal(q)
        full, _ = build_sequence(agent.tok, state, internal, 100000, agent.cfg["head_max_len"])
        if len(full) > agent.cfg["max_len"]:
            raise ValueError(
                f"Observation exceeds model context ({len(full)} tokens, {len(tokens)} state tokens)."
            )


class LayaPolicy:
    def __init__(self):
        self.agent = None
        self.source = None
        self.error = None
        self.metadata = None

    def load(self, source: str, device: str | None = None):
        agent = load_agent(source, device)
        # Swap only once fully loaded, so a failed load preserves the previous model.
        metadata_path = Path(source) / "training_report.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        self.agent, self.source, self.metadata, self.error = agent, source, metadata, None

    def status(self):
        return {
            "loaded": self.agent is not None,
            "source": self.source,
            "error": self.error,
            "device": str(self.agent.device) if self.agent else None,
            "trained": bool(self.metadata),
            "report": self.metadata,
        }

    def predict(self, obs: dict):
        if self.agent is None:
            return {"available": False, "reason": "Load a Laya checkpoint to see model inference."}
        if obs["phase"] != "playing":
            return {"available": False, "reason": "No active decision."}
        state, qs = model_state(obs), questions(obs)
        ensure_fits(self.agent, state, qs)
        start = time.perf_counter()
        result = self.agent.predict(state, qs)
        return {
            "available": True,
            "source": self.source,
            "trained": bool(self.metadata),
            "latency_ms": (time.perf_counter() - start) * 1000,
            **result,
        }
