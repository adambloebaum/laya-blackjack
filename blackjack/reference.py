"""Information-set Monte Carlo: sample hidden worlds from public observations only."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter

from .engine import Game, Hand, Rules, basic_action, total, value

DEALER_LABELS = ("17", "18", "19", "20", "21", "bust")
MAX_REFERENCE_SAMPLES = 16384


def observation_seed(obs: dict) -> int:
    # Deliberately independent of the simulator's shuffle seed and hidden state.
    return int.from_bytes(hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).digest()[:8], "big")


def next_card_probabilities(obs: dict) -> list[float]:
    """Exact rank marginals, including the information from a negative dealer peek."""
    counts = obs["unseen_counts"]
    n = sum(counts)
    if n <= 1:
        return [0.0] * 10
    up = value(obs["dealer"][0])
    excluded = 10 if up == 1 else 1 if up == 10 else None
    eligible = [c if i + 1 != excluded else 0 for i, c in enumerate(counts)]
    denom = sum(eligible)
    if denom == 0:
        raise ValueError("Public observation is inconsistent with a negative dealer peek.")
    return [(c - eligible[i] / denom) / (n - 1) for i, c in enumerate(counts)]


def hit_bust_probability(obs: dict) -> float:
    seat, hi = obs["active"]
    cards = obs["players"][seat][hi]["cards"]
    probs = next_card_probabilities(obs)
    return sum(
        p
        for rank, p in enumerate(probs, 1)
        if total(cards + [("A" if rank == 1 else str(rank)) + "♠"])[0] > 21
    )


def sample_world(obs: dict, rng: random.Random) -> Game:
    """Reconstruct a possible world without accepting the actual Game instance."""
    if obs["phase"] != "playing":
        raise ValueError("A playing observation is required.")
    g = object.__new__(Game)
    g.rules = Rules(**obs["rules"])
    g.seed = 0
    g.rng = random.Random(rng.getrandbits(64))
    g.behavior_rng = g.rng  # Single-round branches never shuffle; preserve paired rollout behavior.
    g.round = obs["round"]
    g.shoe_number = obs["shoe_number"]
    g.phase = "playing"
    g.active = tuple(obs["active"])
    g.hole_revealed = False
    g.bankroll = 0.0
    g.history = []
    g.events = []
    g.hands = [
        [Hand(**{k: h[k] for k in Hand.__dataclass_fields__}) for h in seat] for seat in obs["players"]
    ]
    cards = [
        ("A" if rank == 1 else str(rank)) + "♠"
        for rank, count in enumerate(obs["unseen_counts"], 1)
        for _ in range(count)
    ]
    up = value(obs["dealer"][0])
    excluded = 10 if up == 1 else 1 if up == 10 else None
    eligible = [i for i, c in enumerate(cards) if value(c) != excluded]
    hole = cards.pop(rng.choice(eligible))
    rng.shuffle(cards)
    g.shoe = cards
    g.dealer = [obs["dealer"][0], hole]
    full = [4 * g.rules.decks] * 9 + [16 * g.rules.decks]
    g.seen = [
        ("A" if rank == 1 else str(rank)) + "♠"
        for rank, (count, unseen) in enumerate(zip(full, obs["unseen_counts"], strict=True), 1)
        for _ in range(count - unseen)
    ]
    return g


def clone_world(world: Game) -> Game:
    """Clone a sampled branch without recursively copying RNG state and immutable rules."""
    g = object.__new__(Game)
    g.__dict__ = world.__dict__.copy()
    g.shoe, g.seen, g.dealer = world.shoe.copy(), world.seen.copy(), world.dealer.copy()
    g.hands = [
        [
            Hand(h.cards.copy(), h.bet, h.status, h.from_split, h.split_aces, h.outcome, h.profit)
            for h in hands
        ]
        for hands in world.hands
    ]
    g.rng = random.Random(0)
    g.rng.setstate(world.rng.getstate())
    if world.behavior_rng is world.rng:
        g.behavior_rng = g.rng
    else:
        g.behavior_rng = random.Random(0)
        g.behavior_rng.setstate(world.behavior_rng.getstate())
    g.history = world.history.copy()
    g.events = world.events.copy()
    return g


def rollout_action(g: Game, focal_seat: int) -> str:
    """Equivalent continuation without serializing card-count history on every draw."""
    legal = g.legal_actions()
    hand = g.hands[g.active[0]][g.active[1]]
    if g.active[0] != focal_seat:
        if g.rules.tablemate_policy == "random":
            return g.behavior_rng.choice(legal)
        if g.rules.tablemate_policy == "conservative":
            return "stand" if total(hand.cards)[0] >= 12 else "hit"
    return basic_action(
        {
            "active": (0, 0),
            "players": [[{"cards": hand.cards}]],
            "dealer": g.dealer[:1],
            "legal_actions": legal,
            "rules": {"double_after_split": g.rules.double_after_split, "hit_soft_17": g.rules.hit_soft_17},
        }
    )


def finish_world(g: Game, first_action: str, seat: int) -> float:
    g.step(first_action)
    for _ in range(256):
        if g.phase != "playing":
            return sum(h.profit for h in g.hands[seat])
        g.step(rollout_action(g, seat))
    raise RuntimeError("Round failed to terminate.")


def paired_margins(returns: dict[str, list[float]]) -> tuple[str, dict]:
    """Common-world differences reduce variance relative to independent action intervals."""
    n = len(next(iter(returns.values())))
    best = max(returns, key=lambda a: sum(returns[a]))
    margins = {}
    for action, values in returns.items():
        if action == best:
            continue
        differences = [x - y for x, y in zip(returns[best], values, strict=True)]
        gap = sum(differences) / n
        se = math.sqrt(sum((x - gap) ** 2 for x in differences) / (n - 1) / n)
        margins[action] = {"gap": gap, "standard_error": se}
    return best, margins


def analyze(
    obs: dict, samples: int = 256, seed: int | None = None, *, max_samples: int | None = None
) -> dict:
    if obs["phase"] != "playing":
        return {"available": False, "reason": "Deal a round to analyze a decision."}
    if not 16 <= samples <= MAX_REFERENCE_SAMPLES:
        raise ValueError(f"Use 16–{MAX_REFERENCE_SAMPLES} Monte Carlo samples.")
    adaptive = max_samples is not None
    limit = max_samples if adaptive else samples
    if not samples <= limit <= MAX_REFERENCE_SAMPLES:
        raise ValueError(f"max_samples must be between samples and {MAX_REFERENCE_SAMPLES}.")
    rng = random.Random(observation_seed(obs) if seed is None else seed)
    actions = obs["legal_actions"]
    returns = {a: [] for a in actions}
    voids = Counter()
    dealer_counts = Counter()
    next_check = samples
    for iteration in range(limit):
        world = sample_world(obs, rng)
        # Dealer marginal when no more player cards are drawn. Labeled as such in UI/docs.
        dealer = world.dealer.copy()
        shoe = world.shoe.copy()
        while True:
            score, soft = total(dealer)
            if score > 17 or score == 17 and not (soft and world.rules.hit_soft_17):
                break
            if not shoe:
                break
            dealer.append(shoe.pop())
        dealer_counts["bust" if total(dealer)[0] > 21 else str(total(dealer)[0])] += 1
        for action in actions:
            branch = clone_world(world)
            result = finish_world(branch, action, obs["active"][0])
            returns[action].append(result)
            voids[action] += branch.phase == "void"
        if adaptive and iteration + 1 == next_check:
            _, margins = paired_margins(returns)
            if all(m["gap"] > 3 * m["standard_error"] for m in margins.values()):
                break
            next_check = min(limit, next_check * 2)
    samples = len(next(iter(returns.values())))
    results = {}
    for a, xs in returns.items():
        mean = sum(xs) / samples
        se = math.sqrt(sum((x - mean) ** 2 for x in xs) / (samples - 1) / samples)
        results[a] = {
            "ev": mean,
            "standard_error": se,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "positive": sum(x > 0 for x in xs) / samples,
            "zero": sum(x == 0 for x in xs) / samples,
            "negative": sum(x < 0 for x in xs) / samples,
            "void": voids[a] / samples,
        }
    best = max(actions, key=lambda a: results[a]["ev"])
    result = {
        "available": True,
        "method": "finite-shoe Monte Carlo / basic-strategy continuation",
        "samples": samples,
        "actions": results,
        "recommendation": best,
        "hit_bust": hit_bust_probability(obs),
        "dealer": {k: dealer_counts[k] / samples for k in DEALER_LABELS},
        "dealer_unresolved": 1 - sum(dealer_counts[k] for k in DEALER_LABELS) / samples,
        "next_card": next_card_probabilities(obs),
    }
    if adaptive:
        _, margins = paired_margins(returns)
        result["label_quality"] = {
            "resolved": all(m["gap"] > 3 * m["standard_error"] for m in margins.values()),
            "margins": margins,
            "max_samples": limit,
            "note": "Three-standard-error sequential sampling heuristic; not a formal confidence guarantee.",
        }
    return result
