"""Bounded finite-shoe dynamic programming for one unsplit player hand.

The player chooses from its information set, never conditional on a known hole
card. The unseen pool includes that card; a negative peek changes draw marginals.
Unsupported or expensive states explicitly fall back in ``hybrid_analyze``.
"""

from __future__ import annotations

from functools import cache

from .engine import value
from .reference import DEALER_LABELS, analyze, hit_bust_probability, next_card_probabilities

VERSION = "hybrid-exact-v1"


class NodeBudgetExceeded(RuntimeError):
    pass


def remove(counts, index):
    return counts[:index] + (counts[index] - 1,) + counts[index + 1 :]


def score(low, ace):
    soft = ace and low + 10 <= 21
    return low + 10 if soft else low, soft


def exact_analyze(obs, max_nodes=50000, *, continuation="optimal"):
    """Exact within this supported engine scope, or an explicit unavailable result.

    ``basic`` is a diagnostic fixed-policy continuation after the initial action;
    it permits a same-state check of the benefit from optimal later decisions.
    The bound counts cache misses, making support deterministic across machines.
    """
    if max_nodes < 1 or continuation not in ("optimal", "basic"):
        raise ValueError("Use a positive node budget and optimal/basic continuation.")
    if (
        obs["phase"] != "playing"
        or obs["rules"]["players"] != 1
        or len(obs["players"]) != 1
        or len(obs["players"][0]) != 1
        or obs["active"] != [0, 0]
        or "split" in obs["legal_actions"]
        or not obs.get("dealer_no_blackjack")
        or obs["players"][0][0]["from_split"]
    ):
        return {"available": False, "reason": "unsupported_scope"}
    hand = obs["players"][0][0]
    counts = tuple(obs["unseen_counts"])
    if len(counts) != 10 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) < 1:
        raise ValueError("Invalid unseen counts.")
    up = value(obs["dealer"][0])
    excluded = 9 if up == 1 else 0 if up == 10 else -1
    h17 = obs["rules"]["hit_soft_17"]
    nodes = 0

    def tick():
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise NodeBudgetExceeded

    @cache
    def dealer(pool, low, ace):
        tick()
        total, soft = score(low, ace)
        if total > 17 or total == 17 and not (soft and h17):
            result = [0.0] * 7  # 17..21, bust, void
            result[5 if total > 21 else total - 17] = 1.0
            return tuple(result)
        n = sum(pool)
        if not n:
            return (0.0,) * 6 + (1.0,)
        result = [0.0] * 7
        for i, count in enumerate(pool):
            if count:
                child = dealer(remove(pool, i), low + i + 1, ace or i == 0)
                for k, p in enumerate(child):
                    result[k] += count / n * p
        return tuple(result)

    @cache
    def dealer_marginal(pool):
        tick()
        eligible = sum(c for i, c in enumerate(pool) if i != excluded)
        if not eligible:
            raise ValueError("Unseen pool is inconsistent with a negative dealer peek.")
        result = [0.0] * 7
        for i, count in enumerate(pool):
            if count and i != excluded:
                child = dealer(remove(pool, i), up + i + 1, up == 1 or i == 0)
                for k, p in enumerate(child):
                    result[k] += count / eligible * p
        return tuple(result)

    @cache
    def stand(pool, total):
        tick()
        dist = dealer_marginal(pool)
        return dist[5] + sum(p * ((total > k + 17) - (total < k + 17)) for k, p in enumerate(dist[:5]))

    def draws(pool):
        n = sum(pool)
        eligible = sum(c for i, c in enumerate(pool) if i != excluded)
        if n <= 1:
            return
        for i, count in enumerate(pool):
            if count:
                p = (count - (count / eligible if i != excluded else 0)) / (n - 1)
                if p > 0:
                    yield i, p

    @cache
    def hit(pool, low, ace, double=False):
        tick()
        result = 0.0
        for i, p in draws(pool):
            new_low, new_ace = low + i + 1, ace or i == 0
            total = score(new_low, new_ace)[0]
            after = remove(pool, i)
            v = (
                -1.0
                if total > 21
                else stand(after, total)
                if double or total == 21
                else play(after, new_low, new_ace)
            )
            result += p * v
        # No drawable cards: engine voids the whole round, returning all wagers.
        return result * (2 if double else 1)

    @cache
    def play(pool, low, ace):
        tick()
        total, soft = score(low, ace)
        if continuation == "basic":
            # After a hit there is no double, split, or surrender option.
            d = 11 if up == 1 else up
            keep = (
                (total >= 19 or total == 18 and d <= 8)
                if soft
                else (total >= 17 or 13 <= total <= 16 and d <= 6 or total == 12 and 4 <= d <= 6)
            )
            return stand(pool, total) if keep else hit(pool, low, ace)
        # Max is OUTSIDE hole-card marginalization: no perfect-information policy.
        return max(stand(pool, total), hit(pool, low, ace))

    try:
        ranks = [value(c) for c in hand["cards"]]
        low, ace = sum(ranks), 1 in ranks
        values = {"stand": stand(counts, score(low, ace)[0]), "hit": hit(counts, low, ace)}
        if "double" in obs["legal_actions"]:
            values["double"] = hit(counts, low, ace, True)
        if "surrender" in obs["legal_actions"]:
            values["surrender"] = -0.5
        values = {a: values[a] * hand["bet"] for a in obs["legal_actions"]}
        dist = dealer_marginal(counts)
        best = max(values, key=values.get)
        margins = {
            a: {"gap": values[best] - v, "standard_error": 0.0} for a, v in values.items() if a != best
        }
        return {
            "available": True,
            "method": "Exact finite-shoe dynamic programming / single unsplit hand / " + continuation,
            "teacher_kind": "exact",
            "samples": 0,
            "nodes": nodes,
            "actions": {a: {"ev": v} for a, v in values.items()},
            "recommendation": best,
            "hit_bust": hit_bust_probability(obs),
            "next_card": next_card_probabilities(obs),
            "dealer": dict(zip(DEALER_LABELS, dist[:6], strict=True)),
            "dealer_unresolved": dist[6],
            "label_quality": {
                "resolved": all(m["gap"] > 1e-10 for m in margins.values()),
                "margins": margins,
                "note": "Enumerated supported state space to floating-point precision; no sampling interval.",
            },
        }
    except NodeBudgetExceeded:
        return {"available": False, "reason": "node_budget", "nodes": nodes}
    finally:
        # Recursive cached closures otherwise survive until cyclic GC in worker pools.
        for fn in (dealer, dealer_marginal, stand, hit, play):
            fn.cache_clear()


def hybrid_analyze(obs, samples, *, seed, max_samples, max_nodes=50000):
    exact = exact_analyze(obs, max_nodes)
    if exact["available"]:
        return exact
    result = analyze(obs, samples, seed=seed, max_samples=max_samples)
    result.update(teacher_kind="monte_carlo", exact_fallback=exact["reason"])
    return result
