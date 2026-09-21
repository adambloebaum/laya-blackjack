import itertools
import json
import random
from collections import defaultdict

import pytest

from blackjack.engine import Game, Rules
from blackjack.exact_reference import exact_analyze, hybrid_analyze
from blackjack.reference import clone_world, sample_world


def tiny_observation(up, h17, ranks=(1, 2, 4, 5, 10), cards=(10, 6)):
    game = Game(Rules(players=1, hit_soft_17=h17), 4)
    game.deal()
    game.hands[0][0].cards = [str(c) + "♠" for c in cards]
    game.hands[0][0].status = "playing"
    game.dealer = [("A" if up == 1 else str(up)) + "♠", "2♠"]
    game.phase, game.active = "playing", (0, 0)
    obs = game.observation()
    obs["unseen_counts"] = [ranks.count(i) for i in range(1, 11)]
    obs["cards_remaining"] = len(ranks) - 1
    return obs


def brute_values(obs):
    """Independent enumeration through the real engine; group by public history.

    Each distinct rank permutation is equally likely conditional on the peek.
    Actions are chosen once per public-information group, not per hidden world.
    """
    ranks = tuple(i for i, n in enumerate(obs["unseen_counts"], 1) for _ in range(n))
    worlds = []
    for sequence in set(itertools.permutations(ranks)):
        up = obs["dealer"][0]
        if up == "A♠" and sequence[0] == 10 or up == "10♠" and sequence[0] == 1:
            continue
        g = sample_world(obs, random.Random(0))
        cards = [("A" if r == 1 else str(r)) + "♠" for r in sequence]
        g.dealer = [up, cards[0]]
        g.shoe = list(reversed(cards[1:]))
        worlds.append(g)

    def evaluate(games, action):
        settled, pending = 0.0, defaultdict(list)
        for original in games:
            g = clone_world(original)
            g.step(action)
            if g.phase != "playing":
                settled += g.hands[0][0].profit
            else:
                pending[json.dumps(g.observation(), sort_keys=True)].append(g)
        for same_information in pending.values():
            settled += len(same_information) * max(
                evaluate(same_information, a) for a in same_information[0].legal_actions()
            )
        return settled / len(games)

    return {a: evaluate(worlds, a) for a in obs["legal_actions"]}


@pytest.mark.parametrize("up", [1, 6, 10])
@pytest.mark.parametrize("h17", [False, True])
@pytest.mark.parametrize("cards", [(10, 6), (3, 7), (1, 6)])
def test_exact_matches_independent_engine_enumeration_with_peek_and_void(up, h17, cards):
    obs = tiny_observation(up, h17, cards=cards)
    before = json.dumps(obs, sort_keys=True)
    result = exact_analyze(obs)
    assert result["available"]
    assert {a: v["ev"] for a, v in result["actions"].items()} == pytest.approx(brute_values(obs), abs=1e-12)
    assert sum(result["dealer"].values()) + result["dealer_unresolved"] == pytest.approx(1)
    assert json.dumps(obs, sort_keys=True) == before


def test_exact_budget_and_unsupported_states_fall_back_explicitly():
    obs = tiny_observation(6, False)
    assert exact_analyze(obs, 1)["reason"] == "node_budget"
    ref = hybrid_analyze(obs, 16, seed=42, max_samples=16, max_nodes=1)
    assert ref["teacher_kind"] == "monte_carlo"
    assert ref["exact_fallback"] == "node_budget"
    assert ref["samples"] == 16
    obs["rules"]["players"] = 2
    assert exact_analyze(obs)["reason"] == "unsupported_scope"
    with pytest.raises(ValueError):
        exact_analyze(obs, 0)


def test_exact_probabilities_preserve_duplicate_rank_multiplicity():
    obs = tiny_observation(1, True, ranks=(2, 2, 3, 10, 10), cards=(3, 7))
    result = exact_analyze(obs)
    assert {a: v["ev"] for a, v in result["actions"].items()} == pytest.approx(brute_values(obs), abs=1e-12)


def test_optimal_continuation_dominates_fixed_basic_on_same_information():
    obs = tiny_observation(10, False, ranks=(2, 3, 4, 5, 6, 10), cards=(2, 3))
    optimal, basic = exact_analyze(obs), exact_analyze(obs, continuation="basic")
    assert optimal["available"] and basic["available"]
    for a in obs["legal_actions"]:
        assert optimal["actions"][a]["ev"] >= basic["actions"][a]["ev"] - 1e-12
    assert optimal["actions"]["stand"] == basic["actions"]["stand"]


def test_split_is_never_silently_omitted_from_exact_recommendation():
    obs = tiny_observation(6, False, cards=(8, 8))
    assert "split" in obs["legal_actions"]
    assert exact_analyze(obs)["reason"] == "unsupported_scope"
