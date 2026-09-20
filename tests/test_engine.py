import random
from collections import Counter
from copy import deepcopy

import pytest

from blackjack.engine import Game, Hand, Rules, basic_action, total, value
from blackjack.reference import analyze, hit_bust_probability, next_card_probabilities, sample_world


def rig(player, dealer, draws=(), **rules):
    game = Game(Rules(players=1, **rules))
    game.hands = [[Hand(list(player))]]
    game.dealer = list(dealer)
    game.seen = list(player) + dealer[:1]
    game.shoe = list(reversed(draws))
    game.phase = "playing"
    game.round = 1
    return game


def test_aces_revalue_and_soft_total():
    assert total(["A♠", "A♥", "9♦"]) == (21, True)
    assert total(["A♠", "6♥", "K♦"]) == (17, False)


@pytest.mark.parametrize("payout", [1.2, 1.5])
def test_natural_payout_and_natural_push(payout):
    g = rig(["A♠", "K♥"], ["9♦", "8♣"], blackjack_payout=payout)
    g.hands[0][0].status = "blackjack"
    g._settle()
    assert g.bankroll == payout
    g = rig(["A♠", "K♥"], ["A♦", "Q♣"], blackjack_payout=payout)
    g._settle()
    assert g.bankroll == 0


def test_dealer_peek_settles_before_any_actions():
    g = Game(Rules(players=1))
    # Full shoe length avoids triggering the low-card reserve.
    g.shoe = ["2♠"] * 52 + list(reversed(["9♠", "A♥", "7♦", "K♣"]))
    g.rules = Rules(players=1, decks=1)
    g.deal()
    assert g.phase == "settled"
    assert g.legal_actions() == []
    assert g.bankroll == -1


@pytest.mark.parametrize("h17,profit", [(False, 0), (True, -1)])
def test_soft17_rule(h17, profit):
    g = rig(["10♠", "7♥"], ["A♦", "6♣"], ["2♣"], hit_soft_17=h17)
    g.step("stand")
    assert g.bankroll == profit


def test_double_one_card_and_two_units():
    g = rig(["5♠", "6♥"], ["10♦", "8♣"], ["10♠"])
    g.step("double")
    assert g.bankroll == 2
    assert g.hands[0][0].bet == 2
    assert len(g.hands[0][0].cards) == 3


def test_split_aces_one_card_no_natural_payout():
    g = rig(["A♠", "A♥"], ["10♦", "8♣"], ["10♠", "9♣"])
    g.step("split")
    assert g.phase == "settled"
    assert g.bankroll == 2
    assert all(h.status == "stood" and not h.natural for h in g.hands[0])


def test_surrender_and_double_after_split_rules():
    g = rig(["8♠", "8♥"], ["10♦", "8♣"], ["3♠", "2♣"], double_after_split=False)
    g.step("split")
    assert "double" not in g.legal_actions()
    assert "surrender" not in g.legal_actions()
    g = rig(["10♠", "6♥"], ["10♦", "8♣"])
    g.step("surrender")
    assert g.bankroll == -0.5


def test_split_limit_and_twenty_one_auto_stands():
    g = rig(["8♠", "8♥"], ["10♦", "8♣"], ["8♣", "8♦"], max_hands=2)
    g.step("split")
    assert "split" not in g.legal_actions()
    g = rig(["5♠", "6♥"], ["10♦", "8♣"], ["10♣"])
    g.step("hit")
    assert g.phase == "settled"
    assert g.bankroll == 1


def test_shoe_exhaustion_voids_whole_round():
    g = rig(["8♠", "8♥"], ["10♦", "8♣"], ["2♣"])
    g.step("split")
    assert g.phase == "void"
    assert g.bankroll == 0
    assert g.history[-1]["void"]


@pytest.mark.parametrize("players", range(1, 8))
def test_random_games_conserve_every_card_and_terminate(players):
    g = Game(Rules(players=players, decks=2, tablemate_policy="random"), seed=121 + players)
    rng = random.Random(42)
    for _ in range(50):
        g.deal()
        for _ in range(256):
            hidden = [] if g.hole_revealed else [g.dealer[1]]
            assert Counter(g.seen + g.shoe + hidden) == Counter(
                [
                    r + s
                    for _ in range(g.rules.decks)
                    for s in ("♠", "♥", "♦", "♣")
                    for r in ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
                ]
            )
            obs = g.observation()
            assert sum(obs["unseen_counts"]) == len(g.shoe) + len(hidden)
            if g.phase != "playing":
                break
            g.step(rng.choice(g.legal_actions()))
        else:
            pytest.fail("Round did not terminate")
    assert len(g.history) == 50
    assert g.bankroll == sum(x["profit"] for x in g.history)


def test_seed_and_action_replay_is_deterministic():
    a, b = Game(seed=15), Game(seed=15)
    for _ in range(100):
        if a.phase != "playing":
            a.deal()
            b.deal()
        else:
            action = basic_action(a.observation())
            a.step(action)
            b.step(action)
        assert a.observation() == b.observation()


def test_hidden_card_and_order_are_not_observed_or_used_by_reference():
    g = Game(seed=42)
    g.deal()
    before = deepcopy(g.observation())
    g.dealer[1], g.shoe[0] = g.shoe[0], g.dealer[1]
    g.shoe.reverse()
    g.seed = 999999
    assert g.observation() == before
    assert "seed" not in before and "shoe" not in before
    result = analyze(before, 32)
    assert result == analyze(g.observation(), 32)
    assert g.observation() == before


def test_negative_peek_conditioning_exact_marginals():
    g = Game(Rules(players=1), seed=2)
    g.deal()
    # Abstract public pool: two aces, three twos, five tens, ace up.
    obs = g.observation()
    obs.update(
        phase="playing", dealer=["A♠", "??"], unseen_counts=[2, 3, 0, 0, 0, 0, 0, 0, 0, 5], cards_remaining=9
    )
    probs = next_card_probabilities(obs)
    assert probs[9] == pytest.approx(5 / 9)
    assert probs[0] == pytest.approx((2 - 2 / 5) / 9)
    assert sum(probs) == pytest.approx(1)
    for seed in range(30):
        world = sample_world(obs, random.Random(seed))
        assert value(world.dealer[1]) != 10


def test_reference_ev_and_distribution_contract():
    g = Game(seed=42)
    g.deal()
    obs = g.observation()
    ref = analyze(obs, 64)
    assert ref["recommendation"] in g.legal_actions()
    assert sum(ref["dealer"].values()) + ref["dealer_unresolved"] == pytest.approx(1)
    assert 0 <= hit_bust_probability(obs) <= 1
    assert ref["actions"]["surrender"]["ev"] == -0.5
    assert ref["actions"]["surrender"]["standard_error"] == 0
    for result in ref["actions"].values():
        assert result["positive"] + result["negative"] + result["zero"] == pytest.approx(1)
        assert result["ci95"][0] <= result["ev"] <= result["ci95"][1]


def test_illegal_action_is_transactionally_rejected():
    g = Game(seed=42)
    g.deal()
    before = deepcopy(g.observation())
    with pytest.raises(ValueError):
        g.step("insurance")
    assert g.observation() == before
