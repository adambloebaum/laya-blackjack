"""Finite-shoe, face-up American blackjack with a strictly public observation API."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("♠", "♥", "♦", "♣")
ACTIONS = ("stand", "hit", "double", "split", "surrender")


def value(card: str) -> int:
    rank = card[:-1]
    return 1 if rank == "A" else 10 if rank in ("10", "J", "Q", "K") else int(rank)


def total(cards: list[str]) -> tuple[int, bool]:
    low = sum(value(c) for c in cards)
    soft = any(value(c) == 1 for c in cards) and low + 10 <= 21
    return low + 10 if soft else low, soft


@dataclass(frozen=True)
class Rules:
    players: int = 3
    decks: int = 6
    hit_soft_17: bool = False
    double_after_split: bool = True
    surrender: bool = True
    blackjack_payout: float = 1.5
    penetration: float = 0.75
    max_hands: int = 4
    tablemate_policy: str = "basic"

    def __post_init__(self):
        if not 1 <= self.players <= 7 or self.decks not in (1, 2, 4, 6, 8):
            raise ValueError("Use 1–7 players and 1, 2, 4, 6, or 8 decks.")
        if self.blackjack_payout not in (1.2, 1.5) or not 0.25 <= self.penetration <= 0.85:
            raise ValueError("Payout must be 1.2 or 1.5; penetration must be 0.25–0.85.")
        if not 2 <= self.max_hands <= 4 or self.tablemate_policy not in ("basic", "random", "conservative"):
            raise ValueError("Invalid split limit or tablemate policy.")


@dataclass
class Hand:
    cards: list[str] = field(default_factory=list)
    bet: float = 1.0
    status: str = "playing"
    from_split: bool = False
    split_aces: bool = False
    outcome: str | None = None
    profit: float = 0.0

    @property
    def natural(self):
        return len(self.cards) == 2 and total(self.cards)[0] == 21 and not self.from_split

    def public(self):
        return {
            **asdict(self),
            "total": total(self.cards)[0],
            "soft": total(self.cards)[1],
            "natural": self.natural,
        }


class ShoeExhausted(RuntimeError):
    pass


class Game:
    def __init__(self, rules: Rules | None = None, seed: int = 42):
        self.rules = rules or Rules()
        self.seed = seed  # Never included in a model observation.
        self.rng = random.Random(seed)
        # Behavior sampling must not advance future shuffle randomness. Replays apply
        # explicit actions and therefore do not repeat policy-selection RNG calls.
        self.behavior_rng = random.Random(seed ^ 0xB1AC)
        self.round = 0
        self.shoe_number = 0
        self.phase = "ready"
        self.shoe: list[str] = []
        self.seen: list[str] = []
        self.dealer: list[str] = []
        self.hands: list[list[Hand]] = [[] for _ in range(self.rules.players)]
        self.active = (0, 0)
        self.hole_revealed = False
        self.bankroll = 0.0
        self.history: list[dict] = []
        self.events: list[str] = []
        self._shuffle()

    def _shuffle(self):
        self.shoe = [r + s for _ in range(self.rules.decks) for s in SUITS for r in RANKS]
        self.rng.shuffle(self.shoe)
        self.seen = []
        self.shoe_number += 1

    def _draw(self, visible: bool = True):
        if not self.shoe:
            raise ShoeExhausted("Shoe exhausted: round void; no mid-round reshuffle.")
        card = self.shoe.pop()
        if visible:
            self.seen.append(card)
        return card

    def deal(self):
        if self.phase == "playing":
            raise ValueError("Finish the current round first.")
        # Conservative reserve. Even a fresh single deck can exhaust in rare seven-seat rounds;
        # those rounds are explicitly voided, never silently replenished.
        reserve = min(52 * self.rules.decks, 12 * (self.rules.players + 1))
        shuffled = (
            len(self.shoe) < reserve or len(self.shoe) / (52 * self.rules.decks) <= 1 - self.rules.penetration
        )
        if shuffled:
            self._shuffle()
        self.round += 1
        self.hands = [[Hand()] for _ in range(self.rules.players)]
        self.dealer = []
        self.hole_revealed = False
        self.events = [f"Round {self.round} · shoe {self.shoe_number}" + (" · shuffled" if shuffled else "")]
        for pass_index in range(2):
            for seat in self.hands:
                seat[0].cards.append(self._draw())
            self.dealer.append(self._draw(visible=pass_index == 0))
        self.phase = "playing"
        for seat in self.hands:
            if seat[0].natural:
                seat[0].status = "blackjack"
        self.active = (0, 0)
        # American hole-card peek before any player actions. No insurance in v1.
        if total(self.dealer)[0] == 21:
            self._settle()
        else:
            self._advance()

    def legal_actions(self):
        if self.phase != "playing":
            return []
        seat, hi = self.active
        hand = self.hands[seat][hi]
        if hand.status != "playing":
            return []
        actions = ["stand", "hit"]
        if len(hand.cards) == 2:
            if not hand.from_split or self.rules.double_after_split:
                actions.append("double")
            if value(hand.cards[0]) == value(hand.cards[1]) and len(self.hands[seat]) < self.rules.max_hands:
                actions.append("split")
            if self.rules.surrender and not hand.from_split:
                actions.append("surrender")
        return actions

    def step(self, action: str):
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r}; legal: {self.legal_actions()}")
        seat, hi = self.active
        hand = self.hands[seat][hi]
        self.events.append(f"{'Laya seat' if seat == 0 else f'Seat {seat + 1}'} · {action}")
        try:
            if action == "stand":
                hand.status = "stood"
            elif action == "surrender":
                hand.status = "surrendered"
            elif action in ("hit", "double"):
                if action == "double":
                    hand.bet *= 2
                hand.cards.append(self._draw())
                score = total(hand.cards)[0]
                if score > 21:
                    hand.status = "bust"
                elif score == 21 or action == "double":
                    hand.status = "stood"
            elif action == "split":
                aces = value(hand.cards[0]) == 1
                a, b = hand.cards
                left = Hand([a, self._draw()], hand.bet, from_split=True, split_aces=aces)
                right = Hand([b, self._draw()], hand.bet, from_split=True, split_aces=aces)
                for h in (left, right):
                    if aces or total(h.cards)[0] == 21:
                        h.status = "stood"
                self.hands[seat][hi : hi + 1] = [left, right]
            self._advance()
        except ShoeExhausted:
            self._void()

    def _advance(self):
        for seat, hands in enumerate(self.hands):
            for hi, hand in enumerate(hands):
                if hand.status == "playing":
                    self.active = (seat, hi)
                    return
        self._settle()

    def _reveal(self):
        if not self.hole_revealed and len(self.dealer) == 2:
            self.seen.append(self.dealer[1])
        self.hole_revealed = True

    def _settle(self):
        self._reveal()
        dealer_natural = len(self.dealer) == 2 and total(self.dealer)[0] == 21
        contested = any(h.status == "stood" for hands in self.hands for h in hands)
        try:
            if contested and not dealer_natural:
                while True:
                    score, soft = total(self.dealer)
                    if score > 17 or score == 17 and not (soft and self.rules.hit_soft_17):
                        break
                    self.dealer.append(self._draw())
        except ShoeExhausted:
            self._void()
            return
        dealer_total = total(self.dealer)[0]
        for hands in self.hands:
            for h in hands:
                score = total(h.cards)[0]
                if h.status == "surrendered":
                    h.profit = -h.bet / 2
                elif dealer_natural:
                    h.profit = 0 if h.natural else -h.bet
                elif h.natural:
                    h.profit = self.rules.blackjack_payout * h.bet
                elif score > 21:
                    h.profit = -h.bet
                elif dealer_total > 21 or score > dealer_total:
                    h.profit = h.bet
                elif score == dealer_total:
                    h.profit = 0
                else:
                    h.profit = -h.bet
                h.outcome = "win" if h.profit > 0 else "loss" if h.profit < 0 else "push"
        self.phase = "settled"
        self._record()

    def _void(self):
        self._reveal()
        self.phase = "void"
        for hands in self.hands:
            for h in hands:
                h.profit = 0
                h.outcome = "void"
        self.events.append("Shoe exhausted · all bets returned")
        self._record()

    def _record(self):
        profit = sum(h.profit for h in self.hands[0])
        self.bankroll += profit
        self.history.append(
            {
                "round": self.round,
                "profit": profit,
                "bankroll": self.bankroll,
                "void": self.phase == "void",
                "shoe": self.shoe_number,
            }
        )
        self.history = self.history[-1000:]
        self.events.append(f"Round {'void' if self.phase == 'void' else 'settled'} · {profit:+g} units")

    def observation(self):
        """Sufficient public state; no seed, RNG, future order, or concealed rank."""
        unseen = [4 * self.rules.decks] * 9 + [16 * self.rules.decks]
        for c in self.seen:
            unseen[value(c) - 1] -= 1
        rc = sum(1 if 2 <= value(c) <= 6 else -1 if value(c) in (1, 10) else 0 for c in self.seen)
        return {
            "rules": asdict(self.rules),
            "round": self.round,
            "shoe_number": self.shoe_number,
            "phase": self.phase,
            "active": list(self.active),
            "hero": 0,
            "players": [[h.public() for h in hands] for hands in self.hands],
            "dealer": self.dealer.copy()
            if self.hole_revealed
            else self.dealer[:1] + (["??"] if self.dealer else []),
            "dealer_total": total(self.dealer)[0] if self.hole_revealed else None,
            "dealer_no_blackjack": self.phase == "playing",
            "legal_actions": self.legal_actions(),
            "unseen_counts": unseen,
            "cards_remaining": len(self.shoe),
            "cards_seen": len(self.seen),
            "running_count": rc,
            "true_count": rc / max(len(self.shoe) / 52, 0.25),
            "bankroll": self.bankroll,
        }


def basic_action(obs: dict) -> str:
    """Transparent multi-deck basic-strategy heuristic, also used for rollout continuations."""
    legal = obs["legal_actions"]
    seat, hi = obs["active"]
    h = obs["players"][seat][hi]
    score, soft = total(h["cards"])
    up = value(obs["dealer"][0])
    d = 11 if up == 1 else up
    pair = value(h["cards"][0]) if len(h["cards"]) == 2 else 0
    das = obs["rules"]["double_after_split"]
    if "split" in legal:
        if pair in (1, 8) or pair == 9 and d in (2, 3, 4, 5, 6, 8, 9):
            return "split"
        if pair in (2, 3) and 2 <= d <= 7 and (das or d >= 4):
            return "split"
        if pair == 7 and 2 <= d <= 7 or pair == 6 and (3 <= d <= 6 or d == 2 and das):
            return "split"
        if pair == 4 and das and d in (5, 6):
            return "split"
    if "surrender" in legal and not soft and (score == 16 and d >= 9 or score == 15 and d == 10):
        return "surrender"
    double = False
    if soft:
        double = (
            score in (13, 14)
            and d in (5, 6)
            or score in (15, 16)
            and 4 <= d <= 6
            or score == 17
            and 3 <= d <= 6
            or score == 18
            and 3 <= d <= 6
            or obs["rules"]["hit_soft_17"]
            and score == 19
            and d == 6
        )
        stand = score >= 19 or score == 18 and d <= 8
    else:
        double = (
            score == 9
            and 3 <= d <= 6
            or score == 10
            and d <= 9
            or score == 11
            and (d <= 10 or obs["rules"]["hit_soft_17"])
        )
        stand = score >= 17 or 13 <= score <= 16 and d <= 6 or score == 12 and 4 <= d <= 6
    if double and "double" in legal:
        return "double"
    return "stand" if stand else "hit"


def tablemate_action(game: Game):
    obs = game.observation()
    if game.rules.tablemate_policy == "random":
        return game.behavior_rng.choice(obs["legal_actions"])
    if game.rules.tablemate_policy == "conservative":
        seat, hi = game.active
        return "stand" if total(game.hands[seat][hi].cards)[0] >= 12 else "hit"
    return basic_action(obs)
