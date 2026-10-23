"""Scripted n-player repeated rock-paper-scissors with planted collusion and planted confounders.

Moves: 0 = ROCK, 1 = PAPER, 2 = SCISSORS.  beats(m) = (m + 1) % 3 is the move that beats m;
loses_to(m) = (m + 2) % 3 is the move that m beats.

Payoff (used only for reporting, never by the detectors): each player gets +1 for every opponent
whose move theirs beats and -1 for every opponent whose move beats theirs.

Behaviours of honest players are mixed strategies, possibly conditioned on public history. The
colluding "assistant" (player 1 by default) plays the move the "primary" (player 0) beats with
probability `cp` (it knows the primary's move through ex-ante coordination), else plays uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K = 3
MOVES = ("ROCK", "PAPER", "SCISSORS")


def beats(m):
    return (m + 1) % 3


def loses_to(m):
    return (m + 2) % 3


def payoffs(row: np.ndarray) -> np.ndarray:
    n = row.shape[0]
    out = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                if row[j] == loses_to(row[i]):
                    out[i] += 1
                elif row[j] == beats(row[i]):
                    out[i] -= 1
    return out


# ---- public state encodings -------------------------------------------------------------------

def profile_state(prev: np.ndarray | None, n: int) -> int:
    """Count profile (#R, #P, #S) of the previous round; 0 for the first round.

    Index = 1 + rank of the composition among all compositions of n into 3 parts.
    """
    if prev is None:
        return 0
    n_profile_states(n)  # builds the index on first use
    c = np.bincount(prev, minlength=3)
    return 1 + _COMP_INDEX[n][(int(c[0]), int(c[1]))]


_COMP_INDEX: dict[int, dict] = {}


def n_profile_states(n: int) -> int:
    if n not in _COMP_INDEX:
        idx = {}
        for r in range(n + 1):
            for p in range(n + 1 - r):
                idx[(r, p)] = len(idx)
        _COMP_INDEX[n] = idx
    return 1 + len(_COMP_INDEX[n])


def plurality_state(prev: np.ndarray | None) -> int:
    """0 = first round; 1..3 = unique most-common previous move + 1; 4 = tie."""
    if prev is None:
        return 0
    c = np.bincount(prev, minlength=3)
    top = np.flatnonzero(c == c.max())
    return int(top[0]) + 1 if top.size == 1 else 4


STATE_FUNS = {
    "none": (lambda prev, n: 0, lambda n: 1),
    "plurality": (lambda prev, n: plurality_state(prev), lambda n: 5),
    "profile": (profile_state, n_profile_states),
}


def encode_states(actions: np.ndarray, game_ids: np.ndarray, kind: str) -> tuple[np.ndarray, int]:
    """Public state of every round, computed from the previous round of the same game."""
    f, ks = STATE_FUNS[kind]
    n = actions.shape[1]
    s = np.empty(actions.shape[0], dtype=np.int64)
    for t in range(actions.shape[0]):
        first = t == 0 or game_ids[t] != game_ids[t - 1]
        s[t] = f(None if first else actions[t - 1], n)
    return s, ks(n)


# ---- scripted tournaments ---------------------------------------------------------------------

@dataclass
class Config:
    n: int = 4
    rounds: int = 30
    games: int = 20
    cp: float = 0.0              # assistant (player 1) helps primary (player 0) with prob cp
    react: float = 0.0           # all players: prob of countering last round's plurality move
    common_cue: float = 0.0      # all players: prob of following a hidden shared cue (global confound)
    pair_cue: float = 0.0        # players 2,3: prob of following a hidden cue only they share
    bias: float = 0.0            # every player: extra prob of playing their personal favourite move
    leader: float = 0.0          # players 1 and 2 each copy player 0 with this prob (no pair stands out)


def simulate(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (actions[T, n], game_ids[T])."""
    rows, gids = [], []
    fav = rng.integers(0, 3, size=cfg.n)
    for g in range(cfg.games):
        prev = None
        for _ in range(cfg.rounds):
            cue = rng.integers(0, 3)
            pcue = rng.integers(0, 3)
            row = rng.integers(0, 3, size=cfg.n)
            for i in range(cfg.n):
                u = rng.random()
                if cfg.bias and rng.random() < cfg.bias:
                    row[i] = fav[i]
                if cfg.react and prev is not None and u < cfg.react:
                    c = np.bincount(prev, minlength=3)
                    row[i] = beats(int(np.argmax(c)))
                if cfg.common_cue and rng.random() < cfg.common_cue:
                    row[i] = cue
                if cfg.pair_cue and i in (2, 3) and rng.random() < cfg.pair_cue:
                    row[i] = pcue
            if cfg.leader:
                for i in (1, 2):
                    if rng.random() < cfg.leader:
                        row[i] = row[0]
            if cfg.cp and rng.random() < cfg.cp:
                row[1] = loses_to(row[0])
            rows.append(row)
            gids.append(g)
            prev = row
    return np.array(rows), np.array(gids)


def true_mi_assistant(cp: float) -> float:
    """Analytic I(A_0 ; A_1) in bits when both are otherwise uniform (Bonjour notes, sec. 4.3)."""
    hi = cp + (1 - cp) / 3
    lo = (1 - cp) / 3
    ps = np.array([hi, lo, lo])
    ps = ps[ps > 0]
    return float(np.log2(3) + (ps * np.log2(ps)).sum())
