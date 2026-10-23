"""Pairwise dependence statistics for collusion detection in repeated discrete games.

Data model (simultaneous-move game, public state):
    actions: int array of shape (T, n), actions[t, i] in {0..k-1}
    states:  int array of shape (T,), the public state each round is played in

Statistics (all in bits, plug-in / count estimates):
    gamma(i, j)  = I(A_i ; A_j | S)                      (individual influence, Bonjour et al. 2022 eq. 6)
    Gamma(i; j)  = gamma(i, j) - max_{k != i, j} gamma(k, j)   (net influence, eq. 4)

For a simultaneous game with a shared public state, gamma is symmetric, but Gamma is not.

Pair scores used by the detectors (one score per unordered pair {i, j}):
    "mi"   : I(A_i ; A_j)            unconditional, state ignored          (baseline)
    "cmi"  : gamma(i, j)             conditional on state, no subtraction  (baseline)
    "net"  : min(Gamma(i; j), Gamma(j; i))   bidirectional net influence   (Bonjour's rule uses this)

Two calibrations, answering different questions:

1. Max-T permutation test (`permutation_test`). H0: given the public state, every agent's action is
   independent of every other agent's action. Each permutation shuffles every agent's actions
   independently within every state stratum, recomputes all pair scores and records their maximum;
   adjusted p-values p_ij = (1 + #{perm max >= score_ij}) / (1 + P) control the family-wise error
   rate over pairs. Rejecting this H0 says "some dependence exists"; it does NOT say that the flagged
   pair is more dependent than its strongest rival. With the "net" score it still fires when the
   true min Gamma is exactly 0 (one leader copied by two followers: some pair is flagged in 55.5%
   of 200 runs of 20 games).

2. Game-level cluster bootstrap (`bootstrap_net_test`). H0: min(Gamma(i;j), Gamma(j;i)) <= 0, i.e.
   the pair is no more dependent than its strongest rival. This is the test that matches the
   collusion claim. It uses the basic (reverse-percentile) interval, because resampling games with
   replacement duplicates rows and shifts the bootstrap distribution of net down; the percentile
   interval applies that shift on the conservative side and has almost no power.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

SCORES = ("mi", "cmi", "net")


def _cmi_from_codes(x: np.ndarray, y: np.ndarray, s: np.ndarray, kx: int, ky: int, ks: int) -> float:
    """Plug-in I(X;Y|S) in bits from integer codes."""
    n = x.shape[0]
    if n == 0:
        return 0.0
    joint = np.bincount((s * kx + x) * ky + y, minlength=ks * kx * ky).reshape(ks, kx, ky).astype(float)
    ns = joint.sum(axis=(1, 2))
    nx = joint.sum(axis=2)
    ny = joint.sum(axis=1)
    # sum_{s,x,y} N(sxy)/N * log2( N(sxy) N(s) / (N(sx) N(sy)) )
    num = joint * ns[:, None, None]
    den = nx[:, :, None] * ny[:, None, :]
    mask = joint > 0
    return float((joint[mask] * np.log2(num[mask] / den[mask])).sum() / n)


def gamma_matrix(actions: np.ndarray, states: np.ndarray, k: int, ks: int) -> np.ndarray:
    """g[i, j] = I(A_i ; A_j | S); diagonal set to nan."""
    n = actions.shape[1]
    g = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        v = _cmi_from_codes(actions[:, i], actions[:, j], states, k, k, ks)
        g[i, j] = g[j, i] = v
    return g


def net_influence(g: np.ndarray) -> np.ndarray:
    """G[i, j] = Gamma(i; j) = g[i, j] - max_{k != i, j} g[k, j]  (requires n > 2)."""
    n = g.shape[0]
    if n < 3:
        raise ValueError("net influence needs n > 2 agents")
    G = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = [g[kk, j] for kk in range(n) if kk not in (i, j)]
            G[i, j] = g[i, j] - max(others)
    return G


def pair_scores(actions: np.ndarray, states: np.ndarray, k: int, ks: int) -> dict[str, dict[tuple, float]]:
    """All detector scores for every unordered pair."""
    n = actions.shape[1]
    pairs = list(combinations(range(n), 2))
    g = gamma_matrix(actions, states, k, ks)
    zeros = np.zeros(actions.shape[0], dtype=np.int64)
    out = {
        "mi": {p: _cmi_from_codes(actions[:, p[0]], actions[:, p[1]], zeros, k, k, 1) for p in pairs},
        "cmi": {p: float(g[p]) for p in pairs},
    }
    if n > 2:
        G = net_influence(g)
        out["net"] = {(i, j): float(min(G[i, j], G[j, i])) for i, j in pairs}
    return out


def _shuffle_within_strata(actions: np.ndarray, strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = actions.copy()
    for idx in strata:
        if idx.size < 2:
            continue
        for a in range(actions.shape[1]):
            out[idx, a] = actions[rng.permutation(idx), a]
    return out


@dataclass
class TestResult:
    score: str
    observed: dict           # pair -> score
    p_adj: dict              # pair -> max-T adjusted p-value
    top_pair: tuple
    top_p: float
    null_max: np.ndarray = field(repr=False)

    def flagged(self, alpha: float = 0.05) -> list[tuple]:
        return [p for p, v in self.p_adj.items() if v <= alpha]


def permutation_test(actions: np.ndarray, states: np.ndarray, k: int, ks: int,
                     n_perm: int = 500, seed: int = 0, scores: tuple = SCORES) -> dict[str, TestResult]:
    """Max-T permutation test for each requested pair score (mi uses its own unstratified shuffles)."""
    rng = np.random.default_rng(seed)
    obs = pair_scores(actions, states, k, ks)
    scores = tuple(s for s in scores if s in obs)
    strata = [np.flatnonzero(states == v) for v in np.unique(states)]
    # "mi" ignores state, so its null shuffles over all rows; the others shuffle within strata.
    all_rows = [np.arange(actions.shape[0])]
    null = {s: np.empty(n_perm) for s in scores}
    for b in range(n_perm):
        if "mi" in scores:
            pa = _shuffle_within_strata(actions, all_rows, rng)
            null["mi"][b] = max(pair_scores(pa, np.zeros_like(states), k, 1)["mi"].values())
        cond = [s for s in scores if s != "mi"]
        if cond:
            pa = _shuffle_within_strata(actions, strata, rng)
            ps = pair_scores(pa, states, k, ks)
            for s in cond:
                null[s][b] = max(ps[s].values())
    res = {}
    for s in scores:
        o = obs[s]
        p_adj = {p: (1 + int((null[s] >= v - 1e-12).sum())) / (1 + n_perm) for p, v in o.items()}
        top = max(o, key=o.get)
        res[s] = TestResult(s, o, p_adj, top, p_adj[top], null[s])
    return res


def bootstrap_net_test(actions: np.ndarray, states: np.ndarray, game_ids: np.ndarray, k: int, ks: int,
                       n_boot: int = 400, alpha: float = 0.05, seed: int = 0, method: str = "basic") -> dict:
    """Cluster (game-level) bootstrap test of H0: min(Gamma(i;j), Gamma(j;i)) <= 0, per pair.

    Unlike the permutation test, the null here is not "everyone is independent" but "this pair
    is no more dependent than its strongest rival", so a hidden cue shared by all players does not
    count as evidence. Games are resampled with replacement (states are per-round and travel with
    their rows). A pair is flagged when the lower end of its Bonferroni-adjusted one-sided interval
    is above 0: basic interval 2*obs - q(1 - alpha/m) (default) or percentile interval q(alpha/m).
    Returns {"observed", "lower", "flagged"}.
    """
    rng = np.random.default_rng(seed)
    games_u = np.unique(game_ids)
    rows_of = [np.flatnonzero(game_ids == g) for g in games_u]
    obs = pair_scores(actions, states, k, ks)["net"]
    pairs = list(obs)
    boot = np.empty((n_boot, len(pairs)))
    for b in range(n_boot):
        pick = rng.integers(0, len(rows_of), len(rows_of))
        idx = np.concatenate([rows_of[g] for g in pick])
        sc = pair_scores(actions[idx], states[idx], k, ks)["net"]
        boot[b] = [sc[p] for p in pairs]
    q = alpha / len(pairs)
    if method == "basic":
        lower = 2 * np.array([obs[p] for p in pairs]) - np.quantile(boot, 1 - q, axis=0)
    elif method == "percentile":
        lower = np.quantile(boot, q, axis=0)
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"observed": obs, "lower": dict(zip(pairs, lower.tolist())),
            "flagged": [p for p, lo in zip(pairs, lower) if lo > 0]}


def bonjour_rule(actions: np.ndarray, states: np.ndarray, k: int, ks: int, alpha: float = 0.05):
    """The paper's fixed-threshold rule: collusion iff exactly one pair has Gamma >= alpha both ways.

    Returns the flagged pair or None. (Iterates unordered pairs, i.e. the evident intent of
    Algorithm 1, not its literal double loop over ordered pairs.)
    """
    net = pair_scores(actions, states, k, ks)["net"]
    hits = [p for p, v in net.items() if v >= alpha]
    return hits[0] if len(hits) == 1 else None
