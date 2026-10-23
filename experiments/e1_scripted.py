"""E1 — scripted 4-player repeated RPS: power, false-positive rate and confounders.

For every condition we simulate `reps` independent tournaments and run five detectors:
    mi      unconditional MI, max-T permutation calibrated
    cmi     conditional MI given public state, permutation calibrated
    net     bidirectional net influence given state, permutation calibrated   (ours)
    net-boot  bidirectional net influence, game-level basic bootstrap of H0: min Gamma <= 0   (ours)
    paper   bidirectional net influence >= 0.05 bits for exactly one pair   (Bonjour et al. 2022)

A detector "flags" the set of pairs it accuses. We record, per tournament:
    hit  = the planted pair (0, 1) is flagged and nothing else is
    fp   = some pair other than (0, 1) is flagged
Output: results/e1.json (one row per condition and detector).
"""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion import games, stats  # noqa: E402

PLANTED = (0, 1)
ALPHA = 0.05
N_PERM = 300


def run_one(args):
    cfg_kw, state_kind, seed = args
    rng = np.random.default_rng(seed)
    a, gid = games.simulate(games.Config(**cfg_kw), rng)
    st, ks = games.encode_states(a, gid, state_kind)
    res = stats.permutation_test(a, st, 3, ks, n_perm=N_PERM, seed=seed + 1)
    out = {}
    for name, r in res.items():
        flagged = set(r.flagged(ALPHA))
        out[name] = {"hit": flagged == {PLANTED}, "fp": bool(flagged - {PLANTED}),
                     "planted_score": r.observed[PLANTED], "planted_p": r.p_adj[PLANTED]}
    bt = stats.bootstrap_net_test(a, st, gid, 3, ks, alpha=ALPHA, seed=seed + 2)
    fb = set(bt["flagged"])
    out["net-boot"] = {"hit": fb == {PLANTED}, "fp": bool(fb - {PLANTED}),
                       "planted_score": bt["observed"][PLANTED], "planted_p": None}
    pr = stats.bonjour_rule(a, st, 3, ks, alpha=0.05)
    net = stats.pair_scores(a, st, 3, ks)["net"]
    out["paper"] = {"hit": pr == PLANTED, "fp": pr is not None and pr != PLANTED,
                    "planted_score": net[PLANTED], "planted_p": None,
                    "n_over_alpha": sum(v >= 0.05 for v in net.values())}
    return out


def conditions():
    base = dict(n=4, rounds=30)
    # 1. power curve vs collusion probability and data size (no confounder)
    for games_n in (10, 20, 40):
        for cp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
            yield f"power/g{games_n}/cp{cp}", {**base, "games": games_n, "cp": cp}, "profile"
    # 2. confounders at a fixed size, with and without planted collusion
    for cp in (0.0, 0.4):
        for label, kw, sk in [
            ("react/state-none", {"react": 0.5}, "none"),
            ("react/state-profile", {"react": 0.5}, "profile"),
            ("common-cue", {"common_cue": 0.4}, "profile"),
            ("pair-cue(2,3)", {"pair_cue": 0.4}, "profile"),
            ("bias", {"bias": 0.5}, "profile"),
            ("leader(0->1,2)", {"leader": 0.5}, "profile"),
        ]:
            yield f"confound/{label}/cp{cp}", {**base, "games": 20, "cp": cp, **kw}, sk


def main(reps: int = 200, out_path: str = "results/e1.json"):
    rows = []
    with ProcessPoolExecutor() as ex:
        for name, cfg_kw, sk in conditions():
            seeds = [zlib.crc32(f"{name}/{r}".encode()) for r in range(reps)]
            outs = list(ex.map(run_one, [(cfg_kw, sk, s) for s in seeds], chunksize=8))
            for det in outs[0]:
                hits = np.array([o[det]["hit"] for o in outs])
                fps = np.array([o[det]["fp"] for o in outs])
                sc = np.array([o[det]["planted_score"] for o in outs])
                rows.append({
                    "condition": name, "detector": det, "reps": reps,
                    "N": cfg_kw["games"] * cfg_kw["rounds"], "cp": cfg_kw["cp"], "state": sk,
                    "hit_rate": float(hits.mean()), "fp_rate": float(fps.mean()),
                    "planted_score_mean": float(sc.mean()), "planted_score_sd": float(sc.std()),
                })
            print(name, {r["detector"]: (round(r["hit_rate"], 2), round(r["fp_rate"], 2))
                         for r in rows if r["condition"] == name}, flush=True)
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps({"alpha": ALPHA, "n_perm": N_PERM, "rows": rows}, indent=1))


if __name__ == "__main__":
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
