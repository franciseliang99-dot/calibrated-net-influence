"""E2 — run LLM tournaments. Usage: python experiments/e2_llm.py <condition> <n_games> [parallel_games]

Appends one JSON line per finished game to results/e2/<condition>.jsonl and skips game ids that are
already there, so an interrupted run can be resumed.
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.llm_game import GameSpec, play_game  # noqa: E402

MODELS = {
    "honest-mixed": ("kimi-k3", "kimi-k3", "kimi-k2.6", "kimi-k2.6"),
}


def main(condition: str, n_games: int, parallel: int = 5):
    out = Path("results/e2") / f"{condition}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["game_id"] for l in out.read_text().splitlines() if l.strip()}
    todo = [g for g in range(n_games) if g not in done]
    lock = threading.Lock()

    def run(gid):
        spec = GameSpec(condition=condition, game_id=gid, seed=gid,
                        models=MODELS.get(condition, ("kimi-k2.6",) * 4))
        res = play_game(spec)
        with lock, out.open("a") as f:
            f.write(json.dumps(res) + "\n")
        inv = sum(len(r["invalid"]) for r in res["log"])
        print(f"{condition} game {gid} done, invalid={inv}, scores={res['final_scores']}", flush=True)

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(run, todo))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 5)
