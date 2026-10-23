"""E2 pre-game messages: keyword counts and a plain listing (team-chat and team-evade only).

Counts are by regular expression, not by reading: a message that says "don't mirror me" still counts
as "mirror". The listing in figures/e2_plans.md is there so the counts can be checked by eye.
Output: figures/e2_plans.md and the counts on stdout.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CONDITIONS = ("team-chat", "team-evade")
KEYWORDS = {
    "cycle / alternate / mirror": r"alternat|cycl|rotat|mirror",
    "same move every round": r"\bboth (play|throw|go)\b|same move",
    "random / mixed / unpredictable": r"random|mix|unpredictab",
}


def main():
    lines = ["Pre-game messages in the two chat conditions (P1 writes first, P2 replies once).",
             "Keyword counts are regular-expression matches, not a reading of intent.", ""]
    for c in CONDITIONS:
        gs = sorted((json.loads(l) for l in Path(f"results/e2/{c}.jsonl").read_text().splitlines()
                     if l.strip()), key=lambda g: g["game_id"])
        counts = {who: {k: 0 for k in KEYWORDS} for who in ("P1", "P2")}
        body = []
        for g in gs:
            for msg in g["plans"]:
                who = f"P{msg['from'] + 1}"
                text = msg["text"] or ""
                for k, pat in KEYWORDS.items():
                    counts[who][k] += bool(re.search(pat, text, re.I))
                body.append(f"- game {g['game_id']} {who}: {' '.join(text.split())}")
        lines.append(f"## {c} ({len(gs)} games)")
        lines.append("")
        lines.append("| keyword | P1 messages | P2 replies |")
        lines.append("|---|---|---|")
        for k in KEYWORDS:
            lines.append(f"| {k} | {counts['P1'][k]} | {counts['P2'][k]} |")
        lines.append("")
        lines.extend(body)
        lines.append("")
        print(c, json.dumps(counts))
    Path("figures/e2_plans.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
