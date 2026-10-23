"""Figure and table from results/e2_summary.json (run experiments/e2_analyze.py first).

Figure 2: for every condition and state encoding, the observed net influence of the colluding pair
(players 1 and 2; indices 0 and 1 in the code) and its bootstrap lower confidence bound. A detector
flags the pair when the lower bound is above 0.
Table 2: which pairs each detector flags, per condition, with state = profile.
Pairs are printed 1-indexed (P1-P2) to match the prompts the agents saw.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OBSERVED = "#2a78d6"   # reference palette, categorical slot 1
BOUND = "#8a8a85"      # neutral ink for the interval
DET = {"mi": "MI", "cmi": "CMI given state", "net": "net (perm.)", "net-boot": "net (bootstrap)",
       "paper": "paper rule"}


def pair_label(p) -> str:
    return f"P{p[0] + 1}-P{p[1] + 1}"


def flags(d: dict) -> str:
    return ", ".join(pair_label(p) for p in d["flagged"]) or "-"


def main():
    s = json.loads(Path("results/e2_summary.json").read_text())
    conds = list(s["conditions"])
    states = list(next(iter(s["conditions"].values()))["detectors"])
    fig, axes = plt.subplots(1, len(states), figsize=(3.6 * len(states), 0.45 * len(conds) + 1.2),
                             sharey=True, sharex=True)
    for ax, sk in zip(axes, states):
        for y, c in enumerate(conds):
            nb = s["conditions"][c]["detectors"][sk]["net-boot"]
            ax.plot([nb["planted_lower"], nb["planted"]], [y, y], color=BOUND, lw=2,
                    solid_capstyle="round", label="bootstrap lower bound" if y == 0 else None)
            ax.plot(nb["planted_lower"], y, "|", color=BOUND, ms=10, mew=2)
            ax.plot(nb["planted"], y, "o", color=OBSERVED, ms=8, mec="white", mew=2,
                    label="observed net influence" if y == 0 else None)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"state = {sk}", fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel("min Γ for P1-P2 (bits)")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(conds)), conds)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/e2_planted.pdf")
    fig.savefig("figures/e2_planted.png", dpi=150)

    head = ["| condition | games | invalid | same move P1-P2 (indep.) | same move P3-P4 (indep.) | "
            + " | ".join(DET.values()) + " |",
            "|" + "---|" * (len(DET) + 5)]
    lines = list(head)
    for c in conds:
        row = s["conditions"][c]
        b, d = row["behaviour"], row["detectors"]["profile"]
        lines.append(f"| {c} | {b['games']} | {b['invalid']} | {b['same_move_01']:.2f} ({b['indep_same_01']:.2f}) | "
                     f"{b['same_move_23']:.2f} ({b['indep_same_23']:.2f}) | "
                     + " | ".join(flags(d[k]) for k in DET) + " |")
    note = ("Flagged pairs per detector, state = profile, α = 0.05. "
            "In brackets: the same-move rate two independent players would have with the move "
            "frequencies they actually played.\n\n")
    Path("figures/e2_table.md").write_text(note + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
