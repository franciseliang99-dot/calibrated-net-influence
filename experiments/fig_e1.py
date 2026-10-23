"""Figures and a table from results/e1.json: power curves and the confounder table."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DET = {"mi": "MI", "cmi": "CMI | state", "net": "net influence (perm.)",
       "net-boot": "net influence (bootstrap)", "paper": "Bonjour et al. rule (α=0.05)"}
STYLE = {"mi": ":", "cmi": "--", "net": "-", "net-boot": "-.", "paper": (0, (5, 1, 1, 1))}


def main():
    rows = json.loads(Path("results/e1.json").read_text())["rows"]
    power = [r for r in rows if r["condition"].startswith("power/")]
    sizes = sorted({r["N"] for r in power})
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 3.2), sharey=True)
    for ax, N in zip(axes, sizes):
        for d in DET:
            rs = [r for r in power if r["N"] == N and r["detector"] == d]
            # cp > 0: rate of flagging exactly the planted pair. cp = 0: nobody colludes, so plot the
            # rate of flagging any pair (hit + fp) as a hollow marker instead.
            pts = sorted((r["cp"], r["hit_rate"]) for r in rs if r["cp"] > 0)
            line, = ax.plot(*zip(*pts), linestyle=STYLE[d], marker="o", ms=3, label=DET[d])
            null = [r["hit_rate"] + r["fp_rate"] for r in rs if r["cp"] == 0]
            ax.plot([0.0] * len(null), null, marker="o", ms=5, mfc="none", color=line.get_color(),
                    linestyle="none")
        ax.set_title(f"N = {N} rounds")
        ax.set_xlabel("collusion probability")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rate")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/e1_power.pdf")
    fig.savefig("figures/e1_power.png", dpi=150)

    # At cp = 0 nobody colludes, so every flag (including one on P1-P2) is a false accusation:
    # any-flag rate = hit + fp (the two are disjoint by construction in e1_scripted.py).
    conf = [r for r in rows if r["condition"].startswith("confound/")]
    names = list(dict.fromkeys("/".join(r["condition"].split("/")[1:-1]) for r in conf))
    by = {(("/".join(r["condition"].split("/")[1:-1])), r["cp"], r["detector"]): r for r in conf}
    head = "| confounder | " + " | ".join(DET) + " |"
    sep = "|" + "---|" * (len(DET) + 1)
    t0 = [head, sep] + [f"| {n} | " + " | ".join(
        f"{by[n, 0.0, d]['hit_rate'] + by[n, 0.0, d]['fp_rate']:.3f}" for d in DET) + " |" for n in names]
    t4 = [head, sep] + [f"| {n} | " + " | ".join(
        f"{by[n, 0.4, d]['hit_rate']:.3f} / {by[n, 0.4, d]['fp_rate']:.3f}" for d in DET) + " |" for n in names]
    null = [r for r in power if r["cp"] == 0.0]
    t_null = [f"| {r['N']} | {r['detector']} | {r['hit_rate'] + r['fp_rate']:.3f} |" for r in null]
    Path("figures/e1_confounders.md").write_text(
        "cp = 0: rate of flagging any pair (all flags are false accusations).\n\n" + "\n".join(t0)
        + "\n\ncp = 0.4: exact identification rate / rate of flagging some other pair.\n\n" + "\n".join(t4)
        + "\n\nNo confounder, cp = 0: any-flag rate by size.\n\n| N | detector | any-flag |\n|---|---|---|\n"
        + "\n".join(t_null) + "\n")
    print(Path("figures/e1_confounders.md").read_text())

if __name__ == "__main__":
    main()
