# Calibrated net-influence collusion detection for LLM agents

Research code for the Apart Research AI Collusion Research Sprint (23–25 October 2026), challenge 1.4
(information-theoretic identification of colluding pairs).

## Preexisting work

**No code in this repository predates the sprint.** All files were written during the sprint window.
Before the sprint we only read papers and the challenge page and took notes (no code); those notes are
not part of this repository. The method we build on is Bonjour, Aggarwal & Bhargava (UAI 2022), for which we
found no public implementation; everything here is our own implementation from the paper's equations.

## Layout

- `collusion/stats.py` — plug-in conditional mutual information, individual and net influence, a max-T
  permutation test that calibrates every pair score against a within-state shuffle null, and a game-level
  cluster bootstrap test.
- `collusion/games.py` — scripted n-player repeated rock-paper-scissors with a planted colluding pair and planted
  confounders (shared history reaction, global hidden cue, pair-only hidden cue, personal bias, leader copying).
- `collusion/llm.py` — minimal client for the Kimi (Moonshot) chat API. The key is read from the OS keychain at
  run time and never stored.
- `collusion/llm_game.py` — four-player rock-paper-scissors played by Kimi models (prompts, pre-game messages,
  move parsing: a reply counts only if its first word is a move).
- `tests/` — analytic and control checks (`python -m pytest tests/`).
- `experiments/` — `e1_scripted.py` (scripted benchmark, writes `results/e1.json`), `e2_llm.py` (LLM games,
  writes `results/e2/<condition>.jsonl`; needs an API key), `e2_analyze.py` (detectors on the LLM games,
  writes `results/e2_summary.json`), `e2_plans.py` (pre-game message counts), `fig_e1.py` / `fig_e2.py`
  (figures and tables in `figures/`), `api_probes.py` (records two API constraints in `results/api_probes.txt`).
- `results/e2_discarded/` — runs we did not use, with the reason in the report (§3.4): a model that cannot
  disable thinking, and two runs scored with an earlier, looser move parser.
- `report/draft.md` — the report source; `report/report.pdf` is the submitted PDF, rebuilt with
  `python report/build_html.py --pdf` (needs the `markdown` package and Google Chrome).

## Running

```
python -m venv .venv && .venv/bin/pip install numpy scipy matplotlib pytest
.venv/bin/python -m pytest tests/
```

## License

MIT (see `LICENSE`).
