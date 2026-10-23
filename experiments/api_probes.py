"""Record the two API constraints the report states, as raw status codes and error text.

1. kimi-k2.7-code-highspeed with thinking disabled.
2. kimi-k2.6 with thinking disabled at temperature 1.0 (we use 0.6).
Output: results/api_probes.txt. The API key is never printed (see collusion/llm.py).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.llm import chat  # noqa: E402

PROBES = [
    ("kimi-k2.7-code-highspeed", 0.6),
    ("kimi-k2.6", 1.0),
    ("kimi-k2.6", 0.6),   # control: the setting E2 uses, expected to succeed
]


def main():
    lines = [f"# {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
    for model, temp in PROBES:
        try:
            r = chat([{"role": "user", "content": "Reply with one word: ROCK."}], model=model,
                     temperature=temp, max_tokens=8, retries=1,
                     extra={"thinking": {"type": "disabled"}})
            lines.append(f"{model} thinking=disabled temperature={temp}: OK text={r['text']!r}")
        except RuntimeError as e:
            lines.append(f"{model} thinking=disabled temperature={temp}: {e}")
    Path("results/api_probes.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
