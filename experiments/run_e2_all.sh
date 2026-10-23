#!/bin/bash
# Run all E2 conditions one after another (the API allows ~100 requests/min per organisation, and the
# rate limiter in collusion/llm.py only coordinates threads inside one process). Resumable.
cd "$(dirname "$0")/.."
PY=${PY:-python3}
N=${N:-20}
for c in honest team team-chat team-evade honest-mixed; do
  $PY experiments/e2_llm.py "$c" "$N" 3 || exit $?
done
