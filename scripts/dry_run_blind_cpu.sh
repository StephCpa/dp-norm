#!/usr/bin/env bash
# CPU plumbing check of the blindA phase: tiny random Qwen3, synthetic tickets (each ticket's paging
# question is the binary row), 2-step training. The numbers mean nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${W:-/tmp/dpdecide-dry-blind}
rm -rf "$W" && mkdir -p "$W/data"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/make_tiny_model.py --out "$W/tiny"
python - "$W/data" <<'PYEOF'
import sys
from dpdecide.schema import write_jsonl
from dpdecide.synth import tickets
d = sys.argv[1]
write_jsonl(f"{d}/train.jsonl", tickets(500, seed=1, prefix="r"))
write_jsonl(f"{d}/val.jsonl", tickets(30, seed=2, prefix="v"))
write_jsonl(f"{d}/test.jsonl", tickets(40, seed=3, prefix="t"))
PYEOF
export DATA="$W/data" RUNS="$W/runs" BASE="$W/tiny" DEVICE=cpu SKIP_MD5=1
export WARM_LIMIT=150 POOL_START=150 POOL_LIMIT=450 CAL_START=600 CAL_LIMIT=150 RL_START=750
export EVAL_N=30 PROBE_STEPS=3 SMOKE="--max-steps 2 --max-len 128" POST="--max-len 128"
bash scripts/run_grid.sh phase0
FORCE_BLINDA=1 bash scripts/run_grid.sh blindA   # plumbing only: the phase itself is killed
python scripts/summarize_blind.py --runs "$RUNS" --md "$RUNS/blindA.md"
echo "BLIND DRY RUN OK: $RUNS/blindA.md"
