#!/usr/bin/env bash
# Whole run plan on CPU in a few minutes: tiny random Qwen3, synthetic tickets laid out like eve-rlcd's
# train.jsonl slices, every phase with 2-step training. Checks plumbing only; the numbers mean nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${W:-/tmp/dpdecide-dry}
rm -rf "$W" && mkdir -p "$W/data"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/make_tiny_model.py --out "$W/tiny"
python - "$W/data" <<'PYEOF'
import sys
from dpdecide.schema import write_jsonl
from dpdecide.synth import tickets
d = sys.argv[1]
write_jsonl(f"{d}/train.jsonl", tickets(500, seed=1, prefix="r"))   # 1500 rows
write_jsonl(f"{d}/val.jsonl", tickets(30, seed=2, prefix="v"))
write_jsonl(f"{d}/test.jsonl", tickets(40, seed=3, prefix="t"))
PYEOF
export DATA="$W/data" RUNS="$W/runs" BASE="$W/tiny" DEVICE=cpu SKIP_MD5=1
export WARM_LIMIT=150 POOL_START=150 POOL_LIMIT=450 CAL_START=600 CAL_LIMIT=150 RL_START=750
export CANARY_N=40 MIA_N=60 EVAL_N=30 PROBE_STEPS=3
export SMOKE="--max-steps 2 --max-len 128" POST="--max-len 128"
bash scripts/run_grid.sh phase0
bash scripts/run_grid.sh phase1
export B_STAR=128 EP_STAR=2 LR_STAR=1e-3
bash scripts/run_grid.sh phase1b
bash scripts/run_grid.sh phase2
bash scripts/run_grid.sh phase3
python scripts/summarize.py --runs "$RUNS" --md "$RUNS/summary.md"
echo "DRY RUN OK: $RUNS/summary.md"
