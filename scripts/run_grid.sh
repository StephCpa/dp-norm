#!/usr/bin/env bash
# D1 run plan: private calibrated decision models on eve-rlcd data-v1 (Qwen3-0.6B-Base).
# Phases are gated (see PLAN.md); run one at a time and read its gate before the next:
#   bash scripts/run_grid.sh phase0    # data check, public warmup, clip-norm probes
#   bash scripts/run_grid.sh phase1    # non-private references + DP-SGD tuning sweep (eps=4, seed 0)
#   B_STAR=.. EP_STAR=.. LR_STAR=.. bash scripts/run_grid.sh phase1b   # DP-SGD, clip-only and label-DP arms at eps=4
#   B_STAR=.. EP_STAR=.. LR_STAR=.. bash scripts/run_grid.sh phase2    # eps sweep + seeds 1,2
#   B_STAR=.. EP_STAR=.. LR_STAR=.. bash scripts/run_grid.sh phase3    # canary audits + loss-threshold MIA
#   python scripts/summarize.py --runs runs                             # table + pre-registered checks
#   bash scripts/run_grid.sh blindA     # label-blind clipping, binary rows only (needs only phase0's warmup)
#   python scripts/summarize_blind.py --runs runs                       # B1-B4 checks (PLAN_BLIND.md)
# CPU dry run on synthetic data: see scripts/dry_run_cpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}                  # train/val/test.jsonl from github.com/anthony-maio/eve-rlcd releases/tag/data-v1
RUNS=${RUNS:-runs}
BASE=${BASE:-Qwen/Qwen3-0.6B-Base}
DEVICE=${DEVICE:-cuda}
LORA_LR=${LORA_LR:-2e-4}            # non-private LoRA lr (eve-rlcd used 2e-5 for full fine-tuning)
B_STAR=${B_STAR:-}; EP_STAR=${EP_STAR:-}; LR_STAR=${LR_STAR:-}   # DP-SGD settings chosen from phase1's sweep
# slices of train.jsonl: warmup 0..6399 | canary pool 6400..23999 | DP calibration 24000..31999 | private 32000..
WARM_LIMIT=${WARM_LIMIT:-6400}; POOL_START=${POOL_START:-6400}; POOL_LIMIT=${POOL_LIMIT:-17600}
CAL_START=${CAL_START:-24000}; CAL_LIMIT=${CAL_LIMIT:-8000}; RL_START=${RL_START:-32000}
CANARY_N=${CANARY_N:-1000}; MIA_N=${MIA_N:-4000}; EVAL_N=${EVAL_N:-2000}; PROBE_STEPS=${PROBE_STEPS:-20}
SMOKE=${SMOKE:-}                    # extra train args for a dry run, e.g. "--max-steps 2 --max-len 128"
POST=${POST:-}                      # extra eval/calibrate/audit args for a dry run, e.g. "--max-len 128"
# Mechanism randomness. This grid is a research SIMULATION on public data, so it defaults to "seeded"
# (reproducible; seeds derived per run and per stream, never shared) and every ledger says
# release_private=false. Set RANDOMNESS=secure for anything meant as a private release.
RANDOMNESS=${RANDOMNESS:-seeded}
# A private release also needs a PUBLIC declared unit count to fix the sampling rate and steps; without
# it they come from the private count and the ledger lists that as a release blocker.
N_UNITS_PUBLIC=${N_UNITS_PUBLIC:-}
# The grid slices by file position (--start/--limit), which a private release may not do; the ledgers
# list it as a blocker. For release-protocol runs, pass hash partitions instead, e.g.
#   TRAIN_PARTITION=s1:0.0:0.8 CAL_PARTITION=s1:0.8:1.0 with RL_START=0 CAL_START=0 CAL_LIMIT=0.
TRAIN_PARTITION=${TRAIN_PARTITION:-}; CAL_PARTITION=${CAL_PARTITION:-}

PY="python -m"
TRAIN="$PY dpdecide.train --device $DEVICE --data $DATA/train.jsonl --start $RL_START --eval-data $DATA/val.jsonl \
 --eval-every 25 --eval-n $EVAL_N --grad-checkpointing --micro 8 --unit context --max-rows-per-unit 4 \
 --randomness $RANDOMNESS ${N_UNITS_PUBLIC:+--n-units-public $N_UNITS_PUBLIC} ${TRAIN_PARTITION:+--partition $TRAIN_PARTITION}"
LORA="--lora --lora-r 32"
CAL="--device $DEVICE --data $DATA/train.jsonl --start $CAL_START --limit $CAL_LIMIT --exclude-data $DATA/train.jsonl --exclude-start $RL_START --randomness $RANDOMNESS ${CAL_PARTITION:+--partition $CAL_PARTITION}"

need_star() { : "${B_STAR:?set B_STAR}" "${EP_STAR:?set EP_STAR}" "${LR_STAR:?set LR_STAR}"; }

clip_of() {  # median per-unit gradient norm from the phase0 probe of an objective
  python -c '
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
v = [r["unit_norm_median"] for r in rows if "unit_norm_median" in r and r["unit_norm_median"] == r["unit_norm_median"]]
print(f"{statistics.median(v):.4g}")' "$RUNS/probe-$1/private_diagnostics/train_stats.jsonl"
}

post() {  # $1 run dir, $2 eps for DP temperature scaling (0 = non-private reference)
  local run=$1 eps=$2
  $PY dpdecide.evaluate --device $DEVICE --model "$run" --data $DATA/test.jsonl --out "$run/eval" $POST
  $PY dpdecide.calibrate --model "$run" $CAL --eps "$eps" --out "$run/temperature.json" $POST
  $PY dpdecide.evaluate --device $DEVICE --model "$run" --data $DATA/test.jsonl \
      --temperature "$run/temperature.json" --out "$run/eval_ts" $POST
}

phase0() {
  if [ -z "${SKIP_MD5:-}" ]; then
    printf '%s\n' "cdfee4c9792751cf5b22668eb3f9dc33  train.jsonl" "4b95911ffc76ed1789f7989f623d0a5a  val.jsonl" \
      "15c33b165d70639d8bd7d23d624908f4  test.jsonl" | (cd "$DATA" && md5sum -c -)
  fi
  # public warmup = eve-rlcd's recipe (rows 0..6399, ~100 steps of 64 rows, full fine-tune, lr 2e-5)
  $PY dpdecide.train --device $DEVICE --init $BASE --data $DATA/train.jsonl --start 0 --limit $WARM_LIMIT --unit row \
      --objective ce --privacy none --expected-batch 64 --epochs 1 --lr 2e-5 --grad-checkpointing --micro 8 \
      --eval-data $DATA/val.jsonl --eval-every 25 --eval-n $EVAL_N --out $RUNS/warmup $SMOKE
  $PY dpdecide.evaluate --device $DEVICE --model $RUNS/warmup --data $DATA/test.jsonl --out $RUNS/warmup/eval $POST
  # per-unit gradient-norm probes on the PUBLIC val split (clip at 1e9 = nothing clipped; only the logged norms
  # are used), so choosing the clip spends no privacy
  for obj in ce brier rlcd; do
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy clip --clip 1e9 --expected-batch 128 --epochs 2 \
        --lr $LORA_LR --eval-every 0 --out $RUNS/probe-$obj $SMOKE --data $DATA/val.jsonl --start 0 \
        --max-steps $PROBE_STEPS
  done
  for obj in ce brier rlcd; do echo "median per-unit grad norm, $obj: $(clip_of $obj)"; done
}

phase1() {
  for obj in ce brier rlcd; do
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy none --expected-batch 128 --epochs 2 \
        --lr $LORA_LR --out $RUNS/n-$obj $SMOKE
    post $RUNS/n-$obj 0
  done
  # one full fine-tune RLCD run ties the LoRA references back to eve-rlcd's published 0.808 / 0.023 / 0.267
  $TRAIN --init $RUNS/warmup --objective rlcd --privacy none --expected-batch 128 --epochs 2 --lr 2e-5 \
      --out $RUNS/n-rlcd-full $SMOKE
  post $RUNS/n-rlcd-full 0
  # DP-SGD sweep on DP-CE at eps=4; select on the PUBLIC val split (final eval_brier in train_log.jsonl)
  local c; c=$(clip_of ce)
  for cfg in "128 2" "512 4" "1024 8"; do
    set -- $cfg
    for lr in 5e-4 1e-3; do
      $TRAIN --init $RUNS/warmup $LORA --objective ce --privacy dpsgd --eps 4 --clip $c --expected-batch $1 \
          --epochs $2 --lr $lr --out $RUNS/tune-b$1-ep$2-lr$lr $SMOKE
    done
  done
  python scripts/summarize.py --runs $RUNS --tune-only
}

phase1b() {
  need_star
  local c obj
  for obj in ce brier rlcd; do
    c=$(clip_of $obj)
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy dpsgd --eps 4 --clip $c \
        --expected-batch $B_STAR --epochs $EP_STAR --lr $LR_STAR --out $RUNS/d-$obj-e4 $SMOKE
    post $RUNS/d-$obj-e4 4
  done
  for obj in ce rlcd; do  # clipping without noise: does the miscalibration come from clipping (toy gate) or noise?
    c=$(clip_of $obj)
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy clip --clip $c \
        --expected-batch $B_STAR --epochs $EP_STAR --lr $LR_STAR --out $RUNS/c-$obj $SMOKE
    post $RUNS/c-$obj 0
  done
  for obj in rlcd ce; do  # label DP: RR once per label, corrected objective, non-private optimizer settings
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy labeldp --eps 4 --expected-batch 128 --epochs 2 \
        --lr $LORA_LR --out $RUNS/l-$obj-e4 $SMOKE
    post $RUNS/l-$obj-e4 4
  done
}

phase2() {
  need_star
  local c obj eps s
  for eps in 1 8; do
    for obj in ce brier rlcd; do
      c=$(clip_of $obj)
      $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy dpsgd --eps $eps --clip $c \
          --expected-batch $B_STAR --epochs $EP_STAR --lr $LR_STAR --out $RUNS/d-$obj-e$eps $SMOKE
      post $RUNS/d-$obj-e$eps $eps
    done
  done
  for eps in 2 8; do
    for obj in rlcd ce; do
      $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy labeldp --eps $eps --expected-batch 128 \
          --epochs 2 --lr $LORA_LR --out $RUNS/l-$obj-e$eps $SMOKE
      post $RUNS/l-$obj-e$eps $eps
    done
  done
  $TRAIN --init $RUNS/warmup $LORA --objective rlcd --privacy labeldp_raw --eps 2 --expected-batch 128 --epochs 2 \
      --lr $LORA_LR --out $RUNS/lraw-rlcd-e2 $SMOKE
  post $RUNS/lraw-rlcd-e2 2
  for s in 1 2; do
    for obj in ce rlcd; do
      $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy none --expected-batch 128 --epochs 2 \
          --lr $LORA_LR --seed $s --out $RUNS/n-$obj-s$s $SMOKE
      post $RUNS/n-$obj-s$s 0
      c=$(clip_of $obj)
      $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy dpsgd --eps 4 --clip $c --seed $s \
          --expected-batch $B_STAR --epochs $EP_STAR --lr $LR_STAR --out $RUNS/d-$obj-e4-s$s $SMOKE
      post $RUNS/d-$obj-e4-s$s 4
    done
    $TRAIN --init $RUNS/warmup $LORA --objective rlcd --privacy labeldp --eps 4 --expected-batch 128 --epochs 2 \
        --lr $LORA_LR --seed $s --out $RUNS/l-rlcd-e4-s$s $SMOKE
    post $RUNS/l-rlcd-e4-s$s 4
  done
}

phase3() {
  need_star
  local ex="$DATA/train.jsonl:$RL_START:0 $DATA/train.jsonl:$CAL_START:$CAL_LIMIT $DATA/test.jsonl:0:0 $DATA/val.jsonl:0:0"
  local c r eps kind
  for kind in mislabel labelswap; do
    $PY dpdecide.audit make-canaries --pool $DATA/train.jsonl --pool-start $POOL_START --pool-limit $POOL_LIMIT \
        --exclude $ex --n $CANARY_N --kind $kind --seed 11 --out-dir $RUNS/canaries-$kind
  done
  c=$(clip_of ce)
  $TRAIN --init $RUNS/warmup $LORA --objective ce --privacy none --expected-batch 128 --epochs 2 --lr $LORA_LR \
      --canaries $RUNS/canaries-mislabel/canaries_train.jsonl --out $RUNS/a-n-ce-mislabel $SMOKE
  for eps in 8 4; do
    $TRAIN --init $RUNS/warmup $LORA --objective ce --privacy dpsgd --eps $eps --clip $c --expected-batch $B_STAR \
        --epochs $EP_STAR --lr $LR_STAR --canaries $RUNS/canaries-mislabel/canaries_train.jsonl \
        --out $RUNS/a-d-ce-e$eps-mislabel $SMOKE
  done
  $TRAIN --init $RUNS/warmup $LORA --objective ce --privacy none --expected-batch 128 --epochs 2 --lr $LORA_LR \
      --canaries $RUNS/canaries-labelswap/canaries_train.jsonl --out $RUNS/a-n-ce-labelswap $SMOKE
  $TRAIN --init $RUNS/warmup $LORA --objective ce --privacy labeldp --eps 8 --expected-batch 128 --epochs 2 \
      --lr $LORA_LR --canaries $RUNS/canaries-labelswap/canaries_train.jsonl --out $RUNS/a-l-ce-e8-labelswap $SMOKE
  for r in a-n-ce-mislabel a-d-ce-e8-mislabel a-d-ce-e4-mislabel; do
    $PY dpdecide.audit score --device $DEVICE --model $RUNS/$r --secret $RUNS/canaries-mislabel/canaries_secret.json \
        --out $RUNS/$r/audit.json $POST
  done
  for r in a-n-ce-labelswap a-l-ce-e8-labelswap; do
    $PY dpdecide.audit score --device $DEVICE --model $RUNS/$r --secret $RUNS/canaries-labelswap/canaries_secret.json \
        --out $RUNS/$r/audit.json $POST
  done
  for r in n-rlcd n-ce d-rlcd-e4 d-ce-e4 l-rlcd-e4; do
    $PY dpdecide.audit mia --device $DEVICE --model $RUNS/$r --members $DATA/train.jsonl:$RL_START:0 \
        --nonmembers $DATA/test.jsonl:0:0 --n $MIA_N --out $RUNS/$r/mia.json $POST
  done
}

median_key() {  # median of a logged key over a probe run: $1 run dir, $2 key
  python -c '
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
v = [r[sys.argv[2]] for r in rows if sys.argv[2] in r and r[sys.argv[2]] == r[sys.argv[2]]]
print(f"{statistics.median(v):.4g}")' "$1/private_diagnostics/train_stats.jsonl" "$2"
}

blindA() {  # label-blind vs standard clipping on binary decisions (BoolQ + triage paging), eps = 4
  # KILLED 14 Nov 2026 before any GPU run: with std + DP-TS tuned after calibration, the toy shows no gain
  # (gates/PREREG_AUTO_FAIR.md). Kept for reference; FORCE_BLINDA=1 runs it anyway.
  if [ -z "${FORCE_BLINDA:-}" ]; then
    echo "blindA is killed by gates/PREREG_AUTO_FAIR.md (F3); set FORCE_BLINDA=1 to run it anyway" >&2; return 1
  fi
  local K2="--only-k 2" eps=4 obj clip regime lr run c g best cstd
  local CALK="$CAL $K2"
  # clip-norm probes on the PUBLIC val split (nothing is clipped at C = 1e9; only the logged norms are used)
  for obj in ce rlcd; do
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy clip --clip 1e9 --expected-batch 128 --epochs 2 \
        --lr $LORA_LR --eval-every 0 --out $RUNS/probeA-std-$obj $SMOKE --data $DATA/val.jsonl --start 0 $K2 \
        --max-steps $PROBE_STEPS
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy clip --clip-labels blind --clip 1e9 \
        --expected-batch 128 --epochs 2 --lr $LORA_LR --eval-every 0 --out $RUNS/probeA-blind-$obj $SMOKE \
        --data $DATA/val.jsonl --start 0 $K2 --max-steps $PROBE_STEPS
  done
  for obj in ce rlcd; do
    run=$RUNS/A-n-$obj
    $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy none --expected-batch 128 --epochs 4 --lr $LORA_LR \
        $K2 --out $run $SMOKE
    $PY dpdecide.evaluate --device $DEVICE --model $run --data $DATA/test.jsonl $K2 --out $run/eval $POST
    for regime in thr auto; do
      for clip in std blind; do
        # mechanism isolation (review P1-5): the SAME C (thr) or gamma (auto) for std and blind, both
        # from the standard norm's median on public rows
        cstd=$(median_key $RUNS/probeA-std-$obj unit_norm_median); c=$cstd
        if [ $regime = auto ]; then g=$(python -c "print(0.01 * $cstd)"); c=1.0; else g=0; fi
        for lr in 2e-4 1e-3; do
          run=$RUNS/A-$regime-$clip-$obj-lr$lr
          $TRAIN --init $RUNS/warmup $LORA --objective $obj --privacy dpsgd --eps $eps --clip $c --clip-labels $clip \
              --auto-clip-gamma $g --expected-batch 128 --epochs 4 --lr $lr $K2 --out $run $SMOKE
          $PY dpdecide.calibrate --model $run $CALK --eps $eps --out $run/temperature.json $POST
          $PY dpdecide.evaluate --device $DEVICE --model $run --data $DATA/val.jsonl $K2 \
              --temperature $run/temperature.json --out $run/eval_val_ts $POST
          $PY dpdecide.evaluate --device $DEVICE --model $run --data $DATA/test.jsonl $K2 --out $run/eval $POST
          $PY dpdecide.evaluate --device $DEVICE --model $run --data $DATA/test.jsonl $K2 \
              --temperature $run/temperature.json --out $run/eval_ts $POST
        done
      done
    done
  done
  python scripts/summarize_blind.py --runs $RUNS
}

"${1:?usage: run_grid.sh phase0|phase1|phase1b|phase2|phase3|blindA}"
