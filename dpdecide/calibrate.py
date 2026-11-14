"""DP temperature scaling by the exponential mechanism.

Utility u(T) = -sum over rows of the Brier score at temperature T. A row's Brier score is in [0, 2] and
a privacy unit holds at most m rows, so adding or removing a unit (or changing its labels) moves u by at
most 2m: the mechanism P(T) ~ exp(eps * u(T) / (4m)) is eps-DP for both adjacency notions (delta = 0).
Brier rather than NLL because it is bounded without clipping and proper.

The calibration slice must share no context with the training slice; then the end-to-end guarantee is
max(eps_train, eps_cal) by parallel composition. Pass the training slice with --exclude-* and every
calibration row whose context appears there is dropped.

  python -m dpdecide.calibrate --model runs/x --data data/train.jsonl --start 24000 --limit 8000 \
      --exclude-data data/train.jsonl --exclude-start 32000 --eps 4 --out runs/x/temperature.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dpdecide.metrics import softmax_rows
from dpdecide.privacy import RELEASE_TEMPERATURE_KEYS, mechanism_seed, split_release
from dpdecide.schema import partition_record, select_partition, filter_k, group_units, read_jsonl, select_rows

# T in [1/16, 16]. The earlier [0.25, 4] grid capped the temperature: in the binary toy every public TS fit
# landed on T = 4 and the true optimum was about 4.6-6 (reaudit of df9bb65). at_grid_edge in the ledger is a
# function of the released T only, so it is public; an edge hit means the range is still too narrow.
GRID = np.exp(np.linspace(np.log(1 / 16), np.log(16.0), 97))


def brier_utility(logit_rows, answers, grid=GRID) -> np.ndarray:
    u = np.zeros(len(grid))
    for j, t in enumerate(grid):
        probs = softmax_rows(logit_rows, t)
        u[j] = -sum(((p - np.eye(len(p))[a]) ** 2).sum() for p, a in zip(probs, answers))
    return u


def exp_mech(util: np.ndarray, eps: float, sensitivity: float, rng: np.random.Generator) -> int:
    """Exponential mechanism via the Gumbel-max trick: argmax of eps*u/(2*sens) + Gumbel."""
    scores = eps * util / (2 * sensitivity) + rng.gumbel(size=len(util))
    return int(scores.argmax())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--partition", default="", help="SALT:LO:HI hash partition of units (see train.py)")
    ap.add_argument("--exclude-data", default="")
    ap.add_argument("--exclude-start", type=int, default=0)
    ap.add_argument("--exclude-limit", type=int, default=0)
    ap.add_argument("--unit", choices=("context", "row"), default="context")
    ap.add_argument("--max-rows-per-unit", type=int, default=4)
    ap.add_argument("--eps", type=float, required=True, help="0 = non-private argmin (reference only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomness", choices=("secure", "seeded"), default="secure",
                    help="secure = OS-entropy Gumbel noise (private release); seeded = reproducible simulation")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--only-k", default="", help="keep only rows with these option counts, e.g. 2 or 2-5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = select_rows(read_jsonl(args.data), args.start, args.limit)
    if args.partition:
        rows = select_partition(rows, args.unit, args.partition)
    rows = filter_k(rows, args.only_k)
    excluded = 0
    if args.exclude_data:
        banned = {q.context.strip() for q in select_rows(read_jsonl(args.exclude_data), args.exclude_start,
                                                          args.exclude_limit)}
        keep = [q for q in rows if q.context.strip() not in banned]
        excluded = len(rows) - len(keep)
        rows = keep
    units, dropped = group_units(rows, args.unit, args.max_rows_per_unit)
    used = [rows[i] for u in units for i in u]

    from dpdecide.model import DecisionLM
    model = DecisionLM.load(args.model, device=args.device)
    logit_rows = model.predict(used, args.max_len)
    util = brier_utility(logit_rows, [q.answer for q in used])
    sens = 2.0 * args.max_rows_per_unit
    if args.eps > 0:
        run_id = Path(args.out).resolve().parent.name + "/" + Path(args.out).name
        j = exp_mech(util, args.eps, sens, np.random.default_rng(
            mechanism_seed(args.randomness, args.seed, run_id, "exponential-mechanism")))
    else:
        j = int(util.argmax())
    blockers = ([] if args.eps > 0 else ["non-private argmin (eps = 0)"]) + \
               ([] if args.randomness == "secure" else ["seeded mechanism randomness"])
    if not args.partition or args.start or args.limit:
        blockers.append("positional row slicing or no --partition: unit adjacency on the raw file not preserved")
    if args.exclude_data:
        blockers.append("context exclusion couples this slice to the training slice (no parallel composition)")
    res = dict(temperature=float(GRID[j]), eps=args.eps, delta=0.0, sensitivity=sens, grid_size=len(GRID),
               grid=[float(g) for g in GRID], max_rows_per_unit=args.max_rows_per_unit, n_units=len(units),
               n_rows=len(used), rows_dropped_by_cap=dropped, rows_excluded_shared_context=excluded,
               private=args.eps > 0, randomness=args.randomness, release_private=not blockers,
               partition=partition_record(args.data, args.unit, args.partition, args.start, args.limit),
               excluded_by_context=bool(args.exclude_data), at_grid_edge=j in (0, len(GRID) - 1),
               release_blockers=blockers)
    public, internal = split_release(res, RELEASE_TEMPERATURE_KEYS)  # cardinalities stay internal
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(public, indent=2))
    diag = out.parent / "private_diagnostics"
    diag.mkdir(exist_ok=True)
    (diag / f"{out.stem}_internal.json").write_text(json.dumps(internal, indent=2))
    print(f"T={res['temperature']:.4f} eps={args.eps}" + ("  WARNING: T is on the grid edge" if res["at_grid_edge"] else ""))


if __name__ == "__main__":
    main()
