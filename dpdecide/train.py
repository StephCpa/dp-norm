"""Train a typed decision model on a (pretend-)private slice, with or without differential privacy.

Objectives (--objective): ce (full labels), brier (full labels), rlcd (bandit: reward c - p_a), rlvr (bandit: reward c).
Privacy (--privacy):
  none        no clipping, no noise                                    (reference)
  clip        per-unit clipping, no noise                              (diagnostic, NOT private)
  dpsgd       per-unit clipping + Gaussian noise, Poisson sampling     (eps,delta)-DP, add/remove one unit
  labeldp     k-ary RR on labels once, objective corrected             eps-label-DP (x public), delta = 0
  labeldp_raw RR labels used as if clean                               (negative control)
All modes share the same Poisson sampling of privacy units and the same normalization by the expected
batch size, so the only difference between arms is what the privacy mode changes.

Release scope. The DP guarantee in privacy.json covers the model weights and train_log.jsonl (norms of
the NOISY update and metrics on --eval-data, which must be public). Unnoised statistics of the private
slice (losses, rewards, per-unit norms, clip fractions, realized batch sizes, timing, data hashes) go to
private_diagnostics/, which is operator-only and is not covered. Mechanism randomness (sampling, noise,
randomized response) is OS entropy by default (--randomness secure); --randomness seeded is a
reproducible simulation and is marked release_private = false. Choosing among several runs composes
their budgets (privacy.selection_epsilon); the per-run ledger does not cover that.

  python -m dpdecide.train --init runs/q-warmup --data data/train.jsonl --start 32000 \
      --objective rlcd --privacy dpsgd --eps 4 --clip 1.0 --lora --out runs/dp-rlcd-e4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from dpdecide import labeldp as ldp
from dpdecide import privacy
from dpdecide.evaluate import evaluate_model, stride_sample
from dpdecide.objectives import OBJECTIVES, row_losses
from dpdecide.schema import filter_k, group_units, read_jsonl, select_rows

PRIVACY = ("none", "clip", "dpsgd", "labeldp", "labeldp_raw")


def cosine_lr(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    prog = min(1.0, (step - warmup) / max(total - warmup, 1))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="hub id or checkpoint dir (an eve-rlcd warmup works)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=32000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--canaries", default="", help="canaries_train.jsonl from dpdecide.audit make-canaries")
    ap.add_argument("--unit", choices=("context", "row"), default="context")
    ap.add_argument("--max-rows-per-unit", type=int, default=4)
    ap.add_argument("--objective", choices=OBJECTIVES, required=True)
    ap.add_argument("--privacy", choices=PRIVACY, default="none")
    ap.add_argument("--eps", type=float, default=0.0, help="target epsilon (dpsgd) or label-DP epsilon per unit")
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--sigma", type=float, default=0.0, help="override the calibrated noise multiplier")
    ap.add_argument("--accountant", choices=("pld", "rdp"), default="pld", help="accountant used to calibrate sigma")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--clip-labels", choices=("std", "blind"), default="std",
                    help="blind: clip factor from the worst case over labels (dpdecide/blind.py)")
    ap.add_argument("--blind-budget", choices=("unit", "row"), default="unit",
                    help="blind clipping of multi-row units: one unit factor, or a per-row factor with budget C/max_rows")
    ap.add_argument("--auto-clip-gamma", type=float, default=0.0,
                    help="> 0: automatic clipping, factor = C / (norm + gamma) (std) or C / (bound + gamma) (blind)")
    ap.add_argument("--only-k", default="", help="keep only rows with these option counts, e.g. 2 or 2-5")
    ap.add_argument("--n-units-public", type=int, default=0,
                    help="a PUBLIC unit count (declared, not measured) that fixes the sampling rate q = expected_batch/n "
                         "and the step count; 0 = derive them from the realized private count (simulation only)")
    ap.add_argument("--expected-batch", type=float, default=128.0, help="expected units per step (Poisson)")
    ap.add_argument("--epochs", type=float, default=2.0, help="expected passes over the units")
    ap.add_argument("--group", type=int, default=4, help="sampled actions per row (bandit objectives)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--global-clip", type=float, default=0.0, help="clip the final update norm (0 = off; post-processing)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--micro", type=int, default=8, help="rows per forward in the non-clipping modes")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="non-mechanism randomness (init, RLCD actions)")
    ap.add_argument("--randomness", choices=("secure", "seeded"), default="secure",
                    help="mechanism randomness: secure = OS entropy (private release); seeded = reproducible simulation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests); the ledger still reports the planned steps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else ""


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    log_path = out / "train_log.jsonl"
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(f"{log_path} exists; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    diag_dir = out / "private_diagnostics"
    diag_dir.mkdir(exist_ok=True)
    diag_path = diag_dir / "train_stats.jsonl"
    diag_path.write_text("")

    def log(**kw):  # release-safe: covered by the DP guarantee
        with open(log_path, "a") as f:
            f.write(json.dumps(kw, default=float) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)

    def diag(**kw):  # unnoised statistics of the private slice: operator-only, NOT covered, not printed
        with open(diag_path, "a") as f:
            f.write(json.dumps(kw, default=float) + "\n")

    run_id = out.resolve().name
    mseed = lambda stream: privacy.mechanism_seed(args.randomness, args.seed, run_id, stream)  # noqa: E731

    if args.privacy in ("dpsgd", "labeldp", "labeldp_raw") and args.eps <= 0:
        raise ValueError(f"--privacy {args.privacy} needs --eps > 0")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np_rng = np.random.default_rng(mseed("poisson-sampling"))  # sampling must be secret for amplification

    rows = filter_k(select_rows(read_jsonl(args.data), args.start, args.limit), args.only_k)
    n_train_rows = len(rows)
    if args.clip_labels == "blind" and args.privacy not in ("clip", "dpsgd"):
        raise ValueError("--clip-labels blind only applies to --privacy clip or dpsgd")
    if args.canaries:
        canaries = read_jsonl(args.canaries)
        train_ctx = {q.context.strip() for q in rows}
        clash = sum(q.context.strip() in train_ctx for q in canaries)
        if clash:
            raise ValueError(f"{clash} canaries share a context with the training slice")
        rows = rows + canaries
    assert all(q.answer is not None for q in rows), "every row needs a label"
    units, dropped = group_units(rows, args.unit, args.max_rows_per_unit)
    n_units = len(units)

    # labels the run is allowed to see
    labels = [q.answer for q in rows]
    rr_pk = rr_q = None
    ledger: dict = {}
    if args.privacy in ("labeldp", "labeldp_raw"):
        labels, rr_pk, rr_q, _, st = ldp.privatize(rows, units, args.eps,
                                                    np.random.default_rng(mseed("randomized-response")))
        ledger.update(st)

    # Sampling schedule. Under add/remove adjacency the unit count is private, and q and the step count
    # computed from it would release it. A private release fixes them from a public, declared count.
    schedule = "public" if args.n_units_public > 0 else "from-private-count"
    q, steps = privacy.plan(args.n_units_public or n_units, args.expected_batch, args.epochs)
    agg_mode = {"none": "none", "labeldp": "none", "labeldp_raw": "none", "clip": "clip", "dpsgd": "dp"}[args.privacy]
    sigma = 0.0
    if args.privacy == "dpsgd":
        sigma = args.sigma or privacy.calibrate_sigma(args.eps, args.delta, q, steps, args.accountant)
        ledger.update(eps_rdp=privacy.epsilon(sigma, q, steps, args.delta, "rdp"),
                      eps_pld=privacy.epsilon(sigma, q, steps, args.delta, "pld"))
    blockers = []
    if args.randomness != "secure":
        blockers.append("seeded mechanism randomness")
    if args.privacy not in ("dpsgd", "labeldp", "labeldp_raw"):
        blockers.append(f"privacy mode {args.privacy} is not private")
    if args.privacy == "dpsgd" and schedule != "public":
        blockers.append("sampling rate and steps derived from the private unit count (pass --n-units-public)")
    if args.canaries:
        blockers.append("canaries inserted (audit run)")
    ledger.update(privacy=args.privacy, objective=args.objective, unit=args.unit, n_units=n_units,
                  rows_used=sum(len(u) for u in units), rows_dropped_by_cap=dropped,
                  max_rows_per_unit=args.max_rows_per_unit, train_rows=n_train_rows,
                  canary_rows=len(rows) - n_train_rows, slice_start=args.start, slice_limit=args.limit,
                  expected_batch=args.expected_batch, sampling_rate=q, steps=steps, epochs_expected=args.epochs,
                  schedule=schedule, n_units_public=args.n_units_public,
                  sigma=sigma, accountant=args.accountant, clip=args.clip, clip_labels=args.clip_labels, blind_budget=args.blind_budget,
                  auto_clip_gamma=args.auto_clip_gamma, only_k=args.only_k, delta=args.delta if args.privacy == "dpsgd" else 0.0,
                  eps_target=args.eps, adjacency="add/remove one privacy unit" if args.privacy == "dpsgd" else
                  ("change the labels of one privacy unit" if args.privacy.startswith("labeldp") else "n/a"),
                  randomness=args.randomness,
                  release_private=not blockers, release_blockers=blockers,
                  release_scope=["model weights", "train_log.jsonl", "privacy.json"],
                  not_covered=["private_diagnostics/ (unnoised statistics of the private slice; operator-only)",
                               "selection among several runs: candidates compose (privacy.selection_epsilon)",
                               "non-private hyperparameter tuning on the private slice",
                               "--eval-data must be public; metrics on private data are not covered"],
                  notes=["Per-run guarantee only. End-to-end with a DP temperature: max(eps_train, eps_cal) if the "
                         "calibration slice shares no context with this slice (calibrate.py checks), else the sum; "
                         "a temperature refitted for each of n candidates composes n times on the calibration slice.",
                         "Only allowlisted fields are released (privacy.RELEASE_LEDGER_KEYS); counts, drops and "
                         "label-DP flip statistics are in private_diagnostics/ledger_internal.json."])
    public_ledger, internal_ledger = privacy.split_release(ledger, privacy.RELEASE_LEDGER_KEYS)
    (out / "privacy.json").write_text(json.dumps(public_ledger, indent=2, default=float))
    (diag_dir / "ledger_internal.json").write_text(json.dumps(internal_ledger, indent=2, default=float))
    (diag_dir / "data.json").write_text(json.dumps(dict(data_sha256=file_sha(args.data),
                                                         canaries_sha256=file_sha(args.canaries)), indent=2))
    log(event="plan", **{k: v for k, v in public_ledger.items() if not isinstance(v, list)})

    from dpdecide.model import DecisionLM
    model = DecisionLM.load(args.init, device=args.device, lora=args.lora, lora_r=args.lora_r,
                            grad_checkpointing=args.grad_checkpointing)
    model.train()
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    agg = privacy.Aggregator(params, agg_mode, args.clip, sigma, args.expected_batch, noise_seed=mseed("gaussian-noise"),
                             auto_gamma=args.auto_clip_gamma)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed + 15485863)
    eval_qs = stride_sample(read_jsonl(args.eval_data), args.eval_n) if args.eval_data and args.eval_every else []

    def rr_for(idx: list[int]):
        if args.privacy != "labeldp":
            return None
        return (torch.tensor([rr_pk[i] for i in idx], device=args.device),
                torch.tensor([rr_q[i] for i in idx], device=args.device))

    def unit_batch_loss(idx: list[int]):
        qs = [rows[i] for i in idx]
        logits = model.logits(qs, args.max_len)
        lab = torch.tensor([labels[i] for i in idx], device=logits.device)
        loss, st = row_losses(args.objective, logits, lab, args.group, gen, rr_for(idx),
                              raw_label_dp=args.privacy == "labeldp_raw")
        return loss.sum(), st

    def run_eval(step: int):
        res, _ = evaluate_model(model, eval_qs, args.max_len)
        o = res["overall"]
        log(eval_step=step, eval_acc=o["acc"], eval_ece=o["ece"], eval_brier=o["brier"], eval_nll=o["nll"],
            eval_overconf=o["overconf"], eval_resolution=o["resolution"])

    t0 = time.time()
    run_steps = min(steps, args.max_steps) if args.max_steps else steps
    if eval_qs:
        run_eval(0)
    for step in range(1, run_steps + 1):
        chosen = np.flatnonzero(np_rng.random(n_units) < q)
        sums: dict = {}
        if agg_mode == "none":
            flat = [(i, u) for u in chosen for i in units[u]]
            for s in range(0, len(flat), args.micro):
                part = flat[s:s + args.micro]
                loss, st = unit_batch_loss([i for i, _ in part])
                loss.backward()
                agg.add_unit(n_units=0)
                for k, v in st.items():
                    sums[k] = sums.get(k, 0.0) + v * len(part)
        elif args.clip_labels == "blind":
            from dpdecide import blind
            for u in chosen:
                idx = units[u]
                grads, st = blind.unit_gradient(model, [rows[i] for i in idx], [labels[i] for i in idx],
                                                args.objective, args.group, gen, args.clip, args.auto_clip_gamma,
                                                args.max_len,
                                                row_budget=args.max_rows_per_unit if args.blind_budget == "row" else 0)
                agg.add_clipped(grads, st["blind_bound"], st["blind_factor"])
                for k, v in st.items():
                    sums[k] = sums.get(k, 0.0) + v * len(idx)
        else:
            for u in chosen:
                loss, st = unit_batch_loss(units[u])
                loss.backward()
                agg.add_unit(1)
                for k, v in st.items():
                    sums[k] = sums.get(k, 0.0) + v * len(units[u])
        if agg_mode == "none":
            agg.units = len(chosen)
        ast = agg.finish()
        lr = cosine_lr(step, steps, args.lr, warmup=max(1, steps // 20))
        for g in opt.param_groups:
            g["lr"] = lr
        upd = torch.nn.utils.clip_grad_norm_(params, args.global_clip).item() if args.global_clip > 0 else float("nan")
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            n_rows = sum(len(units[u]) for u in chosen) or 1
            log(step=step, lr=lr, update_norm=upd)
            diag(step=step, sec=time.time() - t0, rows=n_rows, **{k: v / n_rows for k, v in sums.items()}, **ast)
        if eval_qs and step % args.eval_every == 0:
            run_eval(step)
    if eval_qs and run_steps % max(1, args.eval_every) != 0:
        run_eval(run_steps)
    meta = vars(args) | {"steps_run": run_steps, "steps_planned": steps}
    model.save(str(out), meta)
    diag(event="done", wall_sec=time.time() - t0)
    if torch.cuda.is_available():
        print(f"peak_vram_mb={torch.cuda.max_memory_allocated() // 2**20}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
