"""Summarize the blindA phase and check the pre-registered predictions B1-B4 of PLAN_BLIND.md.

For each (objective, regime, clip) the learning rate is chosen by validation Brier AFTER the arm's own
DP temperature (the full standard pipeline, DP-SGD + DP-TS, gets its best shot), then test metrics are
reported at T = 1 and at the DP temperature. Re-runnable: it only reads files.

  python scripts/summarize_blind.py --runs runs [--md runs/blindA.md]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpdecide.privacy import composed_epsilon, epsilon  # noqa: E402
from dpdecide.schema import partitions_disjoint  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text())["overall"] if p.exists() else None


def wall(run: Path):  # operator-side cost measurement, from the private diagnostics
    p = run / "private_diagnostics/train_stats.jsonl"
    for line in reversed(p.read_text().splitlines() if p.exists() else []):
        r = json.loads(line)
        if "wall_sec" in r or "sec" in r:
            return r.get("wall_sec", r.get("sec"))
    return None


def budget(cands: list[Path], calibrated: bool) -> str:
    """Epsilon of releasing the arm's selected model.
    - Selection on public validation data is post-processing of EVERY candidate: each candidate's own
      DP-SGD event is composed (heterogeneous sigma, q, steps allowed).
    - calibrated=True declares that the released model carries a DP temperature refitted per candidate.
      Then every candidate MUST have a private temperature.json; a missing one is a refusal, not a skip.
      calibrated=False declares an uncalibrated release (T = 1); temperatures are ignored and named.
    - Training and calibration combine by parallel composition (max) only when the public partition
      records prove the slices disjoint (schema.partitions_disjoint); otherwise sequentially (sum).
    - No finite number if any record is missing or non-private; blocked records give the accountant's
      number labelled as a simulation, not a guarantee."""
    if not cands:
        return "-"
    events, cal, blocked = [], 0.0, set()
    train_parts, cal_parts = [], []
    delta = None

    def status(rec: dict, name: str):
        """Cross-check release_private, randomness and release_blockers (reaudit of df9bb65): a record that
        claims a private release must be secure and unblocked; one that does not must say why."""
        rp, rnd, bl = rec.get("release_private"), rec.get("randomness"), rec.get("release_blockers")
        if rp is None or bl is None or rnd is None:
            return f"{name} lacks release_private, randomness or release_blockers"
        if rp and (bl or rnd != "secure"):
            return f"{name} claims a private release but is blocked or not secure"
        if not rp and not bl:
            return f"{name} is not a private release but names no blocker"
        blocked.update(bl)
        return None

    for c in cands:
        lp, tp = c / "privacy.json", c / "temperature.json"
        if not lp.exists():
            return f"no guarantee: {c.name} has no privacy.json"
        L = json.loads(lp.read_text())
        if L.get("privacy") != "dpsgd" or not L.get("sigma"):
            return f"no guarantee: {c.name} is not a DP-SGD run"
        if delta is not None and L["delta"] != delta:
            return "no guarantee: candidates use different delta"
        delta = L["delta"]
        events.append((L["sigma"], L["sampling_rate"], L["steps"]))
        bad = status(L, f"{c.name}/privacy.json")
        if bad:
            return f"no guarantee: inconsistent ledger ({bad})"
        train_parts.append(L.get("partition"))
        if calibrated:
            if not tp.exists():
                return f"no guarantee: calibrated release declared but {c.name} has no temperature.json"
            T = json.loads(tp.read_text())
            if not T.get("private") or not T.get("eps"):
                return f"no guarantee: {c.name}'s temperature is non-private"
            cal += T["eps"]
            bad = status(T, f"{c.name}/temperature.json")
            if bad:
                return f"no guarantee: inconsistent ledger ({bad})"
            cal_parts.append(T.get("partition"))
    per_run = [epsilon(*e, delta, "pld") for e in events]
    train = composed_epsilon(events, delta)
    tag = f" [simulation, not a guarantee: {'; '.join(sorted(blocked))}]" if blocked else ""
    if not calibrated:
        return (f"uncalibrated release: per run {min(per_run):.2f}-{max(per_run):.2f}; {len(events)} candidates: "
                f"{train:.2f}{tag}")
    # Parallel composition needs EVERY training slice disjoint from EVERY calibration slice, across
    # candidates too (candidate A's training range may be candidate B's calibration range).
    parallel = all(partitions_disjoint(a, b) for a in train_parts for b in cal_parts)
    total, how = (max(train, cal), "parallel (all train/cal partitions disjoint)") if parallel else \
                 (train + cal, "sequential (disjointness not proven)")
    return (f"calibrated release: per run {min(per_run):.2f}-{max(per_run):.2f}; {len(events)} candidates: "
            f"train {train:.2f}, cal {cal:.1f}, end-to-end {total:.2f} {how}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    R = Path(args.runs)
    best, lines, checks = {}, [], []
    for obj in ("ce", "rlcd"):
        for regime in ("thr", "auto"):
            for clip in ("std", "blind"):
                cands = []
                for run in sorted(R.glob(f"A-{regime}-{clip}-{obj}-lr*")):
                    # std is released calibrated, so it is chosen after its DP temperature (and the budget
                    # counts the temperatures). blind is released at T = 1, so it is chosen on RAW validation
                    # metrics; choosing it after a private temperature would spend the calibration budget
                    # that its uncalibrated ledger leaves out (reaudit of df9bb65, P1).
                    v = load(run / ("eval_val_ts/metrics.json" if clip == "std" else "eval_val/metrics.json"))
                    if v:
                        cands.append((v["brier"], run))
                if cands:
                    best[(obj, regime, clip)] = min(cands)[1]
    lines.append("| objective | regime | clip | lr | acc | ECE (T=1) | overconf | Brier (T=1) | ECE (T_dp) | Brier (T_dp) | resolution (T_dp) | wall s | epsilon (std: calibrated release; blind: T = 1 release) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ref = {}
    for obj in ("ce", "rlcd"):
        n = load(R / f"A-n-{obj}/eval/metrics.json")
        if n:
            ref[obj] = n
            lines.append(f"| {obj} | non-private | - | - | {n['acc']:.4f} | {n['ece']:.4f} | {n['overconf']:+.4f} | "
                         f"{n['brier']:.4f} | - | - | {n['resolution']:.5f} | - | - |")
        for regime in ("thr", "auto"):
            for clip in ("std", "blind"):
                run = best.get((obj, regime, clip))
                if not run:
                    continue
                t1, ts = load(run / "eval/metrics.json"), load(run / "eval_ts/metrics.json")
                lr = run.name.split("-lr")[-1]
                # blind's release is T = 1: its calibrated metrics are not part of it and are not shown
                tsc = (f"{ts['ece']:.4f} | {ts['brier']:.4f} | {ts['resolution']:.5f}" if clip == "std" and ts
                       else "- | - | -")
                lines.append(f"| {obj} | {regime} | {clip} | {lr} | {t1['acc']:.4f} | {t1['ece']:.4f} | "
                             f"{t1['overconf']:+.4f} | {t1['brier']:.4f} | {tsc} | {wall(run) or 0:.0f} | "
                             f"{budget(sorted(R.glob(f'A-{regime}-{clip}-{obj}-lr*')), calibrated=clip == 'std')} |")

    def get(obj, regime, clip, which):
        run = best.get((obj, regime, clip))
        return load(run / which) if run else None

    def say(tag, ok, text):
        checks.append(f"- {tag}: {'HOLDS' if ok else ('FAILS' if ok is not None else 'n/a (runs missing)')} - {text}")

    ok = None
    pairs = [(get(o, "auto", "blind", "eval/metrics.json"), get(o, "auto", "std", "eval_ts/metrics.json")) for o in ("ce", "rlcd")]
    if all(a and b for a, b in pairs):
        ok = all(a["acc"] - b["acc"] >= 0.010 and b["brier"] - a["brier"] >= 0.005 for a, b in pairs)
    say("B1", ok, "automatic clipping: blind (T = 1) beats std + DP-TS by >= 1.0 acc point and >= 0.005 Brier (CE and RLCD)")
    ok = None
    pairs = [(get(o, "thr", "blind", "eval/metrics.json"), get(o, "thr", "std", "eval_ts/metrics.json")) for o in ("ce", "rlcd")]
    if all(a and b for a, b in pairs):
        ok = all(abs(a["brier"] - b["brier"]) < 0.005 for a, b in pairs)
    say("B2", ok, "threshold clipping: blind (T = 1) and std + DP-TS within 0.005 Brier (no gain claimed)")
    ok = None
    trip = [(get(o, r, "blind", "eval/metrics.json"), get(o, r, "std", "eval/metrics.json"), ref.get(o))
            for o in ("ce", "rlcd") for r in ("thr", "auto")]
    if all(a and b and n for a, b, n in trip):
        ok = all(a["ece"] <= n["ece"] + 0.02 and b["ece"] >= 2 * n["ece"] for a, b, n in trip)
    say("B3", ok, "at T = 1 (no temperature) blind is within 0.02 ECE of non-private; std is >= 2x non-private ECE")
    ok = None
    w = [(best.get((o, r, "blind")), best.get((o, r, "std"))) for o in ("ce", "rlcd") for r in ("thr", "auto")]
    if all(a and b for a, b in w):
        ok = all((wall(a) or 0) <= 2.5 * (wall(b) or 1e9) for a, b in w)
    say("B4", ok, "cost on binary rows: blind wall-clock <= 2.5x std")
    text = "\n".join(lines + ["", "Pre-registered predictions (PLAN_BLIND.md):"] + checks)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
