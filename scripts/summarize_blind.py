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
from dpdecide.privacy import epsilon, selection_epsilon  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text())["overall"] if p.exists() else None


def wall(run: Path):  # operator-side cost measurement, from the private diagnostics
    p = run / "private_diagnostics/train_stats.jsonl"
    for line in reversed(p.read_text().splitlines() if p.exists() else []):
        r = json.loads(line)
        if "wall_sec" in r or "sec" in r:
            return r.get("wall_sec", r.get("sec"))
    return None


def budget(cands: list[Path]) -> str:
    """Per-run epsilon vs the epsilon of releasing the arm's selected model (review P0-1): selecting on
    public validation data is post-processing of every candidate, so the candidates compose. The DP
    temperature is refitted per candidate on the calibration slice, so its pure eps composes n times;
    the two slices are disjoint, so the end-to-end budget is the max of the two."""
    led = [json.loads((c / "privacy.json").read_text()) for c in cands if (c / "privacy.json").exists()]
    if not led or "sigma" not in led[0] or not led[0]["sigma"]:
        return "-"
    L, n = led[0], len(led)
    per_run = epsilon(L["sigma"], L["sampling_rate"], L["steps"], L["delta"], "pld")
    sel = selection_epsilon(L["sigma"], L["sampling_rate"], L["steps"], L["delta"], n)
    cal = []
    for c in cands:
        t = c / "temperature.json"
        if t.exists():
            cal.append(json.loads(t.read_text()).get("eps", 0.0))
    cal_eps = sum(cal)
    sim = "" if all(x.get("release_private") for x in led) else " (simulation: seeded randomness)"
    return f"{per_run:.2f} per run; {n} candidates: train {sel:.2f}, cal {cal_eps:.1f}, end-to-end {max(sel, cal_eps):.2f}{sim}"


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
                    v = load(run / "eval_val_ts/metrics.json")
                    if v:
                        cands.append((v["brier"], run))
                if cands:
                    best[(obj, regime, clip)] = min(cands)[1]
    lines.append("| objective | regime | clip | lr | acc | ECE (T=1) | overconf | Brier (T=1) | ECE (T_dp) | Brier (T_dp) | resolution (T_dp) | wall s | epsilon |")
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
                lines.append(f"| {obj} | {regime} | {clip} | {lr} | {t1['acc']:.4f} | {t1['ece']:.4f} | "
                             f"{t1['overconf']:+.4f} | {t1['brier']:.4f} | {ts['ece']:.4f} | {ts['brier']:.4f} | "
                             f"{ts['resolution']:.5f} | {wall(run) or 0:.0f} | "
                             f"{budget(sorted(R.glob(f'A-{regime}-{clip}-{obj}-lr*')))} |")

    def get(obj, regime, clip, which):
        run = best.get((obj, regime, clip))
        return load(run / which) if run else None

    def say(tag, ok, text):
        checks.append(f"- {tag}: {'HOLDS' if ok else ('FAILS' if ok is not None else 'n/a (runs missing)')} - {text}")

    ok = None
    pairs = [(get(o, "auto", "blind", "eval_ts/metrics.json"), get(o, "auto", "std", "eval_ts/metrics.json")) for o in ("ce", "rlcd")]
    if all(a and b for a, b in pairs):
        ok = all(a["acc"] - b["acc"] >= 0.010 and b["brier"] - a["brier"] >= 0.005 for a, b in pairs)
    say("B1", ok, "automatic clipping: blind beats std + DP-TS by >= 1.0 acc point and >= 0.005 Brier (CE and RLCD)")
    ok = None
    pairs = [(get(o, "thr", "blind", "eval_ts/metrics.json"), get(o, "thr", "std", "eval_ts/metrics.json")) for o in ("ce", "rlcd")]
    if all(a and b for a, b in pairs):
        ok = all(abs(a["brier"] - b["brier"]) < 0.005 for a, b in pairs)
    say("B2", ok, "threshold clipping at C = median: blind and std + DP-TS within 0.005 Brier (no gain claimed)")
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
