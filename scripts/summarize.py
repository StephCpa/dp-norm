"""Collect every run's test metrics (T=1 and DP temperature), privacy ledger and audits into one table,
then check the pre-registered predictions of PLAN.md. Re-runnable: it only reads files.

  python scripts/summarize.py --runs runs [--md runs/summary.md]
"""
import argparse
import json
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect(runs: Path) -> dict:
    out = {}
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        m1, m2 = load(d / "eval/metrics.json"), load(d / "eval_ts/metrics.json")
        if not (m1 or (d / "audit.json").exists() or (d / "mia.json").exists()):
            continue
        r = dict(name=d.name, t1=(m1 or {}).get("overall"), ts=(m2 or {}).get("overall"),
                 temp=(load(d / "temperature.json") or {}).get("temperature"),
                 priv=load(d / "privacy.json") or {}, audit=load(d / "audit.json"), mia=load(d / "mia.json"))
        out[d.name] = r
    return out


def row(r):
    t1, ts, p = r["t1"] or {}, r["ts"] or {}, r["priv"]
    eps = p.get("eps_pld") or p.get("label_dp_eps_unit") or ""
    f = lambda d, k: f"{d[k]:.4f}" if k in d else ""
    return (f"| {r['name']} | {p.get('privacy', '')} | {eps if eps == '' else round(eps, 2)} | {f(t1, 'acc')} | {f(t1, 'ece')} | "
            f"{f(t1, 'overconf')} | {f(t1, 'brier')} | {f(t1, 'resolution')} | {'' if r['temp'] is None else round(r['temp'], 3)} | {f(ts, 'ece')} | "
            f"{f(ts, 'brier')} | {f(ts, 'resolution')} |").replace("| None |", "|  |")


def checks(R: dict) -> list[str]:
    g = lambda n, part="t1": (R.get(n) or {}).get(part)
    out = []

    def say(tag, ok, text):
        out.append(f"- {tag}: {'HOLDS' if ok else ('FAILS' if ok is not None else 'n/a (runs missing)')} - {text}")

    pairs = [("d-ce-e4", "n-ce"), ("d-brier-e4", "n-brier"), ("d-rlcd-e4", "n-rlcd")]
    ok = None
    if all(g(a) and g(b) for a, b in pairs):
        ok = all(g(a)["ece"] >= 3 * g(b)["ece"] and g(a)["overconf"] > 0.03 for a, b in pairs)
    say("P1", ok, "every DP-SGD arm at eps=4 is overconfident before recalibration (ECE >= 3x non-private, overconf > +0.03)")
    ok = None
    if all(g(a, "ts") and g(b) for a, b in pairs):
        ok = all(g(a, "ts")["ece"] <= g(b)["ece"] + 0.02 for a, b in pairs)
    say("P2", ok, "DP temperature scaling brings each DP-SGD arm within 0.02 ECE of its non-private arm")
    a, b = g("d-rlcd-e4", "ts"), g("d-ce-e4", "ts")
    say("P3", None if not (a and b) else abs(a["ece"] - b["ece"]) < 0.015 and abs(a["brier"] - b["brier"]) < 0.01,
        "no objective effect under DP-SGD after recalibration (H1 predicted to FAIL: |dECE|<0.015, |dBrier|<0.01)")
    ok = None
    if all(g(x) for x in ("l-rlcd-e4", "n-rlcd", "l-ce-e4", "n-ce")):
        ok = (g("n-rlcd")["acc"] - g("l-rlcd-e4")["acc"] <= 0.015 and g("l-rlcd-e4")["ece"] <= g("n-rlcd")["ece"] + 0.015
              and g("n-ce")["acc"] - g("l-ce-e4")["acc"] <= 0.015 and g("l-ce-e4")["ece"] <= g("n-ce")["ece"] + 0.015)
    say("P4a", ok, "label DP (corrected) at eps=4 within 1.5 acc points and 0.015 ECE of non-private")
    a, b = g("lraw-rlcd-e2"), g("l-rlcd-e2")
    say("P4b", None if not (a and b) else a["overconf"] < -0.05 and b["overconf"] > -0.03,
        "raw RR at eps=2 is underconfident (overconf < -0.05); the corrected arm is not")
    priv = [n for n in R if n.startswith(("d-", "l-")) and g(n, "ts")]
    ok = None
    if priv and g("n-rlcd"):
        base = {"ce": g("n-ce"), "brier": g("n-brier"), "rlcd": g("n-rlcd")}
        ok = all(g(n, "ts")["resolution"] < base[n.split("-")[1]]["resolution"] for n in priv
                 if base.get(n.split("-")[1]))
    say("P5", ok, "every private arm has lower resolution than its non-private arm, even after recalibration")
    a = (R.get("a-n-ce-mislabel") or {}).get("audit")
    say("P6a", None if not a else a["eps_lower_bound"] > 1.0,
        "the canary audit detects the non-private run (eps_lb > 1); if not, audits are uninformative")
    viol = [n for n in R if n.startswith("a-") and R[n]["audit"] and R[n]["priv"].get("eps_target")
            and R[n]["audit"]["eps_lower_bound"] > R[n]["priv"]["eps_target"]]
    say("P6b", None if not any(n.startswith("a-") for n in R) else not viol,
        f"no DP run's audit lower bound exceeds its eps (violations: {viol or 'none'})")
    a, b = g("l-rlcd-e4"), g("d-rlcd-e4", "ts")
    say("P7", None if not (a and b) else a["brier"] < b["brier"],
        "with public inputs, corrected label DP beats DP-SGD+TS in Brier at eps=4")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--md", default="")
    ap.add_argument("--tune-only", action="store_true", help="list the DP-SGD sweep runs by final val Brier")
    args = ap.parse_args()
    tune = []
    for d in sorted(Path(args.runs).glob("tune-*")):
        evals = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if '"eval_brier"' in l]
        if evals:
            tune.append((evals[-1]["eval_brier"], evals[-1]["eval_acc"], evals[-1]["eval_ece"], d.name))
    if tune:
        print("DP-SGD sweep (final PUBLIC val metrics; pick the lowest Brier):")
        for b, a, e, n in sorted(tune):
            print(f"  {n:<28} brier={b:.4f} acc={a:.4f} ece={e:.4f}")
    if args.tune_only:
        return
    R = collect(Path(args.runs))
    lines = ["| run | privacy | eps | acc | ECE | overconf | Brier | resolution | T_dp | ECE+TS | Brier+TS | res+TS |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"] + [row(r) for r in R.values() if r["t1"]]
    au = [f"- {n}: audit eps_lb={r['audit']['eps_lower_bound']:.2f} (AUC {r['audit']['auc']:.3f})"
          for n, r in R.items() if r["audit"]]
    mi = [f"- {n}: MIA AUC={r['mia']['auc']:.3f}, TPR@1%={r['mia']['tpr_at_1pct']:.3f}, TPR@5%={r['mia']['tpr_at_5pct']:.3f}"
          for n, r in R.items() if r["mia"]]
    text = "\n".join(lines + ["", "Audits:"] + (au or ["- none yet"]) + ["", "MIA:"] + (mi or ["- none yet"]) +
                     ["", "Pre-registered predictions (PLAN.md):"] + checks(R))
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
