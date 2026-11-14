"""Evaluate a decision model: predictions, accuracy, ECE, Brier, NLL, overconfidence and the
reliability/resolution split, overall and per source / primitive.

  python -m dpdecide.evaluate --model runs/x --data data/test.jsonl --out runs/x/eval [--temperature runs/x/temperature.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dpdecide.metrics import softmax_rows, summarize
from dpdecide.schema import filter_k, Question, read_jsonl, select_rows


def stride_sample(items: list, n: int) -> list:
    m = min(n, len(items))
    return [items[i * len(items) // m] for i in range(m)]


def evaluate_rows(logit_rows: list[list[float]], questions: list[Question], temperature: float = 1.0) -> dict:
    probs = softmax_rows(logit_rows, temperature)
    answers = np.array([q.answer for q in questions])
    out = {"overall": summarize(probs, answers)}
    for field in ("source", "primitive"):
        vals = sorted({getattr(q, field) for q in questions})
        for v in vals:
            idx = [i for i, q in enumerate(questions) if getattr(q, field) == v]
            out[f"{field}={v}"] = summarize([probs[i] for i in idx], answers[idx])
    return out


def evaluate_model(model, questions: list[Question], max_len: int = 512, temperature: float = 1.0,
                   batch_size: int = 32) -> tuple[dict, list[list[float]]]:
    rows = model.predict(questions, max_len, batch_size)
    return evaluate_rows(rows, questions, temperature), rows


def read_temperature(arg: str | None) -> float:
    if not arg:
        return 1.0
    p = Path(arg)
    return float(json.loads(p.read_text())["temperature"]) if p.exists() else float(arg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", default=None, help="a float or a temperature.json from dpdecide.calibrate")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--only-k", default="", help="keep only rows with these option counts, e.g. 2 or 2-5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from dpdecide.model import DecisionLM
    model = DecisionLM.load(args.model, device=args.device)
    qs = filter_k(select_rows(read_jsonl(args.data), args.start, args.limit), args.only_k)
    t = read_temperature(args.temperature)
    res, rows = evaluate_model(model, qs, args.max_len, t, args.batch)
    res["temperature"] = t
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for q, r in zip(qs, rows):
            f.write(json.dumps({"id": q.id, "source": q.source, "primitive": q.primitive, "k": q.k,
                                "answer": q.answer, "logits": r}) + "\n")
    o = res["overall"]
    print(f"n={o['n']} acc={o['acc']:.4f} ece={o['ece']:.4f} brier={o['brier']:.4f} nll={o['nll']:.4f} "
          f"overconf={o['overconf']:+.4f} resolution={o['resolution']:.5f} T={t:.3f}")


if __name__ == "__main__":
    main()
