"""Privacy audits.

make-canaries : m single-row canaries from a pool of rows used nowhere else.
                mislabel  (audits DP-SGD): the canary carries a random wrong label and is inserted with
                          probability 1/2 (a secret coin).
                labelswap (audits label DP): the canary is always inserted; a secret coin picks which of
                          two random distinct labels it carries.
score         : one-run audit (Steinke, Nasr & Jagielski, NeurIPS 2023). Rank canaries by a membership
                score, guess the top k/2 IN and the bottom k/2 OUT, count correct guesses v out of k. For
                an eps-DP run, P[v' >= v] <= P[Binomial(k, e^eps/(1+e^eps)) >= v]; the reported lower bound
                is the largest eps that this test rejects at level beta (Bonferroni over the preset k's).
                Pure-DP form: for (eps, delta)-DP the paper adds a delta term that is negligible when
                m * delta << beta (e.g. m = 1000, delta = 1e-6); it is not included here.
mia           : loss-threshold membership attack on natural rows (members = a training slice, non-members =
                held-out rows with unseen contexts): ROC-AUC and TPR at low FPR, the metrics of
                Wang et al. 2610.04985 (their NanoJev baseline: mean AUC 0.64, TPR@5%FPR 9.3%).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

from dpdecide.metrics import roc_auc, tpr_at_fpr
from dpdecide.schema import Question, read_jsonl, select_rows, write_jsonl


def eps_lower_bound(k: int, v: int, beta: float) -> float:
    """sup{eps : P[Binom(k, p_eps) >= v] <= beta}, p_eps = 1/(1+e^-eps). 0 if even eps=0 is not rejected."""
    if k == 0:
        return 0.0
    tail = lambda e: binom.sf(v - 1, k, 1.0 / (1.0 + math.exp(-e)))
    if tail(0.0) > beta:
        return 0.0
    lo, hi = 0.0, 1.0
    while tail(hi) <= beta and hi < 64:
        lo, hi = hi, hi * 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if tail(mid) <= beta:
            lo = mid
        else:
            hi = mid
    return lo


def one_run_audit(scores: np.ndarray, truth: np.ndarray, fractions=(0.05, 0.1, 0.2), beta: float = 0.05) -> dict:
    order = np.argsort(-scores, kind="stable")
    m = len(scores)
    out = {"m": m, "beta": beta, "auc": roc_auc(scores[truth == 1], scores[truth == 0]), "tests": []}
    for f in fractions:
        k = max(2, int(round(f * m)) // 2 * 2)
        top, bottom = order[: k // 2], order[m - k // 2:]
        v = int(truth[top].sum() + (1 - truth[bottom]).sum())
        out["tests"].append(dict(k=k, correct=v, eps_lb=eps_lower_bound(k, v, beta / len(fractions))))
    out["eps_lower_bound"] = max(t["eps_lb"] for t in out["tests"])
    return out


def _rng(seed: int, key: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def make_canaries(pool: list[Question], banned_contexts: set[str], m: int, kind: str, seed: int):
    seen, picked = set(), []
    for q in pool:
        c = q.context.strip()
        if c in banned_contexts or c in seen:
            continue
        seen.add(c)
        picked.append(q)
    rng = np.random.default_rng(seed)
    if len(picked) < m:
        raise ValueError(f"only {len(picked)} usable pool rows with distinct, unbanned contexts; need {m}")
    picked = [picked[i] for i in sorted(rng.choice(len(picked), m, replace=False))]
    train, secret = [], []
    for j, q in enumerate(picked):
        r = _rng(seed, q.id or str(j))
        cid = f"canary-{j}-{q.id}"
        coin = int(r.random() < 0.5)
        if kind == "mislabel":
            wrong = int(r.integers(0, q.k - 1))
            label = wrong + (wrong >= q.answer)
            secret.append(dict(id=cid, row=q.__dict__ | {"id": cid}, label=label, include=coin))
            if coin:
                train.append(Question.from_dict(q.__dict__ | {"id": cid, "answer": label}))
        elif kind == "labelswap":
            a, b = (int(x) for x in r.choice(q.k, 2, replace=False))
            used = a if coin else b
            secret.append(dict(id=cid, row=q.__dict__ | {"id": cid}, label_a=a, label_b=b, coin=coin))
            train.append(Question.from_dict(q.__dict__ | {"id": cid, "answer": used}))
        else:
            raise ValueError(kind)
    return train, secret


def _banned(specs: list[str]) -> set[str]:
    """FILE:START:LIMIT specs whose contexts the canary pool must avoid."""
    out = set()
    for s in specs:
        path, start, limit = (s.split(":") + ["0", "0"])[:3]
        out |= {q.context.strip() for q in select_rows(read_jsonl(path), int(start), int(limit))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-canaries")
    a.add_argument("--pool", required=True)
    a.add_argument("--pool-start", type=int, default=0)
    a.add_argument("--pool-limit", type=int, default=0)
    a.add_argument("--exclude", nargs="*", default=[], help="FILE:START:LIMIT slices whose contexts are banned")
    a.add_argument("--n", type=int, default=1000)
    a.add_argument("--kind", choices=("mislabel", "labelswap"), required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out-dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--secret", required=True)
    s.add_argument("--beta", type=float, default=0.05)
    s.add_argument("--max-len", type=int, default=512)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", required=True)
    mi = sub.add_parser("mia")
    mi.add_argument("--model", required=True)
    mi.add_argument("--members", required=True, help="FILE:START:LIMIT")
    mi.add_argument("--nonmembers", required=True, help="FILE:START:LIMIT")
    mi.add_argument("--n", type=int, default=4000)
    mi.add_argument("--max-len", type=int, default=512)
    mi.add_argument("--device", default="cuda")
    mi.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "make-canaries":
        pool = select_rows(read_jsonl(args.pool), args.pool_start, args.pool_limit)
        train, secret = make_canaries(pool, _banned(args.exclude), args.n, args.kind, args.seed)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "canaries_train.jsonl", train)
        (out / "canaries_secret.json").write_text(json.dumps(dict(kind=args.kind, seed=args.seed, canaries=secret)))
        print(f"{args.kind}: {len(secret)} canaries, {len(train)} go into training -> {out}")
        return

    from dpdecide.model import DecisionLM
    model = DecisionLM.load(args.model, device=args.device)
    if args.cmd == "score":
        sec = json.loads(Path(args.secret).read_text())
        qs = [Question.from_dict(c["row"]) for c in sec["canaries"]]
        logits = model.predict(qs, args.max_len)
        lp = [np.asarray(r) - np.log(np.exp(np.asarray(r) - max(r)).sum()) - max(r) for r in logits]
        if sec["kind"] == "mislabel":
            scores = np.array([l[c["label"]] for l, c in zip(lp, sec["canaries"])])
            truth = np.array([c["include"] for c in sec["canaries"]])
        else:
            scores = np.array([l[c["label_a"]] - l[c["label_b"]] for l, c in zip(lp, sec["canaries"])])
            truth = np.array([c["coin"] for c in sec["canaries"]])
        res = one_run_audit(scores, truth, beta=args.beta) | {"kind": sec["kind"]}
    else:
        def load(spec):
            path, start, limit = (spec.split(":") + ["0", "0"])[:3]
            return select_rows(read_jsonl(path), int(start), int(limit))
        mem = load(args.members)
        mem_ctx = {q.context.strip() for q in mem}
        non = [q for q in load(args.nonmembers) if q.context.strip() not in mem_ctx]
        mem = [mem[i * len(mem) // min(args.n, len(mem))] for i in range(min(args.n, len(mem)))]
        non = [non[i * len(non) // min(args.n, len(non))] for i in range(min(args.n, len(non)))]

        def sc(qs):
            rows = model.predict(qs, args.max_len)
            return np.array([np.asarray(r)[q.answer] - max(r) - np.log(np.exp(np.asarray(r) - max(r)).sum())
                             for r, q in zip(rows, qs)])
        s_in, s_out = sc(mem), sc(non)
        res = dict(kind="loss_threshold", n_members=len(mem), n_nonmembers=len(non), auc=roc_auc(s_in, s_out),
                   tpr_at_1pct=tpr_at_fpr(s_in, s_out, 0.01), tpr_at_5pct=tpr_at_fpr(s_in, s_out, 0.05))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
