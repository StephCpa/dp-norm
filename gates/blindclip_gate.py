"""Label-blind clipping gate (toy, zero-GPU).

Standard DP-SGD clips each example by s = min(1, C/||g(x, y)||). The factor depends on the label y,
so clipping reweights labels within an input: surprising labels (large gradient) are down-weighted,
which shifts the fixed point of any proper scoring objective toward overconfidence.

Label-blind clipping uses s(x) = min(1, C / max_{y'} ||g(x, y')||), the worst case over every label the
example could have carried. Sensitivity is still <= C (the bound holds for the label it actually has),
and s no longer depends on y, so E_{y|x}[s g] = s E_{y|x}[g] = 0 at the calibrated predictor.
For RLCD (bandit feedback), the max runs over labels for the SAMPLED actions: s(x, a). Given (x, a),
E_y[c_j] = P(y = a_j | x) = pi(a_j) at calibration, so every reward c_j - pi(a_j) has zero mean and the
calibrated fixed point survives any (x, a)-measurable weighting.

Reuses the toy of h1_gate.py (same data, model, warmup, accountant, DP temperature scaling).
Arms per objective (ce, brier, rlcd): nonpriv; {std, blind} x {clip-only C=median, clip-only C=0.1 median,
DP eps in 1/4/8 at C=median, DP eps=4 at C=0.1 median}. C is the median (over PUBLIC rows) of the
realized per-example norm for std, of the label-max norm for blind.
"""
import argparse
import json
import time

import numpy as np

import h1_gate as h


def gz_for(obj, p, y, a=None):
    """dL/dlogits for labels y (n,), given sampled actions a (n, G) for rlcd."""
    n, K = p.shape
    e = np.eye(K)[y]
    if obj == "ce":
        return p - e
    if obj == "brier":
        r = 2 * (p - e)
        return p * r - p * (p * r).sum(1, keepdims=True)
    c = (a == y[:, None]).astype(float)
    pa = np.take_along_axis(p, a, 1)
    r = c - pa
    G = a.shape[1]
    adv = r - (r.sum(1, keepdims=True) - r) / (G - 1)
    g = np.zeros((n, K))
    for j in range(G):
        g -= adv[:, j:j + 1] * (np.eye(K)[a[:, j]] - p)
    return g / G


def sample_actions(p, rng):
    return (p.cumsum(1)[:, None, :] > rng.random((len(p), h.G, 1))).argmax(2)


def norms(P, x, obj, p, y, a, blind):
    """Realized per-example squared norm, and (if blind) the max over all K hypothetical labels."""
    gz = gz_for(obj, p, y, a)
    hh, da, sq = h.per_example_grads(P, x, gz)
    if not blind:
        return gz, hh, da, sq
    worst = sq.copy()
    for yy in range(h.K):
        _, _, sq_y = h.per_example_grads(P, x, gz_for(obj, p, np.full(len(y), yy), a))
        worst = np.maximum(worst, sq_y)
    return gz, hh, da, worst


def run(cfg, data, seed):
    rng = np.random.default_rng(seed)
    T_, (xp, yp), (xq, yq), (xc, yc), (xv, yv), (xt, yt) = data
    P = h.init_student(np.random.default_rng(1000 + seed))
    opt = h.Adam(P, 3e-3)
    for _ in range(60):  # public warmup, identical to h1_gate
        idx = rng.integers(0, h.N_PUB, 64)
        hh, p = h.forward(P, xp[idx])
        gz = h.dlogits("ce", p, yp[idx], rng)
        _, da, _ = h.per_example_grads(P, xp[idx], gz)
        opt.step(P, h.summed_grad(xp[idx], gz, hh, da, np.ones(len(idx)) / len(idx)))
    blind = cfg["clip"] == "blind"
    _, p = h.forward(P, xp)
    a_pub = sample_actions(p, rng) if cfg["obj"] == "rlcd" else None
    _, _, _, sq_pub = norms(P, xp, cfg["obj"], p, yp, a_pub, blind)
    C = (cfg["cmult"] or 1.0) * np.median(np.sqrt(sq_pub))
    q = h.B / h.N_PRIV
    steps = int(h.EPOCHS / q)
    sigma = h.calibrate_sigma(cfg["eps"], q, steps) if cfg["mode"] == "dp" else 0.0
    opt = h.Adam(P, cfg["lr"])
    scales = []
    for t in range(steps):
        m = rng.random(h.N_PRIV) < q
        xb, yb = xq[m], yq[m]
        _, p = h.forward(P, xb)
        a = sample_actions(p, rng) if cfg["obj"] == "rlcd" else None
        gz, hh, da, sq = norms(P, xb, cfg["obj"], p, yb, a, blind)
        if cfg["mode"] == "nonpriv":
            s = np.ones(len(yb))
        else:
            s = np.minimum(1.0, C / (np.sqrt(sq) + 1e-12))
        scales.append(s.mean() if len(s) else 1.0)
        g = h.summed_grad(xb, gz, hh, da, s)
        if cfg["mode"] == "dp":
            for k in g:
                g[k] = g[k] + rng.standard_normal(g[k].shape) * sigma * C
        for k in g:
            g[k] = g[k] / h.B
        opt.step(P, g, 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / steps)))
    logits = lambda x: np.tanh(x @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]
    out = {"val": h.metrics(h.softmax(logits(xv)), yv), "test": h.metrics(h.softmax(logits(xt)), yt),
           "sigma": sigma, "C": C, "mean_scale": float(np.mean(scales))}
    tmp = h.dp_temperature(logits, xc, yc, cfg["eps"] if cfg["mode"] == "dp" else None, rng)
    out["T"] = float(tmp)
    out["test_ts"] = h.metrics(h.softmax(logits(xt) / tmp), yt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="blindclip_results.json")
    ap.add_argument("--objs", default="ce,brier,rlcd")
    args = ap.parse_args()
    rng = np.random.default_rng(7)  # same data as h1_gate
    T_ = h.make_teacher(rng)
    xs = [h.sample(rng, T_, n) for n in (h.N_PUB, h.N_PRIV, h.N_CAL, h.N_VAL, h.N_TEST)]
    data = (T_,) + tuple((x, y) for x, y, _ in xs)
    print("bayes oracle:", {k: round(float(v), 4) for k, v in h.metrics(xs[-1][2], xs[-1][1]).items()}, flush=True)
    configs = []
    for obj in args.objs.split(","):
        configs.append(dict(obj=obj, mode="nonpriv", clip="std", eps=None, cmult=None))
        for clip in ("std", "blind"):
            for cm in (1.0, 0.1):
                configs.append(dict(obj=obj, mode="cliponly", clip=clip, eps=None, cmult=cm))
            for eps in (1, 4, 8):
                configs.append(dict(obj=obj, mode="dp", clip=clip, eps=eps, cmult=1.0))
            configs.append(dict(obj=obj, mode="dp", clip=clip, eps=4, cmult=0.1))
    lrs = (3e-4, 1e-3, 3e-3, 1e-2)
    results, t0 = [], time.time()
    for cfg in configs:
        best = None
        for lr in lrs:  # lr chosen on val Brier, seed 0 (same protocol as h1_gate)
            r = run(cfg | {"lr": lr}, data, seed=0)
            if best is None or r["val"]["brier"] < best[1]:
                best = (lr, r["val"]["brier"])
        runs = [run(cfg | {"lr": best[0]}, data, seed=s) for s in range(1, args.seeds + 1)]
        agg = {}
        for split in ("test", "test_ts"):
            for k in runs[0][split]:
                v = np.array([r[split][k] for r in runs])
                agg[f"{split}.{k}"] = (float(v.mean()), float(v.std()))
        for k in ("sigma", "C", "mean_scale", "T"):
            agg[k] = float(np.mean([r[k] for r in runs]))
        results.append(dict(cfg=cfg, lr=best[0], agg=agg))
        a = agg
        print(f"{cfg['obj']:>5} {cfg['mode']:>8} {cfg['clip']:>5} eps={str(cfg['eps']):>4} cm={str(cfg['cmult']):>4} "
              f"lr={best[0]:.0e} | acc {a['test.acc'][0]:.3f} ece {a['test.ece'][0]:.3f}±{a['test.ece'][1]:.3f} "
              f"over {a['test.overconf'][0]:+.3f} brier {a['test.brier'][0]:.4f} res {a['test.res_top'][0]:.4f} "
              f"| +TS ece {a['test_ts.ece'][0]:.3f} brier {a['test_ts.brier'][0]:.4f} res {a['test_ts.res_top'][0]:.4f} "
              f"| scale {a['mean_scale']:.2f} [{time.time()-t0:.0f}s]", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
