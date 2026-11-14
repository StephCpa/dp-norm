"""H1 zero-GPU gate: under DP-SGD, does the RLCD objective stay better calibrated than CE?

Pipeline mirrors eve-rlcd at toy scale:
  public warmup (non-private CE on N_PUB labeled rows)
  -> private stage on N_PRIV rows, Poisson sampling (expected batch B), T steps, per-example clipping + Gaussian noise
  -> optional DP temperature scaling on a disjoint private calibration slice (exponential mechanism on Brier)
Objectives in the private stage:
  ce    : full-label cross-entropy
  brier : full-label Brier loss (the full-information version of RLCD's objective)
  rlcd  : bandit feedback only, reward c - p_a, G=4 samples, leave-one-out baseline
Privacy modes: nonpriv (no clip, no noise), cliponly (clip, no noise), dp (clip + noise calibrated with an RDP accountant).
Model: 1-hidden-layer tanh MLP, all weights trained; per-example gradient norms computed exactly (outer-product factorization).
Data: teacher MLP; each example is easy (logit scale 1.5) or ambiguous (scale 0.3, ~25%), ambiguity set by a
learnable gate on x; input norms vary (Gamma(8, 1/8) scale) so per-example gradient norms are heterogeneous.
Accounting here is RDP at delta=1e-5 (toy); the dpdecide package itself defaults to PLD at delta=1e-6.
"""
import argparse
import itertools
import json
import time

import numpy as np
import dp_accounting as dpa
from dp_accounting import rdp

D, H, K = 24, 64, 4
N_PUB, N_PRIV, N_CAL, N_VAL, N_TEST = 500, 32000, 4000, 4000, 40000
B, EPOCHS, G = 128, 2, 4
DELTA = 1e-5


# ---------------------------------------------------------------- data
def make_teacher(rng):
    return dict(W1=rng.standard_normal((H, D)) / np.sqrt(D) * 1.5, b1=rng.standard_normal(H) * 0.3,
                W2=rng.standard_normal((K, H)) / np.sqrt(H) * 3.0, gate=rng.standard_normal(D) / np.sqrt(D),
                gate_thr=0.55)


def teacher_probs(T, x):
    z = np.tanh(x @ T["W1"].T + T["b1"]) @ T["W2"].T
    ambiguous = (x @ T["gate"]) > T["gate_thr"]
    scale = np.where(ambiguous, 0.3, 1.5)[:, None]
    return softmax(z * scale)


def sample(rng, T, n):
    x = rng.standard_normal((n, D))
    # norms vary across rows (heavier tails), which is what makes clipping bite unevenly
    x *= rng.gamma(8.0, 0.125, size=(n, 1))
    p = teacher_probs(T, x)
    y = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return x, y, p


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- model
def init_student(rng):
    return dict(W1=rng.standard_normal((H, D)) / np.sqrt(D), b1=np.zeros(H),
                W2=rng.standard_normal((K, H)) / np.sqrt(H) * 0.1, b2=np.zeros(K))


def forward(P, x):
    h = np.tanh(x @ P["W1"].T + P["b1"])
    return h, softmax(h @ P["W2"].T + P["b2"])


def rr_params(eps):
    pk = np.exp(eps) / (np.exp(eps) + K - 1)
    return pk, (1 - pk) / (K - 1)


def dlogits(obj, p, y, rng, ldp=None):
    """Per-example gradient of the loss w.r.t. the logits, (n, K). ldp=(p_keep, q) marks y as an
    RR-privatized label: rlcd then uses the debiased outcome, ce_fc the forward-corrected NLL."""
    n = len(y)
    e = np.eye(K)[y]
    if obj == "ce_fc":
        pk, qq = ldp
        v = np.zeros((n, K))
        v[np.arange(n), y] = -(pk - qq) / (qq + (pk - qq) * p[np.arange(n), y])
        return p * v - p * (p * v).sum(1, keepdims=True)
    if obj == "ce":
        return p - e
    if obj == "brier":
        r = 2 * (p - e)
        return p * r - p * (p * r).sum(1, keepdims=True)
    if obj == "rlcd":
        a = (p.cumsum(1)[:, None, :] > rng.random((n, G, 1))).argmax(2)          # (n, G)
        c = (a == y[:, None]).astype(float)
        if ldp is not None and obj == "rlcd":
            c = (c - ldp[1]) / (ldp[0] - ldp[1])
        pa = np.take_along_axis(p, a, 1)
        r = c - pa
        adv = r - (r.sum(1, keepdims=True) - r) / (G - 1)
        g = np.zeros((n, K))
        for j in range(G):
            g -= adv[:, j:j + 1] * (np.eye(K)[a[:, j]] - p)
        return g / G
    raise ValueError(obj)


def per_example_grads(P, x, gz):
    """Return per-example factors and squared norms; gradients are never materialized per example."""
    h = np.tanh(x @ P["W1"].T + P["b1"])
    da = (gz @ P["W2"]) * (1 - h ** 2)                                            # (n, H)
    sq = (gz ** 2).sum(1) * ((h ** 2).sum(1) + 1) + (da ** 2).sum(1) * ((x ** 2).sum(1) + 1)
    return h, da, sq


def summed_grad(x, gz, h, da, scale):
    s = scale[:, None]
    return dict(W2=(s * gz).T @ h, b2=(s * gz).sum(0), W1=(s * da).T @ x, b1=(s * da).sum(0))


class Adam:
    def __init__(self, P, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, g, lr_scale=1.0):
        self.t += 1
        for k in P:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            P[k] -= self.lr * lr_scale * mh / (np.sqrt(vh) + 1e-8)


# ---------------------------------------------------------------- privacy
_SIGMA_CACHE = {}


def calibrate_sigma(eps, q, steps):
    key = (eps, q, steps)
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = dpa.calibrate_dp_mechanism(
            lambda: rdp.RdpAccountant(),
            lambda s: dpa.SelfComposedDpEvent(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(s)), steps),
            eps, DELTA, bracket_interval=dpa.LowerEndpointAndGuess(0.3, 1.0))
    return _SIGMA_CACHE[key]


# ---------------------------------------------------------------- metrics
def metrics(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    hit = (pred == y).astype(float)
    ids = np.clip(np.floor(conf * bins + 1e-9).astype(int), 0, bins - 1)
    ece = rel = res = 0.0
    abar = hit.mean()
    for b in range(bins):
        m = ids == b
        if m.any():
            w = m.mean()
            ece += w * abs(hit[m].mean() - conf[m].mean())
            rel += w * (hit[m].mean() - conf[m].mean()) ** 2
            res += w * (hit[m].mean() - abar) ** 2
    brier = ((p - np.eye(K)[y]) ** 2).sum(1).mean()
    return dict(acc=hit.mean(), ece=ece, brier=brier, overconf=conf.mean() - hit.mean(), res_top=res, rel_top=rel)


def dp_temperature(logits_fn, x, y, eps, rng, grid=np.exp(np.linspace(np.log(0.25), np.log(4), 41))):
    """Exponential mechanism over a temperature grid; utility -sum Brier, per-row Brier in [0,2] so
    replace-one sensitivity is 2 (add/remove: also <= 2). eps=None -> argmin (non-private)."""
    z = logits_fn(x)
    util = np.array([-((softmax(z / t) - np.eye(K)[y]) ** 2).sum(1).sum() for t in grid])
    if eps is None:
        return grid[util.argmax()]
    w = eps * (util - util.max()) / (2 * 2.0)
    pr = np.exp(w)
    return grid[rng.choice(len(grid), p=pr / pr.sum())]


# ---------------------------------------------------------------- training
def run(cfg, data, seed):
    rng = np.random.default_rng(seed)
    T_, (xp, yp), (xq, yq), (xc, yc), (xv, yv), (xt, yt) = data
    P = init_student(np.random.default_rng(1000 + seed))
    # public warmup: 60 steps of non-private CE, batch 64
    opt = Adam(P, 3e-3)
    for _ in range(60):
        idx = rng.integers(0, N_PUB, 64)
        h, p = forward(P, xp[idx])
        gz = dlogits("ce", p, yp[idx], rng)
        _, da, _ = per_example_grads(P, xp[idx], gz)
        g = summed_grad(xp[idx], gz, h, da, np.ones(len(idx)) / len(idx))
        opt.step(P, g)
    # median per-example gradient norm at the start of the private stage (from PUBLIC data)
    h, p = forward(P, xp)
    gz = dlogits("ce" if cfg["obj"] == "ce_fc" else cfg["obj"], p, yp, rng)
    _, _, sq = per_example_grads(P, xp, gz)
    C = (cfg["cmult"] or 1.0) * np.median(np.sqrt(sq))
    q = B / N_PRIV
    steps = int(EPOCHS / q)
    ldp = None
    if cfg["mode"] in ("labeldp", "labeldp_raw"):
        pk, qq = rr_params(cfg["eps"])
        flip = rng.random(N_PRIV) > pk
        other = rng.integers(0, K - 1, N_PRIV)
        other = other + (other >= yq)
        yq = np.where(flip, other, yq)
        ldp = (pk, qq) if cfg["mode"] == "labeldp" else None
    sigma = calibrate_sigma(cfg["eps"], q, steps) if cfg["mode"] == "dp" else 0.0
    opt = Adam(P, cfg["lr"])
    for t in range(steps):
        m = rng.random(N_PRIV) < q
        xb, yb = xq[m], yq[m]
        h, p = forward(P, xb)
        gz = dlogits(cfg["obj"], p, yb, rng, ldp)
        h, da, sq = per_example_grads(P, xb, gz)
        if cfg["mode"] in ("nonpriv", "labeldp", "labeldp_raw"):
            scale = np.ones(len(yb))
        else:
            scale = np.minimum(1.0, C / (np.sqrt(sq) + 1e-12))
        g = summed_grad(xb, gz, h, da, scale)
        if cfg["mode"] == "dp":
            for k in g:
                g[k] = g[k] + rng.standard_normal(g[k].shape) * sigma * C
        for k in g:
            g[k] = g[k] / B  # expected batch size: dividing by the realized size would leak it
        lr_scale = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / steps))
        opt.step(P, g, lr_scale)
    logits = lambda x: np.tanh(x @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]
    out = {"val": metrics(softmax(logits(xv)), yv), "test": metrics(softmax(logits(xt)), yt), "sigma": sigma, "C": C}
    eps_cal = cfg["eps"] if cfg["mode"] in ("dp", "labeldp", "labeldp_raw") else None
    tmp = dp_temperature(logits, xc, yc, eps_cal, rng)
    out["T"] = float(tmp)
    out["test_ts"] = metrics(softmax(logits(xt) / tmp), yt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="h1_results.json")
    ap.add_argument("--only-labeldp", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    T_ = make_teacher(rng)
    xs = [sample(rng, T_, n) for n in (N_PUB, N_PRIV, N_CAL, N_VAL, N_TEST)]
    data = (T_,) + tuple((x, y) for x, y, _ in xs)
    xt, yt, pt = xs[-1]
    print("bayes oracle on test:", {k: round(v, 4) for k, v in metrics(pt, yt).items()})
    configs = []
    for eps in (1, 4, 8):
        configs.append(dict(obj="rlcd", mode="labeldp", eps=eps, cmult=None))
        configs.append(dict(obj="rlcd", mode="labeldp_raw", eps=eps, cmult=None))
        configs.append(dict(obj="ce_fc", mode="labeldp", eps=eps, cmult=None))
    for obj in ([] if args.only_labeldp else ("ce", "brier", "rlcd")):
        configs.append(dict(obj=obj, mode="nonpriv", eps=None, cmult=None))
        for cm in (0.1, 1.0):
            configs.append(dict(obj=obj, mode="cliponly", eps=None, cmult=cm))
            for eps in (1, 4, 8):
                configs.append(dict(obj=obj, mode="dp", eps=eps, cmult=cm))
    lrs = (3e-4, 1e-3, 3e-3, 1e-2)
    results = []
    t0 = time.time()
    for cfg in configs:
        # lr picked on validation Brier with seed 0 (non-private tuning, stated as a caveat)
        best = None
        for lr in lrs:
            r = run(cfg | {"lr": lr}, data, seed=0)
            if best is None or r["val"]["brier"] < best[1]:
                best = (lr, r["val"]["brier"])
        runs = [run(cfg | {"lr": best[0]}, data, seed=s) for s in range(1, args.seeds + 1)]
        agg = {}
        for split in ("test", "test_ts"):
            for k in runs[0][split]:
                v = np.array([r[split][k] for r in runs])
                agg[f"{split}.{k}"] = (float(v.mean()), float(v.std()))
        agg["sigma"] = runs[0]["sigma"]
        agg["T"] = float(np.mean([r["T"] for r in runs]))
        results.append(dict(cfg=cfg, lr=best[0], agg=agg))
        c = cfg
        tag = f"{c['obj']:>5} {c['mode']:>8} eps={str(c['eps']):>4} cm={str(c['cmult']):>4} lr={best[0]:.0e}"
        a = agg
        print(f"{tag} | acc {a['test.acc'][0]:.3f} ece {a['test.ece'][0]:.3f} over {a['test.overconf'][0]:+.3f} "
              f"brier {a['test.brier'][0]:.3f} res {a['test.res_top'][0]:.4f} | +TS(T={a['T']:.2f}) ece "
              f"{a['test_ts.ece'][0]:.3f} brier {a['test_ts.brier'][0]:.3f} | sigma {a['sigma']:.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
