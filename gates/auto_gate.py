"""Automatic clipping (per-example normalization, Bu et al. NeurIPS 2023 AUTO-S) vs its label-blind version.

std-auto  : s = 1 / (||g(x, y)|| + gamma)              (label-dependent; ||s g|| < 1)
blind-auto: s = 1 / (max_y' ||g(x, y')|| + gamma)      (label-blind;     ||s g|| < 1)
Sensitivity 1 in both; noise sigma * 1. gamma = 0.01 x median public norm (AUTO-S stability constant).
DP at eps = 1, 4, 8; std-auto also with its DP temperature. Same toy and protocol as blindclip_gate.
"""
import json
import time

import numpy as np

import blindclip_gate as b
import h1_gate as h


def run(cfg, data, seed):
    rng = np.random.default_rng(seed)
    T_, (xp, yp), (xq, yq), (xc, yc), (xv, yv), (xt, yt) = data
    P = h.init_student(np.random.default_rng(1000 + seed))
    opt = h.Adam(P, 3e-3)
    for _ in range(60):
        idx = rng.integers(0, h.N_PUB, 64)
        hh, p = h.forward(P, xp[idx])
        gz = h.dlogits("ce", p, yp[idx], rng)
        _, da, _ = h.per_example_grads(P, xp[idx], gz)
        opt.step(P, h.summed_grad(xp[idx], gz, hh, da, np.ones(len(idx)) / len(idx)))
    blind = cfg["clip"] == "blind"
    _, p = h.forward(P, xp)
    a_pub = b.sample_actions(p, rng) if cfg["obj"] == "rlcd" else None
    _, _, _, sq_pub = b.norms(P, xp, cfg["obj"], p, yp, a_pub, blind)
    gamma = 0.01 * np.median(np.sqrt(sq_pub))
    q = h.B / h.N_PRIV
    steps = int(h.EPOCHS / q)
    sigma = h.calibrate_sigma(cfg["eps"], q, steps)
    opt = h.Adam(P, cfg["lr"])
    for t in range(steps):
        m = rng.random(h.N_PRIV) < q
        xb, yb = xq[m], yq[m]
        _, p = h.forward(P, xb)
        a = b.sample_actions(p, rng) if cfg["obj"] == "rlcd" else None
        gz, hh, da, sq = b.norms(P, xb, cfg["obj"], p, yb, a, blind)
        s = 1.0 / (np.sqrt(sq) + gamma)
        g = h.summed_grad(xb, gz, hh, da, s)
        for k in g:
            g[k] = (g[k] + rng.standard_normal(g[k].shape) * sigma) / h.B
        opt.step(P, g, 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / steps)))
    logits = lambda x: np.tanh(x @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]
    tmp = h.dp_temperature(logits, xc, yc, cfg["eps"], rng)
    return {"val": h.metrics(h.softmax(logits(xv)), yv), "test": h.metrics(h.softmax(logits(xt)), yt),
            "test_ts": h.metrics(h.softmax(logits(xt) / tmp), yt), "T": float(tmp)}


def main():
    rng = np.random.default_rng(7)
    T_ = h.make_teacher(rng)
    xs = [h.sample(rng, T_, n) for n in (h.N_PUB, h.N_PRIV, h.N_CAL, h.N_VAL, h.N_TEST)]
    data = (T_,) + tuple((x, y) for x, y, _ in xs)
    out, t0 = [], time.time()
    for obj in ("ce", "brier", "rlcd"):
        for clip in ("std", "blind"):
            for eps in (1, 4, 8):
                cfg = dict(obj=obj, clip=clip, eps=eps)
                best = None
                for lr in (3e-4, 1e-3, 3e-3, 1e-2):
                    r = run(cfg | {"lr": lr}, data, 0)
                    if best is None or r["val"]["brier"] < best[1]:
                        best = (lr, r["val"]["brier"])
                rs = [run(cfg | {"lr": best[0]}, data, s) for s in (1, 2, 3)]
                m = lambda sp, k: (np.mean([r[sp][k] for r in rs]), np.std([r[sp][k] for r in rs]))
                row = dict(cfg=cfg, lr=best[0], **{f"{sp}.{k}": m(sp, k) for sp in ("test", "test_ts")
                                                   for k in rs[0]["test"]})
                out.append(row)
                print(f"{obj:>5} auto-{clip:<5} eps={eps} lr={best[0]:.0e} | acc {row['test.acc'][0]:.3f}±{row['test.acc'][1]:.3f} "
                      f"ece {row['test.ece'][0]:.3f} over {row['test.overconf'][0]:+.3f} brier {row['test.brier'][0]:.4f} "
                      f"res {row['test.res_top'][0]:.4f} | +TS acc {row['test_ts.acc'][0]:.3f} ece {row['test_ts.ece'][0]:.3f} "
                      f"brier {row['test_ts.brier'][0]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open("auto_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
