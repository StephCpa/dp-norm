"""Fair rerun of auto_gate.py (pre-registration: PREREG_AUTO_FAIR.md).

Differences from auto_gate.py: std + DP-TS chooses its lr by validation Brier after its DP temperature;
blind chooses by validation Brier at T = 1 (and blind + DP-TS like std); one step higher in the lr grid;
the same gamma for both arms (0.01 x median STANDARD per-example norm on public rows); paired effects.
"""
import json
import time

import numpy as np

import blindclip_gate as b
import h1_gate as h

LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
SEEDS = (1, 2, 3)
T975_DF2 = 4.303


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
    _, _, _, sq_pub = b.norms(P, xp, cfg["obj"], p, yp, a_pub, False)  # STANDARD norms for both arms
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
    sm = h.softmax
    return {"val": h.metrics(sm(logits(xv)), yv), "val_ts": h.metrics(sm(logits(xv) / tmp), yv),
            "test": h.metrics(sm(logits(xt)), yt), "test_ts": h.metrics(sm(logits(xt) / tmp), yt),
            "T": float(tmp), "gamma": float(gamma)}


def pick(results, key):
    return min(results, key=lambda lr: results[lr][key]["brier"])


def paired(a, b):
    d = np.array(a) - np.array(b)
    half = T975_DF2 * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def main():
    rng = np.random.default_rng(7)
    T_ = h.make_teacher(rng)
    xs = [h.sample(rng, T_, n) for n in (h.N_PUB, h.N_PRIV, h.N_CAL, h.N_VAL, h.N_TEST)]
    data = (T_,) + tuple((x, y) for x, y, _ in xs)
    out, t0 = [], time.time()
    for obj in ("ce", "brier", "rlcd"):
        for eps in (1, 4, 8):
            arms = {}
            for clip in ("std", "blind"):
                sweep = {lr: run(dict(obj=obj, clip=clip, eps=eps, lr=lr), data, 0) for lr in LRS}
                rules = {"std": [("std+TS", "val_ts", "test_ts")],
                         "blind": [("blind", "val", "test"), ("blind+TS", "val_ts", "test_ts")]}[clip]
                for name, sel, rep in rules:
                    lr = pick(sweep, sel)
                    rs = [run(dict(obj=obj, clip=clip, eps=eps, lr=lr), data, s) for s in SEEDS]
                    arms[name] = dict(lr=lr, sel=sel, rep=rep, sweep={str(k): v[sel]["brier"] for k, v in sweep.items()},
                                      runs=[r[rep] for r in rs], T=[r["T"] for r in rs])
            row = dict(obj=obj, eps=eps, arms={k: {kk: vv for kk, vv in v.items()} for k, v in arms.items()})
            base = arms["std+TS"]["runs"]
            for name in ("blind", "blind+TS"):
                cur = arms[name]["runs"]
                row[f"{name} - std+TS"] = {
                    "acc": paired([r["acc"] for r in cur], [r["acc"] for r in base]),
                    "brier": paired([r["brier"] for r in cur], [r["brier"] for r in base]),
                    "ece": paired([r["ece"] for r in cur], [r["ece"] for r in base])}
            out.append(row)
            msg = " | ".join(f"{n} lr={a['lr']:.0e} acc {np.mean([r['acc'] for r in a['runs']]):.4f} "
                             f"brier {np.mean([r['brier'] for r in a['runs']]):.4f} ece {np.mean([r['ece'] for r in a['runs']]):.3f}"
                             for n, a in arms.items())
            d = row["blind - std+TS"]
            print(f"{obj:>5} eps={eps} | {msg}\n      blind - std+TS: acc {d['acc'][0]*100:+.2f} pts "
                  f"[{d['acc'][1]*100:+.2f}, {d['acc'][2]*100:+.2f}]  brier {d['brier'][0]:+.4f} "
                  f"[{d['brier'][1]:+.4f}, {d['brier'][2]:+.4f}]  | std+TS sweep(val_ts brier) "
                  + " ".join(f"{k}:{v:.4f}" for k, v in arms['std+TS']['sweep'].items())
                  + f" [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open("auto_fair_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
