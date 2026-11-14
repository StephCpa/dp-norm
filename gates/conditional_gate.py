"""Does a single DP temperature fix clipping-induced miscalibration everywhere, or only on average?

Standard clipping down-weights large-gradient examples, and per-example gradient norms scale with the
input norm, so the reweighting (and the overconfidence it causes) should vary with ||x||. A global
temperature corrects the average. Label-blind clipping should leave no input-dependent residue.

Same toy, data and protocol as h1_gate / blindclip_gate (lr taken from blindclip_results.json).
Reports, per arm, overconfidence (mean conf - acc) and ECE within quartiles of ||x|| on the test set,
plus the worst-quartile |overconfidence|, after the arm's own DP temperature (std) or none (blind).
"""
import json

import numpy as np

import blindclip_gate as b
import h1_gate as h


def train(cfg, data, seed):
    """blindclip_gate.run, returning the trained logit function and the DP temperature."""
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
    C = (cfg["cmult"] or 1.0) * np.median(np.sqrt(sq_pub))
    q = h.B / h.N_PRIV
    steps = int(h.EPOCHS / q)
    sigma = h.calibrate_sigma(cfg["eps"], q, steps) if cfg["mode"] == "dp" else 0.0
    opt = h.Adam(P, cfg["lr"])
    for t in range(steps):
        m = rng.random(h.N_PRIV) < q
        xb, yb = xq[m], yq[m]
        _, p = h.forward(P, xb)
        a = b.sample_actions(p, rng) if cfg["obj"] == "rlcd" else None
        gz, hh, da, sq = b.norms(P, xb, cfg["obj"], p, yb, a, blind)
        s = np.ones(len(yb)) if cfg["mode"] == "nonpriv" else np.minimum(1.0, C / (np.sqrt(sq) + 1e-12))
        g = h.summed_grad(xb, gz, hh, da, s)
        if cfg["mode"] == "dp":
            for k in g:
                g[k] = g[k] + rng.standard_normal(g[k].shape) * sigma * C
        for k in g:
            g[k] = g[k] / h.B
        opt.step(P, g, 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / steps)))
    logits = lambda x: np.tanh(x @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]
    tmp = h.dp_temperature(logits, xc, yc, cfg["eps"] if cfg["mode"] == "dp" else None, rng)
    return logits, tmp


def by_quartile(probs, y, x):
    nrm = np.linalg.norm(x, axis=1)
    edges = np.quantile(nrm, [0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        m = (nrm >= edges[i]) & (nrm <= edges[i + 1])
        mt = h.metrics(probs[m], y[m])
        out.append((mt["overconf"], mt["ece"], mt["acc"]))
    return out


def main():
    rng = np.random.default_rng(7)
    T_ = h.make_teacher(rng)
    xs = [h.sample(rng, T_, n) for n in (h.N_PUB, h.N_PRIV, h.N_CAL, h.N_VAL, h.N_TEST)]
    data = (T_,) + tuple((x, y) for x, y, _ in xs)
    xt, yt = xs[-1][0], xs[-1][1]
    lr_of = {json.dumps(r["cfg"], sort_keys=True): r["lr"] for r in json.load(open("blindclip_results.json"))}
    arms = []
    for obj in ("ce", "rlcd"):
        arms.append(dict(obj=obj, mode="nonpriv", clip="std", eps=None, cmult=None))
        for clip in ("std", "blind"):
            arms.append(dict(obj=obj, mode="dp", clip=clip, eps=4, cmult=1.0))
            arms.append(dict(obj=obj, mode="dp", clip=clip, eps=4, cmult=0.1))
    rows = []
    print("overconfidence (conf - acc) by ||x|| quartile, test set, mean over seeds 1-3\n"
          "std arms after their own DP temperature (T=1 shown too); blind and non-private arms at T=1")
    for cfg in arms:
        lr = lr_of[json.dumps(cfg, sort_keys=True)]
        res_t1, res_ts = [], []
        for s in (1, 2, 3):
            logits, tmp = train(cfg | {"lr": lr}, data, s)
            z = logits(xt)
            res_t1.append(by_quartile(h.softmax(z), yt, xt))
            res_ts.append(by_quartile(h.softmax(z / tmp), yt, xt))
        r1, rt = np.array(res_t1).mean(0), np.array(res_ts).mean(0)
        tag = f"{cfg['obj']:>4} {cfg['mode']:>7} {cfg['clip']:>5} cm={str(cfg['cmult']):>4}"
        fmt = lambda r: " ".join(f"{v:+.3f}" for v in r[:, 0]) + f" | worst |over| {np.abs(r[:, 0]).max():.3f} | mean ECE {r[:, 1].mean():.3f}"
        print(f"{tag}  T=1 : {fmt(r1)}")
        if cfg["clip"] == "std" and cfg["mode"] == "dp":
            print(f"{'':>30}  +TS : {fmt(rt)}")
        rows.append(dict(cfg=cfg, t1=r1.tolist(), ts=rt.tolist()))
    json.dump(rows, open("conditional_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
