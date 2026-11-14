"""Calibration quota under a fixed private budget (pre-registration: PREREG_CALQUOTA.md).

Blind(N) vs std(N - m) + DP calibrator(m), binary CE, automatic clipping, same toy as auto_gate_fair.py.
"""
import json
import os
import time

import numpy as np

import h1_gate as h  # noqa: E402

h.K = 2  # binary, like GPU stage 1; module functions read K at call time
import blindclip_gate as b  # noqa: E402

LRS = (1e-3, 3e-3, 1e-2, 3e-2)
SEEDS = (1, 2, 3, 4, 5)
MS = tuple(int(m) for m in os.environ.get("MS", "64,256,1024,4096").split(","))  # pre-registered default
T975_DF4 = 2.776
# Calibrator search ranges. The pre-registered run used T in [0.25, 4] and a in [0.25, 4]; the reaudit of
# df9bb65 found every public TS hitting T = 4. Env overrides (counts unchanged): T_MAX, A_MIN, B_MAX.
TGRID = np.exp(np.linspace(np.log(0.25), np.log(float(os.environ.get("T_MAX", 4))), 41))
AGRID = np.exp(np.linspace(np.log(float(os.environ.get("A_MIN", 0.25))), np.log(4), 21))
BGRID = np.linspace(-float(os.environ.get("B_MAX", 1.5)), float(os.environ.get("B_MAX", 1.5)), 13)


def make_data(seed, n_total, spec):
    rng = np.random.default_rng(10_000 + seed)
    T = h.make_teacher(rng)
    if spec == "well":
        T["gate_thr"] = np.inf  # no ambiguity gate: teacher = tanh MLP of the student's width, realizable
    sets = [h.sample(rng, T, n)[:2] for n in (h.N_PUB, n_total, h.N_VAL, h.N_TEST)]
    return sets


def train(clip, lr, eps, pub, xq, yq, seed):
    rng = np.random.default_rng(seed)
    xp, yp = pub
    P = h.init_student(np.random.default_rng(1000 + seed))
    opt = h.Adam(P, 3e-3)
    for _ in range(60):
        idx = rng.integers(0, len(xp), 64)
        hh, p = h.forward(P, xp[idx])
        gz = h.dlogits("ce", p, yp[idx], rng)
        _, da, _ = h.per_example_grads(P, xp[idx], gz)
        opt.step(P, h.summed_grad(xp[idx], gz, hh, da, np.ones(len(idx)) / len(idx)))
    _, p = h.forward(P, xp)
    _, _, _, sq_pub = b.norms(P, xp, "ce", p, yp, None, False)
    gamma = 0.01 * np.median(np.sqrt(sq_pub))
    n = len(xq)
    q = h.B / n
    steps = int(h.EPOCHS / q)
    sigma = h.calibrate_sigma(eps, q, steps)
    opt = h.Adam(P, lr)
    for t in range(steps):
        m = rng.random(n) < q
        xb, yb = xq[m], yq[m]
        _, p = h.forward(P, xb)
        gz, hh, da, sq = b.norms(P, xb, "ce", p, yb, None, clip == "blind")
        g = h.summed_grad(xb, gz, hh, da, 1.0 / (np.sqrt(sq) + gamma))
        for k in g:
            g[k] = (g[k] + rng.standard_normal(g[k].shape) * sigma) / h.B
        opt.step(P, g, 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / steps)))
    return lambda x: np.tanh(x @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"]


def brier_rows(p1, y):
    return 2 * (p1 - y) ** 2  # binary: sum over both classes, in [0, 2]


def exp_mech(util, eps, rng, sens=2.0):
    w = eps * (util - util.max()) / (2 * sens)
    pr = np.exp(w)
    return rng.choice(len(util), p=pr / pr.sum())


class Cal:
    """A fitted calibrator: callable on margins, with its parameters and whether it sits on a range edge."""
    def __init__(self, f, params, edge):
        self.f, self.params, self.edge = f, params, edge

    def __call__(self, dd):
        return self.f(dd)


def fit_ts(d, y, eps, rng):
    util = np.array([-brier_rows(1 / (1 + np.exp(-d / t)), y).sum() for t in TGRID])
    j = int(util.argmax() if eps is None else exp_mech(util, eps, rng))
    t = float(TGRID[j])
    return Cal(lambda dd: dd / t, {"T": t}, j in (0, len(TGRID) - 1))


def fit_platt(d, y, eps, rng):
    cand = [(ia, ib) for ia in range(len(AGRID)) for ib in range(len(BGRID))]
    util = np.array([-brier_rows(1 / (1 + np.exp(-(AGRID[ia] * d + BGRID[ib]))), y).sum() for ia, ib in cand])
    j = int(util.argmax() if eps is None else exp_mech(util, eps, rng))
    ia, ib = cand[j]
    a, c = float(AGRID[ia]), float(BGRID[ib])
    return Cal(lambda dd: a * dd + c, {"a": a, "b": c},
               ia in (0, len(AGRID) - 1) or ib in (0, len(BGRID) - 1))


def score(f, cal, x, y):
    d = f(x)
    d = d[:, 1] - d[:, 0]
    p1 = 1 / (1 + np.exp(-cal(d)))
    out = dict(brier=float(brier_rows(p1, y).mean()), acc=float(((p1 > 0.5) == y).mean()),
               ece=float(h.metrics(np.stack([1 - p1, p1], 1), y)["ece"]))
    if isinstance(cal, Cal):
        out.update(cal_params=cal.params, cal_at_edge=bool(cal.edge))
    return out


ident = lambda dd: dd  # noqa: E731


def margin(f, x):
    d = f(x)
    return d[:, 1] - d[:, 0]


TUNE_SEEDS = tuple(int(s) for s in os.environ.get("TUNE_SEEDS", "0").split(","))  # pre-registered: 0 only


FROZEN = os.environ.get("FROZEN_LR_FROM", "")  # reuse another run's tuning (isolates a calibrator change)


def tune(spec, n_total, eps):
    if FROZEN:
        for r in json.load(open(FROZEN)):
            if (r["spec"], r["N"], r["eps"]) == (spec, n_total, eps):
                return r["tuning"]
        raise KeyError((spec, n_total, eps))
    out = {}
    for clip in ("std", "blind"):
        res = {lr: 0.0 for lr in LRS}
        for ts in TUNE_SEEDS:
            pub, (xq, yq), (xv, yv), _ = make_data(-1 - ts if ts else 0, n_total, spec)
            for lr in LRS:
                f = train(clip, lr, eps, pub, xq, yq, ts)
                cal = fit_ts(margin(f, xv), yv, None, None) if clip == "std" else ident
                res[lr] += score(f, cal, xv, yv)["brier"] / len(TUNE_SEEDS)
        out[clip] = (min(res, key=res.get), res)
    return out


def paired(a, c):
    d = np.array(a) - np.array(c)
    half = T975_DF4 * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def main():
    t0, report = time.time(), []
    for spec in ("well", "mis"):
        for n_total in tuple(int(n) for n in os.environ.get("NS", "36000,8000").split(",")):
            for eps in tuple(int(e) for e in os.environ.get("EPS", "1,4,8").split(",")):
                tuned = tune(spec, n_total, eps)
                lr_s, lr_b = tuned["std"][0], tuned["blind"][0]
                rows = {}
                for seed in SEEDS:
                    pub, (xq, yq), (xv, yv), (xt, yt) = make_data(seed, n_total, spec)
                    crng = np.random.default_rng(777 + seed)
                    r = {}
                    fb = train("blind", lr_b, eps, pub, xq, yq, seed)
                    r["blind"] = score(fb, ident, xt, yt)
                    fs = train("std", lr_s, eps, pub, xq, yq, seed)
                    r["std raw"] = score(fs, ident, xt, yt)
                    r["std+publicTS"] = score(fs, fit_ts(margin(fs, xv), yv, None, None), xt, yt)
                    for m in MS:
                        fm = train("std", lr_s, eps, pub, xq[:-m], yq[:-m], seed)
                        dc, yc = margin(fm, xq[-m:]), yq[-m:]
                        r[f"std+DPTS m={m}"] = score(fm, fit_ts(dc, yc, eps, crng), xt, yt)
                        r[f"std+DPPlatt m={m}"] = score(fm, fit_platt(dc, yc, eps, crng), xt, yt)
                    for k, v in r.items():
                        rows.setdefault(k, []).append(v)
                mean = {k: {mm: float(np.mean([x[mm] for x in v])) for mm in ("brier", "acc", "ece")}
                        for k, v in rows.items()}
                private = [k for k in rows if k.startswith("std+DP") or k == "std raw"]
                best = min(private, key=lambda k: mean[k]["brier"])
                contrasts = {}
                for k in rows:
                    if k == "blind":
                        continue
                    contrasts[k] = dict(brier=paired([x["brier"] for x in rows["blind"]], [x["brier"] for x in rows[k]]),
                                        acc=paired([x["acc"] for x in rows["blind"]], [x["acc"] for x in rows[k]]))
                cb = contrasts[best]
                s1 = cb["brier"][0] <= -0.003 and cb["brier"][2] < 0 and cb["acc"][0] >= -0.005
                cp = contrasts["std+publicTS"]
                s2 = cp["brier"][0] <= -0.003 and cp["brier"][2] < 0 and cp["acc"][0] >= -0.005
                weak = [m for m in MS if any(contrasts[f"std+DP{c} m={m}"]["brier"][0] <= -0.003 and
                                             contrasts[f"std+DP{c} m={m}"]["brier"][2] < 0 for c in ("TS",))]
                edges = {k: sum(x.get("cal_at_edge", False) for x in v) for k, v in rows.items() if "cal_params" in v[0]}
                report.append(dict(spec=spec, N=n_total, eps=eps, lr_std=lr_s, lr_blind=lr_b, tuning=tuned,
                                   per_seed=rows, cal_at_edge_counts=edges,
                                   ranges=dict(T=[float(TGRID[0]), float(TGRID[-1])], a=[float(AGRID[0]), float(AGRID[-1])],
                                               b=[float(BGRID[0]), float(BGRID[-1])]),
                                   mean=mean, contrasts=contrasts, best_private=best, S1=s1, S2=s2,
                                   S1_weak_ms_vs_DPTS=weak))
                print(f"\n== {spec} N={n_total} eps={eps} (lr std {lr_s:g}, blind {lr_b:g}) [{time.time()-t0:.0f}s]")
                for k in rows:
                    c = contrasts.get(k)
                    extra = (f"  blind-this: Brier {c['brier'][0]:+.4f} [{c['brier'][1]:+.4f},{c['brier'][2]:+.4f}]"
                             f" acc {c['acc'][0]*100:+.2f}pts") if c else ""
                    print(f"  {k:20s} Brier {mean[k]['brier']:.4f} acc {mean[k]['acc']:.4f} ece {mean[k]['ece']:.4f}{extra}")
                print(f"  best private baseline: {best}; S1 {'PASS' if s1 else 'fail'}; S2 {'PASS' if s2 else 'fail'}; "
                      f"blind beats DP-TS at m in {weak}", flush=True)
                hit = {k: n for k, n in edges.items() if n}
                if hit:
                    print(f"  WARNING calibrator on a range edge (seeds out of {len(SEEDS)}): {hit}", flush=True)
    tag = ("" if TUNE_SEEDS == (0,) else "_tune" + "-".join(map(str, TUNE_SEEDS))) + \
          ("" if "MS" not in os.environ else "_m" + "-".join(map(str, MS))) + \
          os.environ.get("OUT_TAG", "")
    json.dump(report, open(f"calquota_results{tag}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
