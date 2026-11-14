"""Calibration quota, confirmation run with frozen wide calibrator ranges and fresh seeds.

Pre-registration: gates/PREREG_CALQUOTA_CONFIRM.md, pushed in its own commit before this script is run on
the confirmation seeds. Reuses the toy, training and calibrators of calquota_screen.py.

Differences from the screen (reaudit of df9bb65):
- calibrator ranges wide enough that the public fit does not sit on an edge (checked and reported);
- every complete pipeline (training size N - m, calibrator, lr) tuned on separate dev seeds; the deployable
  std pipeline (m, calibrator, lr) is chosen on dev data, not by test means;
- quotas up to half of N;
- a decomposition of each quota's cost on the same trained model;
- full per-seed records (metrics, calibrator parameters, edge hits, steps, sigma).
"""
import json
import os
import time

import numpy as np

import calquota_screen as c
import h1_gate as h

np.seterr(over="ignore")  # sigmoid of large |margin / T| saturates to 0 or 1, which is the intended value

# frozen ranges (pre-registered)
c.TGRID = np.exp(np.linspace(np.log(1 / 32), np.log(32), 61))
c.AGRID = np.exp(np.linspace(np.log(1 / 32), np.log(4), 31))
c.BGRID = np.linspace(-3, 3, 25)

LRS = (1e-3, 3e-3, 1e-2, 3e-2)
FRACS = (1 / 128, 1 / 32, 1 / 8, 1 / 4, 1 / 2)
DEV = tuple(int(s) for s in os.environ.get("DEV_SEEDS", "201,202,203").split(","))
EVAL = tuple(int(s) for s in os.environ.get("EVAL_SEEDS", ",".join(str(s) for s in range(101, 111))).split(","))
NS = tuple(int(n) for n in os.environ.get("NS", "36000,8000").split(","))
EPSS = tuple(int(e) for e in os.environ.get("EPS", "4,1,8").split(","))
SPECS = tuple(os.environ.get("SPECS", "well,mis").split(","))
T975 = {9: 2.262, 4: 2.776, 2: 4.303, 1: 12.706}
FIT = {"TS": c.fit_ts, "Platt": c.fit_platt}


def quotas(n):
    return tuple(max(16, int(round(f * n))) for f in FRACS)


def pipeline(spec, n, eps, seed, m, lr, cal_name, crng, keep_model=False):
    """std trained on N - m, DP calibrator fitted on the held-out m rows. Returns (val Brier, model, cal)."""
    pub, (xq, yq), (xv, yv), _ = c.make_data(seed, n, spec)
    f = c.train("std", lr, eps, pub, xq[:-m], yq[:-m], seed)
    cal = FIT[cal_name](c.margin(f, xq[-m:]), yq[-m:], eps, crng)
    return c.score(f, cal, xv, yv)["brier"], (f if keep_model else None), cal


def tune_cell(spec, n, eps):
    """Everything chosen on dev seeds and public validation data only."""
    res = {"std_pipelines": {}, "blind": {}, "std_publicTS": {}}
    for lr in LRS:
        vb, vp = [], []
        for s in DEV:
            pub, (xq, yq), (xv, yv), _ = c.make_data(s, n, spec)
            fb = c.train("blind", lr, eps, pub, xq, yq, s)
            vb.append(c.score(fb, c.ident, xv, yv)["brier"])
            fs = c.train("std", lr, eps, pub, xq, yq, s)
            vp.append(c.score(fs, c.fit_ts(c.margin(fs, xv), yv, None, None), xv, yv)["brier"])
        res["blind"][lr], res["std_publicTS"][lr] = float(np.mean(vb)), float(np.mean(vp))
        for m in quotas(n):
            for cal_name in FIT:
                v = [pipeline(spec, n, eps, s, m, lr, cal_name, np.random.default_rng(5000 + s))[0] for s in DEV]
                res["std_pipelines"][f"m={m} {cal_name} lr={lr:g}"] = dict(m=m, cal=cal_name, lr=lr, val=float(np.mean(v)))
    pick = lambda d: min(d, key=d.get)  # noqa: E731
    deploy = min(res["std_pipelines"].values(), key=lambda r: r["val"])
    per_m = {}
    for r in res["std_pipelines"].values():
        if r["m"] not in per_m or r["val"] < per_m[r["m"]]["val"]:
            per_m[r["m"]] = r
    return dict(blind_lr=pick(res["blind"]), publicTS_lr=pick(res["std_publicTS"]), deploy=deploy,
                per_m=per_m, dev=res)


def eval_seed(spec, n, eps, seed, tun):
    pub, (xq, yq), (xv, yv), (xt, yt) = c.make_data(seed, n, spec)
    crng = np.random.default_rng(9000 + seed)
    q_full = h.B / n
    out = {"seed": seed}
    fb = c.train("blind", tun["blind_lr"], eps, pub, xq, yq, seed)
    out["blind"] = c.score(fb, c.ident, xt, yt)
    fp = c.train("std", tun["publicTS_lr"], eps, pub, xq, yq, seed)
    oracle_full = c.fit_ts(c.margin(fp, xv), yv, None, None)
    out["std(N)+publicTS"] = c.score(fp, oracle_full, xt, yt)
    out["std(N) raw"] = c.score(fp, c.ident, xt, yt)
    curve = {}
    for m, r in sorted(tun["per_m"].items()):
        f = c.train("std", r["lr"], eps, pub, xq[:-m], yq[:-m], seed)
        dc, yc = c.margin(f, xq[-m:]), yq[-m:]
        fit = FIT[r["cal"]]
        curve[m] = dict(
            cal=r["cal"], lr=r["lr"],
            dp=c.score(f, fit(dc, yc, eps, crng), xt, yt),                 # what a private release gets
            nonpriv_m=c.score(f, fit(dc, yc, None, None), xt, yt),         # same m rows, no privacy
            public=c.score(f, c.fit_ts(c.margin(f, xv), yv, None, None), xt, yt),  # large public set
            steps=int(h.EPOCHS / (h.B / (n - m))), q=h.B / (n - m))
        if (m, r["cal"], r["lr"]) == (tun["deploy"]["m"], tun["deploy"]["cal"], tun["deploy"]["lr"]):
            out["deployable std"] = dict(curve[m]["dp"], m=m, cal=r["cal"], lr=r["lr"])
    if "deployable std" not in out:  # the overall dev winner is always one of the per-m winners
        raise RuntimeError("deployable pipeline missing from the per-m curve")
    out["curve"] = curve
    out["steps_full"], out["q_full"] = int(h.EPOCHS / q_full), q_full
    return out


def ci(a, b):
    d = np.array(a) - np.array(b)
    t = T975.get(len(d) - 1, 2.0)
    half = t * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - half), float(d.mean() + half)]


def summarize(cell):
    S = cell["seeds"]
    B = [s["blind"]["brier"] for s in S]
    D = [s["deployable std"]["brier"] for s in S]
    res = dict(blind_minus_deployable=ci(B, D),
               acc_blind_minus_deployable=ci([s["blind"]["acc"] for s in S], [s["deployable std"]["acc"] for s in S]),
               blind_minus_publicTS=ci(B, [s["std(N)+publicTS"]["brier"] for s in S]))
    ms = sorted(S[0]["curve"])
    res["oracle_best_m"] = min(ms, key=lambda m: np.mean([s["curve"][m]["dp"]["brier"] for s in S]))
    res["blind_minus_oracle_best"] = ci(B, [s["curve"][res["oracle_best_m"]]["dp"]["brier"] for s in S])
    res["decomposition"] = {}
    full_pub = [s["std(N)+publicTS"]["brier"] for s in S]
    for m in ms:
        dp = [s["curve"][m]["dp"]["brier"] for s in S]
        npm = [s["curve"][m]["nonpriv_m"]["brier"] for s in S]
        pubm = [s["curve"][m]["public"]["brier"] for s in S]
        res["decomposition"][m] = dict(privacy_of_calibration=ci(dp, npm), finite_calibration_set=ci(npm, pubm),
                                       fewer_training_rows=ci(pubm, full_pub), blind_minus_this=ci(B, dp))
    edges = {"publicTS_full": sum(s["std(N)+publicTS"].get("cal_at_edge", False) for s in S)}
    for m in ms:
        edges[f"public m={m}"] = sum(s["curve"][m]["public"].get("cal_at_edge", False) for s in S)
        edges[f"nonpriv m={m}"] = sum(s["curve"][m]["nonpriv_m"].get("cal_at_edge", False) for s in S)
    res["edge_hits"] = edges
    res["range_ok"] = edges["publicTS_full"] == 0 and all(v == 0 for k, v in edges.items() if k.startswith("public"))
    return res


def main():
    t0, cells = time.time(), []
    for eps in EPSS:
        for n in NS:
            for spec in SPECS:
                tun = tune_cell(spec, n, eps)
                seeds = [eval_seed(spec, n, eps, s, tun) for s in EVAL]
                cell = dict(spec=spec, N=n, eps=eps, quotas=quotas(n), tuning=tun, seeds=seeds,
                            ranges=dict(T=[float(c.TGRID[0]), float(c.TGRID[-1])], a=[float(c.AGRID[0]), float(c.AGRID[-1])],
                                        b=[float(c.BGRID[0]), float(c.BGRID[-1])]))
                cell["summary"] = sm = summarize(cell)
                cells.append(cell)
                d = sm["blind_minus_deployable"]
                print(f"== {spec} N={n} eps={eps}: deployable std {tun['deploy']} | blind lr {tun['blind_lr']:g}\n"
                      f"   blind - deployable {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}]  acc "
                      f"{sm['acc_blind_minus_deployable'][0]*100:+.2f} pts | vs public TS {sm['blind_minus_publicTS'][0]:+.5f}"
                      f" | vs oracle-best m={sm['oracle_best_m']} {sm['blind_minus_oracle_best'][0]:+.5f}"
                      f" | range ok {sm['range_ok']} {sm['edge_hits'] if not sm['range_ok'] else ''} [{time.time()-t0:.0f}s]",
                      flush=True)
                json.dump(cells, open(os.environ.get("OUT", "calquota_confirm_results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
