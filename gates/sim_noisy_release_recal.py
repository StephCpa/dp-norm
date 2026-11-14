"""Does DP noise on a released probability cost reliability or resolution?

Binary (Noul-style) decision. A calibrated model outputs p; a private-prediction layer releases
p~ = clip(p + N(0, sigma^2), 0, 1) (e.g. the mean of M disjoint-shard teachers plus Gaussian noise).
Because the mechanism is known, a recalibration map g(p~) ~= E[y | p~] can be fitted on PUBLIC data
(post-processing, so the DP guarantee is unchanged). Check: raw p~ is miscalibrated; g(p~) is calibrated
again; what is lost is resolution (sharpness).
Murphy decomposition on B bins: Brier = reliability - resolution + uncertainty.
Sigma mapping (binary, mean of M teachers, sensitivity 1/M, Q queries per record, GDP mu):
   sigma = sqrt(Q) / (M * mu)
"""
import numpy as np

rng = np.random.default_rng(1)
B = 20


def murphy(f, y, bins=B):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(f, edges) - 1, 0, bins - 1)
    ybar = y.mean()
    rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            w = m.mean()
            rel += w * (f[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - ybar) ** 2
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(f[m].mean() - y[m].mean())
    return dict(brier=((f - y) ** 2).mean(), rel=rel, res=res, unc=ybar * (1 - ybar), ece=ece)


def draw(n, sigma):
    p = rng.beta(0.5, 0.5, n)
    y = (rng.random(n) < p).astype(float)
    pt = np.clip(p + sigma * rng.standard_normal(n), 0, 1)
    return p, y, pt


def fit_binning(pt, y, bins=40):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pt, edges) - 1, 0, bins - 1)
    table = np.array([y[idx == b].mean() if (idx == b).any() else (edges[b] + edges[b + 1]) / 2
                      for b in range(bins)])
    return lambda q: table[np.clip(np.digitize(q, edges) - 1, 0, bins - 1)]


print(f"{'sigma':>6} | {'raw: ece':>9} {'rel':>7} {'res':>7} {'brier':>7} | "
      f"{'recal: ece':>10} {'rel':>7} {'res':>7} {'brier':>7}")
p, y, _ = draw(200000, 0.0)
m0 = murphy(p, y)
print(f"{'clean':>6} | {m0['ece']:9.4f} {m0['rel']:7.4f} {m0['res']:7.4f} {m0['brier']:7.4f} |")
for sigma in [0.05, 0.1, 0.2, 0.3]:
    _, ypub, ptpub = draw(50000, sigma)
    g = fit_binning(ptpub, ypub)
    _, y, pt = draw(200000, sigma)
    a, b = murphy(pt, y), murphy(g(pt), y)
    print(f"{sigma:6.2f} | {a['ece']:9.4f} {a['rel']:7.4f} {a['res']:7.4f} {a['brier']:7.4f} | "
          f"{b['ece']:10.4f} {b['rel']:7.4f} {b['res']:7.4f} {b['brier']:7.4f}")

print("\nsigma = sqrt(Q)/(M*mu) for a few settings (mu at delta=1e-6):")
MU = {1: 0.237, 4: 0.838, 8: 1.532}
for M in [100, 500]:
    for Q in [10, 100, 1000]:
        print(f"M={M:>4} Q={Q:>5}: " + "  ".join(f"eps={e}: {np.sqrt(Q)/(M*mu):.3f}" for e, mu in MU.items()))
