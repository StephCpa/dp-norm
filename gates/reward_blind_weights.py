"""Reward-blind clipping for binary-reward GRPO: Monte-Carlo check of the per-prompt weighting.

Predictions W1-W7 are in PREREG_REWARD_BLIND.md (written before this ran).

Score model: s_i = c_i u_hat + xi_perp_i with c_i = (R_i - p) a + z_i, z_i ~ N(0, sigma^2/d), and
xi_perp Gram ~ (sigma^2/d) Wishart_K(d-1) (Bartlett), so the K x K Gram is exact without
materializing d-dim vectors. Clipped signal = factor * (1/K) sum_i A_i c_i (component along u).
"""
import argparse, itertools, json, math
import torch

torch.set_default_dtype(torch.float64)


def wishart_bartlett(T, K, n, gen):
    L = torch.zeros(T, K, K)
    for i in range(K):
        chi2 = torch.distributions.Chi2(torch.tensor(float(n - i))).sample((T,))
        L[:, i, i] = chi2.sqrt()
        if i:
            L[:, i, :i] = torch.randn(T, i, generator=gen)
    return L @ L.transpose(1, 2)


def advantages(R, kind):
    A = R - R.mean(-1, keepdim=True)
    if kind == "std":
        sd = R.std(-1, unbiased=False, keepdim=True)
        A = torch.where(sd > 0, A / sd.clamp_min(1e-12), torch.zeros_like(A))
    return A


def configs(K, kind):
    cfg = torch.tensor(list(itertools.product([0.0, 1.0], repeat=K)))
    cfg = cfg[(cfg.sum(1) > 0) & (cfg.sum(1) < K)]
    return cfg, advantages(cfg, kind)


def draw(T, K, p, a, sigma, d, gen):
    R = (torch.rand(T, K, generator=gen) < p).double()
    c = (R - p) * a + torch.randn(T, K, generator=gen) * sigma / math.sqrt(d)
    W = wishart_bartlett(T, K, d - 1, gen) * sigma**2 / d
    G = c.unsqueeze(2) * c.unsqueeze(1) + W
    return R, c, G


def blind_bound(G, K, kind, chunk=512):
    cfg, Ac = configs(K, kind)
    AA = (Ac.unsqueeze(2) * Ac.unsqueeze(1)).reshape(len(Ac), K * K)
    Bs, ms = [], []
    for t0 in range(0, G.shape[0], chunk):
        Q = AA @ G[t0:t0 + chunk].reshape(-1, K * K).T  # (C, t)
        q, idx = Q.max(0)
        Bs.append(q.clamp_min(0).sqrt() / K)
        ms.append(cfg[idx].sum(1))
    return torch.cat(Bs), torch.cat(ms)


def tri_bound(G, K, kind):
    maxabs = (K - 1) / K if kind == "center" else math.sqrt(K - 1)
    return maxabs * torch.diagonal(G, dim1=1, dim2=2).clamp_min(0).sqrt().sum(1) / K


def binom_moments(K, p):
    ES = ESr = 0.0
    for m in range(K + 1):
        w = math.comb(K, m) * p**m * (1 - p) ** (K - m)
        S = m * (K - m) / K**2
        ES += w * S
        ESr += w * math.sqrt(S)
    return ES, ESr


def run_cell(K, kind, a, sigma, d, ps, T, seed, gamma_rel=1e-3):
    gen = torch.Generator().manual_seed(seed)
    gamma = gamma_rel * sigma
    # threshold = median standard norm over non-degenerate groups at p = 0.5
    R0, c0, G0 = draw(T, K, 0.5, a, sigma, d, gen)
    A0 = advantages(R0, kind)
    n0 = torch.einsum("tk,tkl,tl->t", A0, G0, A0).clamp_min(0).sqrt() / K
    C_thr = float(n0[n0 > 0].median())
    out = {"K": K, "adv": kind, "a": a, "sigma": sigma, "d": d, "C_thr": C_thr, "rows": []}
    for p in ps:
        R, c, G = draw(T, K, p, a, sigma, d, gen)
        A = advantages(R, kind)
        sig = (A * c).sum(1) / K
        nrm = torch.einsum("tk,tkl,tl->t", A, G, A).clamp_min(0).sqrt() / K
        B, mstar = blind_bound(G, K, kind)
        Bt = tri_bound(G, K, kind)
        assert bool((B + 1e-9 >= nrm).all()), "blind bound below realized norm"
        f = {
            "none": torch.ones_like(nrm),
            "std_auto": 1.0 / (nrm + gamma),
            "std_thr": torch.clamp(C_thr / nrm.clamp_min(1e-300), max=1.0),
            "blind_auto": 1.0 / (B + gamma),
            "blind_thr": torch.clamp(C_thr / B, max=1.0),
            "tri_auto": 1.0 / (Bt + gamma),
        }
        nondeg = (R.sum(1) > 0) & (R.sum(1) < K)
        ES, ESr = binom_moments(K, p)
        row = {"p": p, "ES": ES, "ESr": ESr,
               "mean_B": float(B.mean()), "mean_Btri": float(Bt.mean()),
               "frac_argmax_balanced": float((mstar[nondeg] == K // 2).double().mean()) if nondeg.any() else float("nan"),
               "frac_argmax_extreme": float(((mstar[nondeg] == 1) | (mstar[nondeg] == K - 1)).double().mean()) if nondeg.any() else float("nan")}
        for k, v in f.items():
            row["sig_" + k] = float((v * sig).mean())
            row["se_" + k] = float((v * sig).std() / math.sqrt(T))
        out["rows"].append(row)
    base = {k: out["rows"][-1]["sig_" + k] for k in ("none", "std_auto", "std_thr", "blind_auto", "blind_thr", "tri_auto")}
    for row in out["rows"]:
        for k, v in base.items():
            row["shape_" + k] = row["sig_" + k] / v
        row["pred_shape_unclipped"] = row["ES"] / out["rows"][-1]["ES"]
        row["pred_shape_arcsine"] = row["ESr"] / out["rows"][-1]["ESr"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reward_blind_weights.json")
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--T", type=int, default=20000)
    args = ap.parse_args()
    ps = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    cells = []
    for kind in ("center", "std"):
        for a, regime in ((0.05, "noise"), (0.7, "mixed"), (20.0, "signal")):
            cells.append((8, kind, a, regime, args.T))
        cells.append((4, kind, 0.05, "noise", args.T))
        cells.append((16, kind, 0.05, "noise", 2000))
    res = []
    for K, kind, a, regime, T in cells:
        r = run_cell(K, kind, a, 1.0, args.d, ps, T, seed=1000 + K)
        r["regime"] = regime
        res.append(r)
        print(f"\n== K={K} adv={kind} regime={regime} (a={a}, sigma=1, d={args.d}, T={T}) C_thr={r['C_thr']:.4f}")
        print(" p     | shape: none  std_auto std_thr blind_auto blind_thr tri_auto | pred p(1-p) arcsine | "
              "blind/std abs (pred 2ES/ESr) | Btri/B | argmax bal / ext | mean B")
        for w in r["rows"]:
            ratio = w["sig_blind_auto"] / w["sig_std_auto"] if w["sig_std_auto"] else float("nan")
            pred_ratio = 2 * w["ES"] / w["ESr"] * (1.0 if K % 2 == 0 else 1.0)
            print(f" {w['p']:.2f}  | {w['shape_none']:.3f} {w['shape_std_auto']:.3f} {w['shape_std_thr']:.3f} "
                  f"{w['shape_blind_auto']:.3f} {w['shape_blind_thr']:.3f} {w['shape_tri_auto']:.3f} | "
                  f"{w['pred_shape_unclipped']:.3f} {w['pred_shape_arcsine']:.3f} | "
                  f"{ratio:.3f} ({pred_ratio:.3f}) | {w['mean_Btri']/w['mean_B']:.2f} | "
                  f"{w['frac_argmax_balanced']:.2f} / {w['frac_argmax_extreme']:.2f} | {w['mean_B']:.4f}")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
