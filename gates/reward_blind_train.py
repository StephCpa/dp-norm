"""Reward-blind vs standard per-prompt clipping in DP-GRPO: linearized training toy.

Predictions T1-T3 are in PREREG_REWARD_BLIND.md (written before this ran).

Prompt x: p_x(theta) = sigmoid(b_x + <theta, psi_x>), psi_x = kappa mu + U w_x + beta e_x (mu: a skill
every prompt shares; U: 16-dim conflict subspace). Response score s_i = (R_i - p_x) psi_x + xi_i,
xi_i ~ N(0, sigma_xi^2/D I), so E[group gradient] = (K-1)/K grad p_x (noise-dominated if
sigma_xi >> ||psi||). Poisson-sampled prompts, per-prompt group gradient clipped once (the prompt is
the privacy unit), Gaussian noise sigma_DP * C, plain SGD ascent on sum_x p_x.
"""
import argparse, itertools, json, math, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpdecide.privacy import calibrate_sigma  # noqa: E402

torch.set_default_dtype(torch.float32)


def make_task(N, D, r, alpha, beta, seed, kappa=1.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, r + 1, generator=g))
    mu, U = Q[:, 0], Q[:, 1:]  # shared skill direction mu (helps every prompt), conflict subspace U
    w = torch.randn(N, r, generator=g) * alpha / math.sqrt(r)
    e = torch.randn(N, D, generator=g) * beta / math.sqrt(D)
    psi = kappa * mu + w @ U.T + e
    u = torch.rand(N, generator=g)
    kind = torch.multinomial(torch.tensor([0.35, 0.30, 0.35]), N, replacement=True, generator=g)
    lo = torch.tensor([0.01, 0.2, 0.9])[kind]
    hi = torch.tensor([0.1, 0.8, 0.99])[kind]
    p0 = lo + (hi - lo) * u
    b = torch.log(p0 / (1 - p0))
    return psi, b, kind


def adv(R, kind):
    A = R - R.mean(-1, keepdim=True)
    if kind == "std":
        sd = R.std(-1, unbiased=False, keepdim=True)
        A = torch.where(sd > 0, A / sd.clamp_min(1e-12), torch.zeros_like(A))
    return A


def cfg_quad(K, kind):
    cfg = torch.tensor(list(itertools.product([0.0, 1.0], repeat=K)))
    cfg = cfg[(cfg.sum(1) > 0) & (cfg.sum(1) < K)]
    Ac = adv(cfg, kind)
    return (Ac.unsqueeze(2) * Ac.unsqueeze(1)).reshape(len(Ac), K * K)


METHODS = {
    # name: (advantage, clip, private)
    "drgrpo_none_nodp": ("center", "none", False),
    "grpo_none_nodp": ("std", "none", False),
    "std_auto_nodp": ("center", "std_auto", False),
    "blind_auto_nodp": ("center", "blind_auto", False),
    "std_auto": ("center", "std_auto", True),
    "blind_auto": ("center", "blind_auto", True),
    "tri_auto": ("center", "tri_auto", True),
    "grpo_std_auto": ("std", "std_auto", True),
    "std_thr": ("center", "std_thr", True),
    "blind_thr": ("center", "blind_thr", True),
}


def train(method, task, args, lr, seed, sigma_dp, C_thr):
    advk, clip, private = METHODS[method]
    psi, b, kind = task
    N, D = psi.shape
    K = args.K
    g = torch.Generator().manual_seed(seed)
    theta = torch.zeros(D)
    AA = cfg_quad(K, advk)
    maxabs = (K - 1) / K if advk == "center" else math.sqrt(K - 1)
    q = args.batch / N
    C = C_thr if clip.endswith("_thr") else 1.0
    gamma = args.gamma
    for _ in range(args.steps):
        idx = torch.nonzero(torch.rand(N, generator=g) < q).squeeze(1)
        if len(idx):
            ps = psi[idx]
            p = torch.sigmoid(b[idx] + ps @ theta)
            R = (torch.rand(len(idx), K, generator=g) < p[:, None]).float()
            xi = torch.randn(len(idx), K, D, generator=g) * (args.sigma_xi / math.sqrt(D))
            s = (R - p[:, None]).unsqueeze(2) * ps.unsqueeze(1) + xi  # (n,K,D)
            A = adv(R, advk)
            gx = (A.unsqueeze(2) * s).sum(1) / K  # (n,D)
            nrm = gx.norm(dim=1)
            if clip == "none":
                f = torch.ones_like(nrm)
            elif clip == "std_auto":
                f = 1.0 / (nrm + gamma)
            elif clip == "std_thr":
                f = torch.clamp(C / nrm.clamp_min(1e-12), max=1.0)
            else:
                G = s @ s.transpose(1, 2)
                if clip.startswith("blind"):
                    B = (AA @ G.reshape(len(idx), K * K).T).max(0).values.clamp_min(0).sqrt() / K
                else:
                    B = maxabs * torch.diagonal(G, dim1=1, dim2=2).sqrt().sum(1) / K
                f = 1.0 / (B + gamma) if clip.endswith("auto") else torch.clamp(C / B, max=1.0)
            if clip != "none":
                assert float((f * nrm).max()) <= C * (1 + 1e-4)
            tot = (f.unsqueeze(1) * gx).sum(0)
        else:
            tot = torch.zeros(D)
        if private:
            tot = tot + torch.randn(D, generator=g) * sigma_dp * C
        theta = theta + lr * tot / args.batch
    p = torch.sigmoid(b + psi @ theta)
    return {"J": float(p.mean()), "J_hard": float(p[kind == 0].mean()),
            "J_mid": float(p[kind == 1].mean()), "J_easy": float(p[kind == 2].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--sigma-xi", type=float, default=10.0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--batch", type=float, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--gamma", type=float, default=1e-3)
    ap.add_argument("--lrs", default="0.1,0.3,1,3,10,30,100,300,1000")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default="reward_blind_train.json")
    args = ap.parse_args()
    q = args.batch / args.N
    sigma_dp = calibrate_sigma(args.eps, args.delta, q, args.steps)
    print(f"N={args.N} D={args.D} K={args.K} q={q:.4f} steps={args.steps} eps={args.eps} -> sigma_DP={sigma_dp:.3f}")
    lrs = [float(x) for x in args.lrs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    res = {"args": vars(args), "sigma_dp": sigma_dp, "runs": []}
    for seed in seeds:
        task = make_task(args.N, args.D, args.r, args.alpha, args.beta, seed, args.kappa)
        p0 = torch.sigmoid(task[1])
        # threshold = median standard (centered) norm at init, from a public-like probe
        gp = torch.Generator().manual_seed(10_000 + seed)
        idx = torch.randperm(args.N, generator=gp)[:512]
        R = (torch.rand(512, args.K, generator=gp) < p0[idx, None]).float()
        xi = torch.randn(512, args.K, args.D, generator=gp) * (args.sigma_xi / math.sqrt(args.D))
        s = (R - p0[idx, None]).unsqueeze(2) * task[0][idx].unsqueeze(1) + xi
        nrm = ((adv(R, "center").unsqueeze(2) * s).sum(1) / args.K).norm(dim=1)
        C_thr = float(nrm[nrm > 0].median())
        print(f"seed {seed}: J0={float(p0.mean()):.4f} hard0={float(p0[task[2]==0].mean()):.4f} C_thr={C_thr:.3f}")
        for m in args.methods.split(","):
            for lr in lrs:
                t0 = time.time()
                out = train(m, task, args, lr, 100 * seed + 7, sigma_dp, C_thr)
                out.update(method=m, lr=lr, seed=seed, J0=float(p0.mean()))
                res["runs"].append(out)
                print(f"  {m:18s} lr={lr:<5g} J={out['J']:.4f} hard={out['J_hard']:.4f} "
                      f"mid={out['J_mid']:.4f} easy={out['J_easy']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
    # best lr per method by seed-mean J
    print("\nBest lr per method (seed-mean):")
    summ = {}
    for m in args.methods.split(","):
        best = None
        for lr in lrs:
            rs = [r for r in res["runs"] if r["method"] == m and r["lr"] == lr]
            J = sum(r["J"] for r in rs) / len(rs)
            if best is None or J > best[1]:
                best = (lr, J, rs)
        lr, J, rs = best
        n = len(rs)
        sd = (sum((r["J"] - J) ** 2 for r in rs) / max(1, n - 1)) ** 0.5
        hard = sum(r["J_hard"] for r in rs) / n
        summ[m] = {"lr": lr, "J": J, "J_sd": sd, "J_hard": hard,
                   "J_mid": sum(r["J_mid"] for r in rs) / n, "J_easy": sum(r["J_easy"] for r in rs) / n,
                   "per_seed": [r["J"] for r in rs]}
        print(f"  {m:18s} lr={lr:<5g} J={J:.4f} (sd {sd:.4f}) hard={hard:.4f} "
              f"mid={summ[m]['J_mid']:.4f} easy={summ[m]['J_easy']:.4f}")
    res["summary"] = summ
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
