"""Label-blind per-unit clipping for DP-SGD.

Standard DP-SGD scales a unit's gradient g(x, y) by min(1, C/||g(x, y)||). The factor depends on the
label, so clipping reweights labels within an input (surprising labels have large gradients and are
down-weighted). That turns a proper scoring objective into an improper one (Menon et al. 2020: clipping
= Huberisation) and is a mechanism behind DP-SGD overconfidence (Zhang et al. 2022; Bu et al. 2023).

Label-blind clipping scales by a factor computed from the inputs, the model and (for bandit objectives)
the model's own sampled actions, but not the labels:
    B(unit) = sum over rows r of  max over every label y' the row could carry  ||g_r(x_r, y')||
    factor  = min(1, C / B)            (threshold)      or      C / (B + gamma)      (automatic, AUTO-S)
The realized unit gradient has norm <= sum_r ||g_r(x_r, y_r)|| <= B, so ||factor * g|| <= C: the
sensitivity is C, exactly as in standard DP-SGD, and the accountant is unchanged.

Why it preserves calibration: at a predictor with p(.|x) = P(.|x), E_{y|x}[dLoss/dlogits] = 0 for any
proper scoring loss (CE, Brier), and for RLCD E_{y|x,a}[c_j - pi(a_j)] = 0 for every sampled action, so
any weighting measurable in (x, a) leaves that point stationary. A label-dependent weighting does not.

Computation: every per-row gradient is J_r^T v for a logit-space vector v whose entries sum to zero
(CE: p - e_y; Brier: (diag p - p p^T) 2(p - e_y); RLCD: mean of adv_j (p - e_{a_j})). So it lies in the
span of the k - 1 contrast gradients u_j = d(z_j - z_0)/d theta, j >= 1, with coefficients v_j. With their
Gram matrix G, ||J^T v||^2 = w^T G w (w = v[1:]) for every label at once, and the realized gradient is
sum_j v_j u_j. Cost: k - 1 backward passes per row, i.e. ONE for binary decisions, the same as standard
clipping (review, 14 Nov: the earlier version used k). Memory: k - 1 trainable-size vectors (use LoRA).

Units with several rows. The unit factor depends on every row's inputs and actions, so the fixed-point
argument needs E[g_r | X_unit, A_unit] = 0 for each row: the label of row r must carry no information
from the other rows' inputs beyond x_r (true for single-row units; a counterexample otherwise is in
gates/group_condition_gate.py). row_budget = M gives each row its own factor from its own bound with
budget C/M, which needs only the single-row condition, at the cost of unused budget in units with
fewer than M rows. Sensitivity stays <= C either way.
"""
from __future__ import annotations

import torch

from dpdecide.objectives import loo_pg  # noqa: F401  (documents the RLCD form the vectors below match)


def _logit_vector(objective: str, p: torch.Tensor, y: int, actions: torch.Tensor | None) -> torch.Tensor:
    """dLoss/dlogits restricted to the k valid logits, for label y. p: (k,) float64."""
    k = p.numel()
    e = torch.zeros(k, dtype=p.dtype, device=p.device)
    e[y] = 1.0
    if objective == "ce":
        return p - e
    if objective == "brier":
        jac = torch.diag(p) - torch.outer(p, p)
        return jac @ (2.0 * (p - e))
    if objective in ("rlcd", "rlvr"):
        a = actions
        g = a.numel()
        c = (a == y).to(p.dtype)
        r = c - p[a] if objective == "rlcd" else c
        adv = r - (r.sum() - r) / (g - 1) if g > 1 else r
        v = torch.zeros(k, dtype=p.dtype, device=p.device)
        for j in range(g):
            ea = torch.zeros(k, dtype=p.dtype, device=p.device)
            ea[a[j]] = 1.0
            v -= adv[j] * (ea - p) / g
        return v
    raise ValueError(f"label-blind clipping is not defined for objective {objective}")


def unit_gradient(model, rows, labels: list[int], objective: str, group: int, generator: torch.Generator | None,
                  clip: float, auto_gamma: float = 0.0, max_len: int = 512, check: bool = True, row_budget: int = 0):
    """Label-blind clipped gradient of one privacy unit.
    row_budget = 0: one factor for the unit, from B = sum_r max_y ||g_r||.
    row_budget = M > 0: each row r gets its own factor from max_y ||g_r|| with budget C/M (needs len(rows) <= M).
    Returns (grads aligned with model.trainable_parameters(), stats). The caller adds the grads to the
    noisy sum as is (they already have norm <= clip)."""
    if row_budget:
        assert len(rows) <= row_budget, "row_budget must be >= the per-unit row cap"
    params = model.trainable_parameters()
    logits = model.logits(rows, max_len)
    probs32 = torch.log_softmax(logits.detach(), -1).exp()  # the same expression objectives.row_losses samples from
    probs = probs32.double()
    actions = None
    if objective in ("rlcd", "rlvr"):
        actions = torch.multinomial(probs32, group, replacement=True, generator=generator)
    stats = {"mean_conf": probs.max(-1).values.mean().item()}
    rewards = []
    per_row = []  # (bound_r, realized_sq_r, contrast grads u, coefficients w_true)
    for r, q in enumerate(rows):
        k = q.k
        u = []
        for j in range(1, k):
            gr = torch.autograd.grad(logits[r, j] - logits[r, 0], params, retain_graph=True, allow_unused=True)
            u.append([torch.zeros_like(p, dtype=torch.float32) if t is None else t.float() for t in gr])
        gram = torch.zeros(k - 1, k - 1, dtype=torch.float64)
        for i in range(k - 1):
            for j in range(i, k - 1):
                sdot = sum(float((a * b).sum()) for a, b in zip(u[i], u[j]))
                gram[i, j] = gram[j, i] = sdot
        p = probs[r, :k].cpu()
        p = p / p.sum()
        act = actions[r].cpu() if actions is not None else None
        worst = 0.0
        for y in range(k):
            v = _logit_vector(objective, p, y, act)
            assert abs(float(v.sum())) < 1e-9, "logit vector must sum to zero for the contrast basis"
            worst = max(worst, float(v[1:] @ gram @ v[1:]))
        v_true = _logit_vector(objective, p, labels[r], act)
        per_row.append((worst ** 0.5, max(0.0, float(v_true[1:] @ gram @ v_true[1:])), u, v_true[1:]))
        if act is not None:
            cc = (act == labels[r]).double()
            rewards.append(float((cc - p[act]).mean()) if objective == "rlcd" else float(cc.mean()))
            stats.setdefault("sampled_acc_vs_train_label", 0.0)
            stats["sampled_acc_vs_train_label"] += float(cc.mean()) / len(rows)
    if rewards:
        stats["reward"] = sum(rewards) / len(rewards)

    def factor_for(bound: float, budget: float, gamma: float) -> float:
        if auto_gamma > 0:
            return budget / (bound + gamma)
        return min(1.0, budget / bound) if bound > 0 else 1.0

    bound = sum(b for b, _, _, _ in per_row)
    if row_budget:
        factors = [factor_for(b, clip / row_budget, auto_gamma / row_budget) for b, _, _, _ in per_row]
    else:
        factors = [factor_for(bound, clip, auto_gamma)] * len(per_row)
    total = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for (_, _, u, w), f in zip(per_row, factors):
        for j, uj in enumerate(u):
            c = float(w[j]) * f
            if c != 0.0:
                for t, x in zip(total, uj):
                    t.add_(x, alpha=c)
    norm = sum(float(t.pow(2).sum()) for t in total) ** 0.5
    if check and norm > clip * (1 + 1e-4) + 1e-8:
        # cannot happen in exact arithmetic (norm <= sum_r f_r bound_r <= clip); guard against float error
        for t in total:
            t.mul_(clip / norm)
        stats["guard_rescaled"] = 1.0
    stats.update(blind_bound=bound, blind_factor=sum(factors) / len(factors), clipped_norm=min(norm, clip),
                 unclipped_norm_upper=sum(rs for _, rs, _, _ in per_row) ** 0.5)
    return total, stats
