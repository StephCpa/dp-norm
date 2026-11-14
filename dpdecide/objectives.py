"""Per-row losses. Every function returns a (B,) tensor so a privacy unit's loss is the sum over
its rows and its gradient can be clipped as one vector.

Label-DP variants take the per-row randomized-response parameters (p_keep, q): P(y~ = y) = p_keep,
P(y~ = j) = q for each j != y. Because each objective is linear (rlcd reward, Brier target) or has a
closed-form noisy likelihood (CE), the corrections below are exact, not approximations:
  rlcd  : c~ = (1{y~ = a} - q) / (p_keep - q) is unbiased for c, and the reward is linear in c
  brier : e~ = (onehot(y~) - q) / (p_keep - q) is unbiased for onehot(y); the Brier gradient is linear in the target
  ce_fc : NLL of the noisy label under the noise channel, -log(q + (p_keep - q) * pi_y~)  (forward correction)
"""
from __future__ import annotations

import torch

OBJECTIVES = ("ce", "brier", "rlcd", "rlvr")


def loo_pg(logp: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with a leave-one-out baseline over the G actions of each row. (B,K),(B,G),(B,G) -> (B,)."""
    g = rewards.shape[1]
    base = (rewards.sum(1, keepdim=True) - rewards) / (g - 1) if g > 1 else torch.zeros_like(rewards)
    adv = (rewards - base).detach()
    return -(adv * torch.gather(logp, 1, actions)).mean(1)


def row_losses(objective: str, logits: torch.Tensor, labels: torch.Tensor, group: int,
               generator: torch.Generator | None = None, rr: tuple[torch.Tensor, torch.Tensor] | None = None,
               raw_label_dp: bool = False) -> tuple[torch.Tensor, dict]:
    """logits (B,26) fp32 masked; labels (B,) long (true labels, or RR labels under label DP).
    rr = (p_keep, q) per row when the labels are randomized-response outputs and the objective should
    be corrected; raw_label_dp=True uses RR labels as if they were clean (the negative control)."""
    logp = torch.log_softmax(logits, -1)
    probs = logp.exp()
    stats: dict = {}
    corrected = rr is not None and not raw_label_dp
    if objective == "ce":
        if corrected:
            pk, q = rr
            p_lab = probs.gather(1, labels[:, None]).squeeze(1)
            loss = -torch.log(q + (pk - q) * p_lab)
        else:
            loss = -logp.gather(1, labels[:, None]).squeeze(1)
    elif objective == "brier":
        onehot = torch.zeros_like(probs).scatter_(1, labels[:, None], 1.0)
        if corrected:
            pk, q = rr
            onehot = (onehot - q[:, None]) / (pk - q)[:, None]
        # only the declared options carry mass; masked letters have probs == 0 and target 0
        valid = (logits > -1e3).float()
        loss = (((probs - onehot) * valid) ** 2).sum(1)
    elif objective in ("rlcd", "rlvr"):
        actions = torch.multinomial(probs.detach(), group, replacement=True, generator=generator)
        c = (actions == labels[:, None]).float()
        stats["sampled_acc_vs_train_label"] = c.mean().item()
        if corrected:
            pk, q = rr
            c = (c - q[:, None]) / (pk - q)[:, None]
        p_a = probs.detach().gather(1, actions)
        rewards = c - p_a if objective == "rlcd" else c
        loss = loo_pg(logp, actions, rewards)
        stats["reward"] = rewards.mean().item()
        stats["p_taken"] = p_a.mean().item()
    else:
        raise ValueError(f"unknown objective {objective}")
    stats["mean_conf"] = probs.max(-1).values.mean().item()
    return loss, stats
