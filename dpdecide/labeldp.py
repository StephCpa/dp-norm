"""Label differential privacy by k-ary randomized response, applied ONCE per row before training.

Unit-level label DP: the epsilon of a privacy unit is split equally over its rows (basic composition
of pure-DP randomized responses), so a unit with m rows spends eps/m per label. Everything after the
RR draw (all epochs, all sampled actions, evaluation, checkpoints) is post-processing: the whole run
is eps-label-DP with delta = 0, with no accountant needed.
"""
from __future__ import annotations

import math

import numpy as np


def rr_params(eps: float, k: int) -> tuple[float, float]:
    e = math.exp(eps)
    p_keep = e / (e + k - 1)
    return p_keep, (1 - p_keep) / (k - 1)


def privatize(questions, units: list[list[int]], eps_unit: float, rng: np.random.Generator):
    """Returns (noisy_labels, p_keep, q, eps_row) as per-row lists aligned with `questions`, plus stats.
    Rows that belong to no unit (dropped by the per-unit cap) get label -1 and are never trained on.

    `rng` must be ONE secret stream consumed row by row (privacy.mechanism_seed('secure', ...)). An
    earlier version seeded each row from (public seed, row id): the draws were recomputable from the
    seed, and rows sharing an id reused the same draws, so the basic-composition argument failed."""
    n = len(questions)
    noisy = [-1] * n
    pk = [1.0] * n
    qq = [0.0] * n
    eps_row = [0.0] * n
    flips = 0
    for u in units:
        e = eps_unit / len(u)
        for i in u:
            q = questions[i]
            k = q.k
            p_keep, q_other = rr_params(e, k)
            y = q.answer
            if rng.random() < p_keep:
                yt = y
            else:
                other = int(rng.integers(0, k - 1))
                yt = other + (other >= y)
                flips += 1
            noisy[i], pk[i], qq[i], eps_row[i] = yt, p_keep, q_other, e
    n_used = sum(len(u) for u in units)
    stats = dict(label_dp_eps_unit=eps_unit, label_dp_rows=n_used, label_dp_flip_rate=flips / max(1, n_used),
                 label_dp_min_eps_row=min((eps_row[i] for u in units for i in u), default=0.0))
    return noisy, pk, qq, eps_row, stats
