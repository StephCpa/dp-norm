"""Calibration, sharpness and membership-attack metrics."""
from __future__ import annotations

import numpy as np


def softmax_rows(rows: list[list[float]], temperature: float = 1.0) -> list[np.ndarray]:
    out = []
    for r in rows:
        z = np.asarray(r, float) / temperature
        z = z - z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return out


def _bins(conf: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip(np.floor(conf * n_bins + 1e-9).astype(int), 0, n_bins - 1)


def murphy_top(conf: np.ndarray, hit: np.ndarray, n_bins: int = 15) -> dict:
    """Binned top-label decomposition of the Brier score of (confidence, correctness):
    brier_top ~= reliability - resolution + uncertainty. ECE is the L1 analogue of reliability.
    Resolution is what privacy noise costs even after recalibration."""
    ids = _bins(conf, n_bins)
    abar = hit.mean()
    ece = rel = res = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            w = m.mean()
            gap = hit[m].mean() - conf[m].mean()
            ece += w * abs(gap)
            rel += w * gap ** 2
            res += w * (hit[m].mean() - abar) ** 2
    return dict(ece=ece, reliability=rel, resolution=res, uncertainty=abar * (1 - abar),
                brier_top=float(((conf - hit) ** 2).mean()))


def summarize(probs: list[np.ndarray], answers: np.ndarray, n_bins: int = 15) -> dict:
    conf = np.array([p.max() for p in probs])
    pred = np.array([p.argmax() for p in probs])
    hit = (pred == answers).astype(float)
    nll = -np.mean([np.log(max(p[a], 1e-12)) for p, a in zip(probs, answers)])
    brier = np.mean([((p - np.eye(len(p))[a]) ** 2).sum() for p, a in zip(probs, answers)])
    out = dict(n=len(answers), acc=hit.mean(), nll=nll, brier=brier, mean_conf=conf.mean(),
               overconf=conf.mean() - hit.mean())
    out.update(murphy_top(conf, hit, n_bins))
    return {k: float(v) for k, v in out.items()}


def roc_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half."""
    s = np.concatenate([scores_pos, scores_neg])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(scores_pos), len(scores_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> float:
    thr = np.quantile(scores_neg, 1 - fpr, method="higher")
    return float((scores_pos > thr).mean())
