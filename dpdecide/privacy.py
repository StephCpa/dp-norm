"""DP-SGD accounting (Poisson-sampled Gaussian, add/remove-one-unit adjacency) and the update rule.

Accounting: dp_accounting RDP for calibration (conservative), PLD reported alongside (tighter).
Composition across stages: the training slice, the DP temperature-scaling slice and the canary pool
must be DISJOINT sets of privacy units; then the end-to-end guarantee is the max over stages
(parallel composition), not the sum. train.py and calibrate.py enforce disjointness by context.
"""
from __future__ import annotations

import hashlib
import math
import secrets

import dp_accounting as dpa
import torch
from absl import logging as _absl_logging
from dp_accounting import pld, rdp

# RDP orders that fail to converge are dropped by dp_accounting with a warning; that is expected and safe.
_absl_logging.set_verbosity(_absl_logging.ERROR)


def _event(sigma: float, q: float, steps: int):
    return dpa.SelfComposedDpEvent(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma)), steps)


def epsilon(sigma: float, q: float, steps: int, delta: float, method: str = "rdp") -> float:
    acc = rdp.RdpAccountant() if method == "rdp" else pld.PLDAccountant()
    acc.compose(_event(sigma, q, steps))
    return float(acc.get_epsilon(delta))


def calibrate_sigma(eps: float, delta: float, q: float, steps: int, method: str = "pld") -> float:
    """Smallest noise multiplier meeting (eps, delta). PLD is tight (default); RDP is the conservative
    classic. The ledger reports the epsilon under both for whatever sigma is used."""
    make = rdp.RdpAccountant if method == "rdp" else pld.PLDAccountant
    return float(dpa.calibrate_dp_mechanism(make, lambda s: _event(s, q, steps), eps, delta,
                                            bracket_interval=dpa.LowerEndpointAndGuess(0.1, 1.0)))


def mechanism_seed(randomness: str, base_seed: int, run_id: str, stream: str) -> int:
    """Seed for a privacy-critical random stream (Poisson sampling, Gaussian noise, randomized response,
    the exponential mechanism). The guarantees assume these draws are secret.
    'secure': fresh OS entropy, never logged. This is the only mode whose output is a private release.
    'seeded': reproducible research simulation; the seed is derived from (base seed, run id, stream) so
              no two runs or streams share draws, but anyone who knows the seed can recompute the noise,
              so the ledger marks the run release_private = false."""
    if randomness == "secure":
        return secrets.randbits(63)
    if randomness != "seeded":
        raise ValueError(randomness)
    h = hashlib.sha256(f"{base_seed}:{run_id}:{stream}".encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


def selection_epsilon(sigma: float, q: float, steps: int, delta: float, n_candidates: int) -> float:
    """Epsilon (PLD) of releasing one model chosen among n_candidates DP-SGD runs on the same private
    slice. Choosing on public data is post-processing of ALL candidates, so the candidates compose; the
    per-run epsilon does not cover the selection (Liu & Talwar 2019; Papernot & Steinke 2022 for cheaper
    randomized-repetition tuning)."""
    return epsilon(sigma, q, steps * n_candidates, delta, "pld")


# Fields that may appear in a released ledger (privacy.json, temperature.json, the plan log line).
# Everything else a run computes about the private slice (unit and row counts, drops, flip rates,
# per-unit statistics, hashes) is internal: it is written to private_diagnostics/ and never released.
# Releasing exact cardinalities is incompatible with add/remove adjacency (neighbours differ by one
# unit, so the count itself distinguishes them), and RR flip counts are the mechanism's secret coins.
RELEASE_LEDGER_KEYS = frozenset({
    "privacy", "objective", "unit", "max_rows_per_unit", "expected_batch", "sampling_rate", "steps",
    "epochs_expected", "schedule", "n_units_public", "sigma", "accountant", "clip", "clip_labels", "blind_budget",
    "auto_clip_gamma", "only_k", "delta", "eps_target", "eps_rdp", "eps_pld", "label_dp_eps_unit", "adjacency",
    "randomness", "release_private", "release_blockers", "release_scope", "not_covered", "notes", "partition"})
RELEASE_TEMPERATURE_KEYS = frozenset({
    "temperature", "eps", "delta", "sensitivity", "grid_size", "grid", "private", "randomness",
    "release_private", "release_blockers", "max_rows_per_unit", "partition", "excluded_by_context"})


def split_release(record: dict, allowed: frozenset) -> tuple[dict, dict]:
    """(public, internal): the allowlisted part may be released; the rest goes to private_diagnostics/."""
    return ({k: v for k, v in record.items() if k in allowed},
            {k: v for k, v in record.items() if k not in allowed})


def composed_epsilon(events: list[tuple[float, float, int]], delta: float) -> float:
    """PLD epsilon of running every (sigma, q, steps) DP-SGD event on the same private slice. Use this,
    not one candidate's parameters times n, when candidates differ (review of e9190c3, P1)."""
    acc = pld.PLDAccountant()
    for sigma, q, steps in events:
        if not sigma or sigma <= 0:
            return math.inf
        acc.compose(_event(sigma, q, steps))
    return float(acc.get_epsilon(delta))


def plan(n_units: int, expected_batch: float, epochs: float) -> tuple[float, int]:
    q = min(1.0, expected_batch / n_units)
    return q, max(1, math.ceil(epochs / q))


class Aggregator:
    """Per-unit gradient aggregation.
    mode 'none' : plain sum (units may be batched together)
    mode 'clip' : per-unit clipping to C, no noise (isolates the clipping effect; NOT private)
    mode 'dp'   : per-unit clipping to C plus N(0, sigma^2 C^2) on the sum
    The result is divided by the EXPECTED batch size (Poisson sampling): dividing by the realized size
    would make the update depend on how many units were sampled, which the accountant does not cover."""

    def __init__(self, params: list[torch.nn.Parameter], mode: str, clip: float, sigma: float,
                 expected_batch: float, noise_seed: int, auto_gamma: float = 0.0):
        assert mode in ("none", "clip", "dp")
        self.params, self.mode, self.clip, self.sigma, self.b = params, mode, clip, sigma, expected_batch
        # auto_gamma > 0: automatic clipping (Bu et al. 2023, AUTO-S) scaled to C: factor = C / (||g|| + gamma),
        # so every unit's contribution has norm < C and the sensitivity is C, as with threshold clipping.
        self.auto_gamma = auto_gamma
        self.acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        dev = params[0].device
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(noise_seed)
        self.reset_stats()

    def reset_stats(self):
        self.units, self.clipped, self.norms = 0, 0, []

    def add_unit(self, n_units: int = 1) -> None:
        """Call right after backward() of one unit's summed loss (or, in mode 'none', of a batch of
        n_units units). Moves .grad into the accumulator and clears it."""
        grads = [p.grad for p in self.params]
        if self.mode == "none":
            for a, g in zip(self.acc, grads):
                if g is not None:
                    a.add_(g.float())
        else:
            assert n_units == 1, "clipping needs one unit per backward"
            sq = torch.zeros((), device=self.acc[0].device)
            for g in grads:
                if g is not None:
                    sq += g.float().pow(2).sum()
            norm = sq.sqrt().item()
            if self.auto_gamma > 0:
                factor = self.clip / (norm + self.auto_gamma)
            else:
                factor = min(1.0, self.clip / (norm + 1e-12))
            self.norms.append(norm)
            self.clipped += int(factor < 1.0)
            for a, g in zip(self.acc, grads):
                if g is not None:
                    a.add_(g.float(), alpha=factor)
        for p in self.params:
            p.grad = None
        self.units += n_units

    def add_clipped(self, grads: list[torch.Tensor], bound: float, factor: float) -> None:
        """Add one unit's gradient that was already clipped elsewhere (label-blind clipping). The caller
        guarantees ||grads|| <= clip; it is re-checked here because the privacy proof depends on it."""
        assert self.mode in ("clip", "dp")
        norm = sum(float(g.float().pow(2).sum()) for g in grads) ** 0.5
        if norm > self.clip * (1 + 1e-3) + 1e-8:
            raise RuntimeError(f"externally clipped gradient has norm {norm} > clip {self.clip}")
        for a, g in zip(self.acc, grads):
            a.add_(g.float())
        self.norms.append(bound)
        self.clipped += int(factor < 1.0)
        self.units += 1

    def finish(self) -> dict:
        """Write the private mean gradient into .grad and reset. Returns step stats."""
        noise_norm_sq = 0.0
        for p, a in zip(self.params, self.acc):
            if self.mode == "dp":
                z = torch.normal(0.0, self.sigma * self.clip, size=a.shape, generator=self.gen, device=a.device)
                noise_norm_sq += float(z.pow(2).sum())
                a.add_(z)
            p.grad = (a / self.b).to(p.dtype)
            a.zero_()
        norms = sorted(self.norms)
        stats = dict(units=self.units,
                     clip_frac=(self.clipped / self.units) if self.units and self.mode != "none" else 0.0,
                     unit_norm_median=norms[len(norms) // 2] if norms else float("nan"),
                     noise_norm=math.sqrt(noise_norm_sq))
        self.reset_stats()
        return stats
