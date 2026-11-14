"""Label-blind clipping: exactness, label invariance and the sensitivity bound, on a tiny random Qwen3."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dpdecide import blind, privacy, train
from dpdecide.model import DecisionLM
from dpdecide.objectives import row_losses
from dpdecide.schema import write_jsonl
from dpdecide.synth import tickets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("blind")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_tiny_model.py"), "--out", str(d / "tiny")],
                   check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    return d


def _model(tiny, lora=False):
    torch.manual_seed(0)
    return DecisionLM.load(str(tiny / "tiny"), device="cpu", lora=lora, lora_r=4)


def _flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


@pytest.mark.parametrize("objective", ["ce", "brier", "rlcd"])
def test_unclipped_blind_gradient_equals_autograd(tiny, objective):
    m = _model(tiny)
    qs = tickets(1, seed=5)  # one unit: 3 rows (k = 4, 4, 2)
    labels = [q.answer for q in qs]
    g1 = torch.Generator().manual_seed(3)
    grads, st = blind.unit_gradient(m, qs, labels, objective, 4, g1, clip=1e9, max_len=128)
    assert st["blind_factor"] == 1.0
    g2 = torch.Generator().manual_seed(3)
    loss, _ = row_losses(objective, m.logits(qs, 128), torch.tensor(labels), 4, g2)
    ref = torch.autograd.grad(loss.sum(), m.trainable_parameters())
    assert torch.allclose(_flat(grads), _flat(ref), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("objective", ["ce", "brier", "rlcd"])
def test_factor_is_label_blind_and_norm_bounded(tiny, objective):
    m = _model(tiny)
    qs = tickets(1, seed=7)
    factors, norms = [], []
    for labels in ([q.answer for q in qs], [0, 0, 0], [3, 2, 1], [1, 3, 0]):
        g = torch.Generator().manual_seed(11)  # same sampled actions for rlcd
        grads, st = blind.unit_gradient(m, qs, labels, objective, 4, g, clip=1e-3, max_len=128)
        factors.append(st["blind_factor"])
        norms.append(float(_flat(grads).norm()))
    assert max(factors) == pytest.approx(min(factors), rel=1e-9)
    assert max(norms) <= 1e-3 * (1 + 1e-4)


def test_auto_blind_norm_below_clip(tiny):
    m = _model(tiny)
    qs = tickets(2, seed=9)[:3]
    grads, st = blind.unit_gradient(m, qs, [q.answer for q in qs], "ce", 4, None, clip=0.5, auto_gamma=0.01,
                                    max_len=128)
    assert float(_flat(grads).norm()) < 0.5
    assert st["blind_factor"] == pytest.approx(0.5 / (st["blind_bound"] + 0.01))


def test_lora_blind_runs_and_aggregator_rejects_oversized(tiny):
    m = _model(tiny, lora=True)
    qs = tickets(1, seed=2)
    grads, st = blind.unit_gradient(m, qs, [q.answer for q in qs], "rlcd", 4, None, clip=0.1, max_len=128)
    agg = privacy.Aggregator(m.trainable_parameters(), "dp", 0.1, 1.0, 8.0, noise_seed=0)
    agg.add_clipped(grads, st["blind_bound"], st["blind_factor"])
    with pytest.raises(RuntimeError):
        agg.add_clipped([g * 10 for g in grads], 1.0, 1.0)


def test_std_auto_clip_bound():
    w = torch.nn.Parameter(torch.zeros(3))
    agg = privacy.Aggregator([w], "clip", clip=1.0, sigma=0.0, expected_batch=1.0, noise_seed=0, auto_gamma=0.01)
    w.grad = torch.tensor([3.0, 4.0, 0.0])
    agg.add_unit(1)
    agg.finish()
    assert float(w.grad.norm()) == pytest.approx(5.0 / 5.01)


@pytest.mark.parametrize("objective,extra", [("rlcd", ["--only-k", "2"]), ("ce", []), ("brier", ["--auto-clip-gamma", "0.01"])])
def test_train_blind_end_to_end(tiny, objective, extra):
    d = tiny
    write_jsonl(d / "train.jsonl", tickets(60, seed=1))
    out = d / f"run-{objective}"
    train.main(["--init", str(d / "tiny"), "--data", str(d / "train.jsonl"), "--start", "0", "--objective", objective,
                "--privacy", "dpsgd", "--eps", "4", "--clip-labels", "blind", "--clip", "0.05", "--expected-batch", "4",
                "--epochs", "1", "--max-steps", "2", "--device", "cpu", "--max-len", "128", "--out", str(out),
                "--overwrite"] + extra)
    import json
    led = json.loads((out / "privacy.json").read_text())
    assert led["clip_labels"] == "blind" and led["eps_pld"] <= 4.0 * 1.001
    if extra[:1] == ["--only-k"]:
        assert led["rows_used"] == 60  # only the paging question of each ticket is binary


@pytest.mark.parametrize("auto", [0.0, 0.01])
def test_row_budget_factors_depend_only_on_own_row(tiny, auto):
    """--blind-budget row: each row's factor comes from its own bound with budget C/M, so the unit
    gradient is the sum of the rows' gradients computed alone, and the norm stays <= C."""
    m = _model(tiny)
    qs = tickets(1, seed=9)  # 3 rows
    labels = [q.answer for q in qs]
    C, M = 1e-3, 4
    whole, st = blind.unit_gradient(m, qs, labels, "ce", 4, None, clip=C, auto_gamma=auto, max_len=128, row_budget=M)
    parts = [blind.unit_gradient(m, [q], [y], "ce", 4, None, clip=C, auto_gamma=auto, max_len=128, row_budget=M)[0]
             for q, y in zip(qs, labels)]
    summed = [sum(ps) for ps in zip(*parts)]
    rel = lambda a, b: float((_flat(a) - _flat(b)).norm() / _flat(b).norm())  # noqa: E731
    assert rel(whole, summed) < 1e-4  # batch-of-3 vs batch-of-1 forward passes differ only by float error
    assert float(_flat(whole).norm()) <= C * (1 + 1e-4)
    # the unit-level factor does NOT decompose this way
    unit, _ = blind.unit_gradient(m, qs, labels, "ce", 4, None, clip=C, auto_gamma=auto, max_len=128)
    alone = [blind.unit_gradient(m, [q], [y], "ce", 4, None, clip=C, auto_gamma=auto, max_len=128)[0]
             for q, y in zip(qs, labels)]
    assert rel(unit, [sum(ps) for ps in zip(*alone)]) > 1e-2


def test_row_budget_rejects_units_above_cap(tiny):
    m = _model(tiny)
    qs = tickets(1, seed=9)
    with pytest.raises(AssertionError):
        blind.unit_gradient(m, qs, [q.answer for q in qs], "ce", 4, None, clip=1.0, max_len=128, row_budget=2)
