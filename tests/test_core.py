import math

import numpy as np
import pytest
import torch

from dpdecide import labeldp, privacy
from dpdecide.audit import eps_lower_bound, one_run_audit
from dpdecide.calibrate import exp_mech
from dpdecide.metrics import murphy_top, roc_auc
from dpdecide.objectives import row_losses
from dpdecide.schema import Question, group_units, render_prompt
from dpdecide.synth import tickets


def test_prompt_matches_eve_rlcd_format():
    q = Question("choice", "  ctx  ", "Q?", ["a", "b"], answer=0)
    assert render_prompt(q) == ("User: Context:\nctx\n\nQuestion: Q?\nOptions:\nA) a\nB) b\n"
                                "Answer with the letter only.\nAssistant: The answer is")


def test_units_group_ticket_questions_and_cap():
    qs = tickets(10, seed=0)
    units, dropped = group_units(qs, "context", 4)
    assert len(units) == 10 and all(len(u) == 3 for u in units) and dropped == 0
    units, dropped = group_units(qs, "context", 2)
    assert dropped == 10 and all(len(u) == 2 for u in units)
    units, _ = group_units(qs, "row", 4)
    assert len(units) == 30


def test_rr_params_and_unit_split():
    pk, q = labeldp.rr_params(math.log(3), 2)
    assert pk == pytest.approx(0.75) and q == pytest.approx(0.25)
    qs = tickets(50, seed=1)
    units, _ = group_units(qs, "context", 4)
    noisy, pk, qq, eps_row, st = labeldp.privatize(qs, units, 3.0, np.random.default_rng(0))
    assert all(e == pytest.approx(1.0) for e in eps_row)  # 3 rows per unit -> eps/3 each
    assert all(0 <= y < q.k for y, q in zip(noisy, qs))


def test_rr_debias_is_unbiased_empirically():
    qs = [Question("choice", f"c{i}", "q", [str(j) for j in range(5)], answer=i % 5, id=f"r{i}") for i in range(20000)]
    units = [[i] for i in range(len(qs))]
    noisy, pk, qq, _, _ = labeldp.privatize(qs, units, 1.0, np.random.default_rng(3))
    noisy, pk, qq = map(np.array, (noisy, pk, qq))
    y = np.array([q.answer for q in qs])
    for a in range(5):  # c~ for "action a" averaged over rows whose true label is / is not a
        ct = ((noisy == a) - qq) / (pk - qq)
        assert ct[y == a].mean() == pytest.approx(1.0, abs=0.08)
        assert ct[y != a].mean() == pytest.approx(0.0, abs=0.04)


def _exact_expected_grad(objective, logits, y, eps, k):
    """E over the RR label of the corrected loss gradient, by enumeration."""
    pk, q = labeldp.rr_params(eps, k)
    total = torch.zeros_like(logits)
    for yt in range(k):
        w = pk if yt == y else q
        lg = logits.clone().requires_grad_(True)
        loss, _ = row_losses(objective, lg, torch.tensor([yt]), 4,
                             rr=(torch.tensor([pk]), torch.tensor([q])))
        loss.sum().backward()
        total += w * lg.grad
    return total


def test_debiased_brier_gradient_equals_clean_in_expectation():
    torch.manual_seed(0)
    logits = torch.full((1, 26), -1e4)
    logits[0, :4] = torch.randn(4)
    lg = logits.clone().requires_grad_(True)
    clean, _ = row_losses("brier", lg, torch.tensor([2]), 4)
    clean.sum().backward()
    exp = _exact_expected_grad("brier", logits, 2, 1.0, 4)
    assert torch.allclose(exp[0, :4], lg.grad[0, :4], atol=1e-5)


def test_forward_corrected_ce_is_noisy_channel_nll():
    logits = torch.full((1, 26), -1e4)
    logits[0, :3] = torch.tensor([0.3, -0.2, 1.1])
    pk, q = labeldp.rr_params(0.7, 3)
    loss, _ = row_losses("ce", logits, torch.tensor([1]), 4, rr=(torch.tensor([pk]), torch.tensor([q])))
    p = torch.softmax(logits[0, :3], -1)
    T = torch.full((3, 3), q)
    T.fill_diagonal_(pk)
    assert loss.item() == pytest.approx(-math.log((p @ T)[1].item()), rel=1e-5)


def test_rlcd_expected_gradient_is_half_brier_gradient():
    """sum_a pi_a (c_a - pi_a) grad log pi_a = -1/2 grad Brier, checked through the per-row loss by
    enumerating every action with its probability (no sampling)."""
    torch.manual_seed(1)
    z = torch.randn(4, requires_grad=True)
    p = torch.softmax(z, -1)
    y = 1
    c = torch.zeros(4)
    c[y] = 1
    surrogate = -(p.detach() * (c - p.detach()) * torch.log(p)).sum()  # E_a[-(r_a) log pi_a] with r = c - pi_a
    g1 = torch.autograd.grad(surrogate, z)[0]
    brier = ((torch.softmax(z, -1) - c) ** 2).sum()
    g2 = torch.autograd.grad(brier, z)[0]
    assert torch.allclose(g1, 0.5 * g2, atol=1e-6)


def test_aggregator_clips_per_unit_and_divides_by_expected_batch():
    w = torch.nn.Parameter(torch.zeros(3))
    agg = privacy.Aggregator([w], "clip", clip=1.0, sigma=0.0, expected_batch=4.0, noise_seed=0)
    for g in ([3.0, 4.0, 0.0], [0.3, 0.4, 0.0]):  # norms 5 (clipped to 1) and 0.5 (kept)
        w.grad = torch.tensor(g)
        agg.add_unit(1)
    st = agg.finish()
    assert torch.allclose(w.grad, torch.tensor([0.6 + 0.3, 0.8 + 0.4, 0.0]) / 4.0)
    assert st["clip_frac"] == pytest.approx(0.5)


def test_aggregator_noise_scale():
    w = torch.nn.Parameter(torch.zeros(200000))
    agg = privacy.Aggregator([w], "dp", clip=0.5, sigma=2.0, expected_batch=10.0, noise_seed=1)
    st = agg.finish()  # an empty Poisson batch still gets noise
    assert w.grad.std().item() == pytest.approx(2.0 * 0.5 / 10.0, rel=0.02)
    assert st["units"] == 0


def test_accountant_calibration_round_trip():
    q, steps = privacy.plan(32000, 128, 2)
    assert steps == 500
    for eps in (1.0, 4.0, 8.0):
        s = privacy.calibrate_sigma(eps, 1e-6, q, steps, "rdp")
        e_rdp = privacy.epsilon(s, q, steps, 1e-6, "rdp")
        e_pld = privacy.epsilon(s, q, steps, 1e-6, "pld")
        assert eps * 0.95 <= e_rdp <= eps * 1.001
        assert e_pld <= e_rdp + 1e-6
        s2 = privacy.calibrate_sigma(eps, 1e-6, q, steps, "pld")
        assert s2 <= s and privacy.epsilon(s2, q, steps, 1e-6, "pld") <= eps * 1.001


def test_one_run_audit_bounds():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 1000)
    assert one_run_audit(rng.standard_normal(1000), truth)["eps_lower_bound"] < 0.5
    perfect = one_run_audit(truth + 0.01 * rng.standard_normal(1000), truth)
    assert perfect["eps_lower_bound"] > 3.0 and perfect["auc"] == pytest.approx(1.0)
    assert eps_lower_bound(10, 5, 0.05) == 0.0


def test_exp_mech_extremes():
    u = np.array([0.0, -10.0, -20.0])
    rng = np.random.default_rng(0)
    assert all(exp_mech(u, 1e6, 2.0, rng) == 0 for _ in range(50))
    picks = [exp_mech(u, 1e-6, 2.0, rng) for _ in range(3000)]
    assert min(np.bincount(picks, minlength=3)) > 800


def test_murphy_identity_when_bins_are_pure():
    rng = np.random.default_rng(0)
    conf = rng.choice([0.55, 0.75, 0.95], 50000)
    hit = (rng.random(50000) < conf - 0.05).astype(float)
    m = murphy_top(conf, hit, 15)
    assert m["brier_top"] == pytest.approx(m["reliability"] - m["resolution"] + m["uncertainty"], abs=1e-9)
    assert m["ece"] == pytest.approx(0.05, abs=0.01)


def test_auc():
    assert roc_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert roc_auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_rr_duplicate_ids_get_independent_draws():
    """Regression (review P1-6): rows sharing an id used to reuse one random stream."""
    qs = [Question("choice", f"c{i}", "q", ["a", "b"], answer=0, id="dup") for i in range(4000)]
    units = [[i] for i in range(len(qs))]
    noisy, pk, _, _, _ = labeldp.privatize(qs, units, 0.5, np.random.default_rng(11))
    noisy = np.array(noisy)
    assert 0.0 < noisy.mean() < 1.0                       # not all identical
    assert abs((noisy == 0).mean() - pk[0]) < 0.03        # each row an independent RR draw


def test_mechanism_seeds():
    """Review P0-2: secure seeds are fresh; seeded ones are reproducible but distinct per run and stream."""
    s1 = {privacy.mechanism_seed("secure", 0, "run", "noise") for _ in range(5)}
    assert len(s1) == 5
    a = privacy.mechanism_seed("seeded", 0, "run-a", "noise")
    assert a == privacy.mechanism_seed("seeded", 0, "run-a", "noise")
    assert a != privacy.mechanism_seed("seeded", 0, "run-b", "noise")
    assert a != privacy.mechanism_seed("seeded", 0, "run-a", "poisson-sampling")
    with pytest.raises(ValueError):
        privacy.mechanism_seed("fixed", 0, "run", "noise")


def test_selection_epsilon_composes_candidates():
    one = privacy.epsilon(1.0, 0.01, 500, 1e-6, "pld")
    two = privacy.selection_epsilon(1.0, 0.01, 500, 1e-6, 2)
    assert two > one * 1.2 and two == pytest.approx(privacy.epsilon(1.0, 0.01, 1000, 1e-6, "pld"))


def test_composed_epsilon_heterogeneous_and_refusals(tmp_path):
    """Reaudit of e9190c3, P1: compose each candidate's own event; refuse missing or non-private records."""
    import json, sys
    from pathlib import Path
    e1, e2 = (1.0, 0.01, 500), (0.6, 0.02, 800)
    both = privacy.composed_epsilon([e1, e2], 1e-6)
    assert both > privacy.epsilon(*e2, 1e-6, "pld") > privacy.epsilon(*e1, 1e-6, "pld")
    assert privacy.composed_epsilon([e1, e1], 1e-6) == pytest.approx(privacy.selection_epsilon(*e1, 1e-6, 2))
    assert privacy.composed_epsilon([e1, (0.0, 0.01, 500)], 1e-6) == float("inf")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_blind import budget
    runs = []
    for name, (s, q, st) in (("a", e1), ("b", e2)):
        r = tmp_path / name
        r.mkdir()
        (r / "privacy.json").write_text(json.dumps(dict(privacy="dpsgd", sigma=s, sampling_rate=q, steps=st,
                                                         delta=1e-6, release_blockers=[])))
        (r / "temperature.json").write_text(json.dumps(dict(private=True, eps=1.0, release_blockers=[])))
        runs.append(r)
    text = budget(runs)
    assert f"train {both:.2f}" in text and "cal 2.0" in text and "simulation" not in text
    (runs[1] / "temperature.json").write_text(json.dumps(dict(private=False, eps=0.0)))
    assert budget(runs).startswith("no guarantee")
    (runs[1] / "privacy.json").unlink()
    assert budget(runs).startswith("no guarantee")
