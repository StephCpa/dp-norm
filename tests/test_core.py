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
                                                         delta=1e-6, release_blockers=[], release_private=True,
                                                         randomness="secure")))
        (r / "temperature.json").write_text(json.dumps(dict(private=True, eps=1.0, release_blockers=[],
                                                            release_private=True, randomness="secure")))
        runs.append(r)
    # no partition records: disjointness unproven, so training and calibration add up
    text = budget(runs, calibrated=True)
    assert f"train {both:.2f}" in text and "cal 2.0" in text and "simulation" not in text
    assert f"end-to-end {both + 2.0:.2f} sequential" in text
    # provably disjoint hash partitions: parallel composition
    part = lambda lo, hi: dict(scheme="hash", data="d.jsonl", unit="context", salt="s", lo=lo, hi=hi,  # noqa: E731
                               start=0, limit=0)
    for r in runs:
        L = json.loads((r / "privacy.json").read_text()); L["partition"] = part(0.0, 0.8)
        (r / "privacy.json").write_text(json.dumps(L))
        T = json.loads((r / "temperature.json").read_text()); T["partition"] = part(0.8, 1.0)
        (r / "temperature.json").write_text(json.dumps(T))
    assert f"end-to-end {max(both, 2.0):.2f} parallel" in budget(runs, calibrated=True)
    T["partition"] = part(0.7, 1.0)  # overlaps the training range
    (runs[1] / "temperature.json").write_text(json.dumps(T))
    assert "sequential" in budget(runs, calibrated=True)
    # reaudit gap 1: a missing calibration ledger is a refusal when a calibrated release is declared
    (runs[1] / "temperature.json").unlink()
    assert budget(runs, calibrated=True).startswith("no guarantee")
    assert budget(runs, calibrated=False).startswith("uncalibrated release")
    (runs[1] / "temperature.json").write_text(json.dumps(dict(private=False, eps=0.0, release_private=False,
                                                              randomness="secure", release_blockers=["x"])))
    assert budget(runs, calibrated=True).startswith("no guarantee")
    (runs[1] / "privacy.json").unlink()
    assert budget(runs, calibrated=False).startswith("no guarantee")


def test_hash_partition_preserves_unit_adjacency():
    """Reaudit gap 3: adding one raw unit changes a hash partition by that unit only; positional slicing
    can replace several units."""
    from dpdecide.schema import group_units, select_partition, select_rows
    from dpdecide.synth import tickets
    from dpdecide import schema
    base = tickets(300, seed=4)
    extra = tickets(1, seed=99)  # one new unit (3 rows sharing a context)
    raw2 = extra + base           # inserted at the front: shifts every position
    key = lambda rows: {schema.unit_key(q, "context") for q in rows}  # noqa: E731
    h1, h2 = key(select_partition(base, "context", "s:0.0:0.5")), key(select_partition(raw2, "context", "s:0.0:0.5"))
    assert len(h1 ^ h2) <= 1
    p1, p2 = key(select_rows(base, 300, 300)), key(select_rows(raw2, 300, 300))
    assert len(p1 ^ p2) >= 2
    assert not schema.partitions_disjoint(None, None)



def test_budget_cross_candidate_overlap_and_ledger_consistency(tmp_path):
    """Reaudit of df9bb65, P1: every train_i must be disjoint from every cal_j (the review's counterexample:
    A trains on [0, .5) and calibrates on [.5, 1), B the other way round), and ledgers must be consistent."""
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_blind import budget
    part = lambda lo, hi: dict(scheme="hash", data="d.jsonl", unit="context", salt="s", lo=lo, hi=hi,  # noqa: E731
                               start=0, limit=0)
    ok = dict(release_private=True, randomness="secure", release_blockers=[])
    runs = []
    for i, (sig, ceps, tr, ca) in enumerate([(1.0, 0.01, (0, .5), (.5, 1)), (100.0, 5.0, (.5, 1), (0, .5))]):
        r = tmp_path / f"c{i}"
        r.mkdir()
        (r / "privacy.json").write_text(json.dumps(dict(privacy="dpsgd", sigma=sig, sampling_rate=1.0, steps=1,
                                                         delta=1e-6, partition=part(*tr), **ok)))
        (r / "temperature.json").write_text(json.dumps(dict(private=True, eps=ceps, partition=part(*ca), **ok)))
        runs.append(r)
    text = budget(runs, calibrated=True)
    assert "sequential" in text and "parallel" not in text, text
    # contradictory ledgers are refused
    L = json.loads((runs[0] / "privacy.json").read_text())
    for bad in (dict(release_private=True, release_blockers=["seeded mechanism randomness"]),
                dict(release_private=True, randomness="seeded"),
                dict(release_private=False, release_blockers=[]),
                {"release_private": None}):
        (runs[0] / "privacy.json").write_text(json.dumps({**L, **bad}))
        assert budget(runs, calibrated=True).startswith("no guarantee: inconsistent ledger"), bad


def test_row_unit_key_ignores_the_label():
    """Reaudit of df9bb65, P0: without ids, label-only neighbours must get the same row key, hence the same
    hash-partition membership, so membership cannot reveal the label before randomized response."""
    from dpdecide import schema
    from dpdecide.schema import Question
    qs = [Question("choice", "a context", "q", ["a", "b"], answer=y) for y in (0, 1)]
    keys = {schema.unit_key(q, "row") for q in qs}
    assert len(keys) == 1
    for cut in (0.1, 0.3, 0.5, 0.7, 0.9):
        sel = [len(schema.select_partition([q], "row", f"audit:0:{cut}")) for q in qs]
        assert sel[0] == sel[1]
    # different inputs still get different keys
    assert schema.unit_key(Question("choice", "another", "q", ["a", "b"], answer=0), "row") not in keys


def test_blind_selection_ignores_private_temperature(tmp_path):
    """Reaudit of df9bb65, P1: blind is released at T = 1, so its candidate is chosen on raw validation
    metrics; changing a calibrated validation metric must not change the winner."""
    import contextlib, io, json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import summarize_blind as sb
    m = lambda v: {"overall": dict(brier=v, acc=.7, ece=.1, overconf=.1, resolution=.1)}  # noqa: E731
    for i, raw in enumerate((0.30, 0.20)):  # lr2 is better on RAW validation
        d = tmp_path / f"A-thr-blind-ce-lr{i + 1}"
        for sub, v in (("eval_val", raw), ("eval_val_ts", .1 + i * .1), ("eval", .3), ("eval_ts", .2)):
            (d / sub).mkdir(parents=True)
            (d / sub / "metrics.json").write_text(json.dumps(m(v)))

    def winner():
        sys.argv = ["summarize_blind", "--runs", str(tmp_path)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sb.main()
        return [ln for ln in buf.getvalue().splitlines() if ln.startswith("| ce | thr | blind")][0].split("|")[4].strip()

    assert winner() == "2"
    (tmp_path / "A-thr-blind-ce-lr1/eval_val_ts/metrics.json").write_text(json.dumps(m(.9)))
    assert winner() == "2"
