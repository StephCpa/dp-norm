"""End-to-end CPU smoke test on a tiny random Qwen3: every privacy mode trains a few steps, DP
temperature scaling, evaluation, canary audit and the loss-threshold MIA all run and write their files."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dpdecide import privacy, audit, calibrate, evaluate, train
from dpdecide.schema import write_jsonl
from dpdecide.synth import tickets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_tiny_model.py"), "--out", str(d / "tiny")],
                   check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    write_jsonl(d / "train.jsonl", tickets(400, seed=1, prefix="a"))      # 1200 rows
    write_jsonl(d / "val.jsonl", tickets(40, seed=2, prefix="v"))
    write_jsonl(d / "test.jsonl", tickets(60, seed=3, prefix="t"))
    write_jsonl(d / "pool.jsonl", tickets(80, seed=4, prefix="p"))
    common = ["--device", "cpu", "--max-len", "128", "--overwrite"]
    train.main(["--init", str(d / "tiny"), "--data", str(d / "train.jsonl"), "--start", "0", "--limit", "300",
                "--objective", "ce", "--privacy", "none", "--expected-batch", "16", "--epochs", "1", "--lr", "3e-3",
                "--out", str(d / "warmup")] + common)
    return d, common


ARMS = [("rlcd", "none", []), ("rlcd", "dpsgd", ["--eps", "4"]), ("ce", "dpsgd", ["--eps", "1"]),
        ("brier", "clip", []), ("rlcd", "labeldp", ["--eps", "2"]), ("ce", "labeldp", ["--eps", "2"]),
        ("brier", "labeldp", ["--eps", "2"]), ("rlcd", "labeldp_raw", ["--eps", "2"]), ("rlvr", "none", [])]


@pytest.mark.parametrize("objective,priv,extra", ARMS)
def test_arm_trains(env, objective, priv, extra):
    d, common = env
    out = d / f"{objective}-{priv}"
    train.main(["--init", str(d / "warmup"), "--data", str(d / "train.jsonl"), "--start", "600",
                "--objective", objective, "--privacy", priv, "--expected-batch", "8", "--epochs", "1",
                "--max-steps", "3", "--lr", "1e-3", "--eval-data", str(d / "val.jsonl"), "--eval-every", "3",
                "--eval-n", "30", "--out", str(out)] + extra + common)
    led = json.loads((out / "privacy.json").read_text())
    internal = json.loads((out / "private_diagnostics/ledger_internal.json").read_text())
    assert internal["n_units"] == 200 and internal["rows_used"] == 600
    # reaudit of e9190c3, P0: only allowlisted fields are released; no cardinalities, no RR flip statistics
    assert set(led) <= privacy.RELEASE_LEDGER_KEYS, sorted(set(led) - privacy.RELEASE_LEDGER_KEYS)
    assert not {"n_units", "rows_used", "label_dp_flip_rate", "label_dp_rows"} & set(led)
    if priv == "dpsgd":
        assert led["sigma"] > 0 and led["eps_pld"] <= float(extra[1]) * 1.001
    if priv.startswith("labeldp"):
        assert led["label_dp_eps_unit"] == float(extra[1]) and led["delta"] == 0.0
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert any("eval_ece" in r for r in log)
    # review P0-2: the release-safe log carries no unnoised private statistics
    allowed = {"step", "lr", "update_norm", "eval_step", "eval_acc", "eval_ece", "eval_brier", "eval_nll",
               "eval_overconf", "eval_resolution"}
    extra_keys = [sorted(set(r) - allowed) for r in log if r.get("event") != "plan"]
    assert not any(extra_keys), extra_keys
    assert "data_sha256" not in led and led["randomness"] == "secure"
    # dpsgd without --n-units-public derives q and steps from the private count: not a private release
    assert led["release_private"] == (priv in ("labeldp", "labeldp_raw")), led["release_blockers"]
    plan = [r for r in log if r.get("event") == "plan"][0]
    assert set(plan) - {"event"} <= privacy.RELEASE_LEDGER_KEYS
    diag = [json.loads(l) for l in (out / "private_diagnostics/train_stats.jsonl").read_text().splitlines()]
    assert diag and any("wall_sec" in r for r in diag)
    assert (out / "model.safetensors").exists()


def test_calibrate_evaluate_audit(env):
    d, common = env
    model = d / "rlcd-dpsgd"
    if not model.exists():
        pytest.skip("depends on test_arm_trains")
    calibrate.main(["--model", str(model), "--data", str(d / "train.jsonl"), "--start", "300", "--limit", "300",
                    "--exclude-data", str(d / "train.jsonl"), "--exclude-start", "600", "--eps", "4",
                    "--device", "cpu", "--max-len", "128", "--out", str(model / "temperature.json")])
    t = json.loads((model / "temperature.json").read_text())
    assert t["private"] and set(t) <= privacy.RELEASE_TEMPERATURE_KEYS and "n_units" not in t
    ti = json.loads((model / "private_diagnostics/temperature_internal.json").read_text())
    assert ti["n_units"] == 100 and ti["rows_excluded_shared_context"] == 0
    evaluate.main(["--model", str(model), "--data", str(d / "test.jsonl"), "--temperature",
                   str(model / "temperature.json"), "--device", "cpu", "--max-len", "128", "--out", str(model / "eval")])
    m = json.loads((model / "eval/metrics.json").read_text())
    assert m["overall"]["n"] == 180 and "resolution" in m["overall"] and "primitive=noul" in m

    for kind in ("mislabel", "labelswap"):
        cdir = d / f"can-{kind}"
        audit.main(["make-canaries", "--pool", str(d / "pool.jsonl"), "--exclude", f"{d / 'train.jsonl'}:600:0",
                    "--n", "60", "--kind", kind, "--out-dir", str(cdir)])
        priv = ["--privacy", "dpsgd", "--eps", "8"] if kind == "mislabel" else ["--privacy", "labeldp", "--eps", "8"]
        out = d / f"canary-{kind}"
        train.main(["--init", str(d / "warmup"), "--data", str(d / "train.jsonl"), "--start", "600",
                    "--canaries", str(cdir / "canaries_train.jsonl"), "--objective", "ce", "--expected-batch", "8",
                    "--epochs", "1", "--max-steps", "2", "--out", str(out)] + priv + common)
        audit.main(["score", "--model", str(out), "--secret", str(cdir / "canaries_secret.json"), "--device", "cpu",
                    "--max-len", "128", "--out", str(out / "audit.json")])
        res = json.loads((out / "audit.json").read_text())
        assert res["m"] == 60 and res["eps_lower_bound"] >= 0.0
    audit.main(["mia", "--model", str(d / "rlcd-none"), "--members", f"{d / 'train.jsonl'}:600:0",
                "--nonmembers", f"{d / 'test.jsonl'}:0:0", "--n", "100", "--device", "cpu", "--max-len", "128",
                "--out", str(d / "mia.json")])
    mia = json.loads((d / "mia.json").read_text())
    assert mia["n_members"] == 100 and 0.0 <= mia["auc"] <= 1.0


def test_public_schedule_releases(env):
    """With a declared public unit count, q and steps no longer depend on the private count."""
    d, common = env
    outs = []
    for start, name in (("600", "pub-a"), ("700", "pub-b")):  # different private slices (different counts)
        out = d / name
        train.main(["--init", str(d / "warmup"), "--data", str(d / "train.jsonl"), "--start", start,
                    "--objective", "ce", "--privacy", "dpsgd", "--eps", "4", "--n-units-public", "500",
                    "--expected-batch", "8", "--epochs", "1", "--max-steps", "1", "--out", str(out)] + common)
        outs.append(json.loads((out / "privacy.json").read_text()))
    a, b = outs
    assert a["release_private"] and a["schedule"] == "public"
    assert {k: a[k] for k in ("sampling_rate", "steps", "sigma")} == {k: b[k] for k in ("sampling_rate", "steps", "sigma")}
