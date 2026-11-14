"""End-to-end CPU smoke test on a tiny random Qwen3: every privacy mode trains a few steps, DP
temperature scaling, evaluation, canary audit and the loss-threshold MIA all run and write their files."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dpdecide import audit, calibrate, evaluate, train
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
    assert led["n_units"] == 200 and led["rows_used"] == 600
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
    assert led["release_private"] == (priv in ("dpsgd", "labeldp", "labeldp_raw"))
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
    assert t["private"] and t["n_units"] == 100 and t["rows_excluded_shared_context"] == 0
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
