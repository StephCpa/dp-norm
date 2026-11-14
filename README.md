# dpdecide: differentially private training of typed decision models

Research code for direction D1 in `claude/dp-jev-feasibility.md`: train Jev-style "System One" decision models
(eve-rlcd's RLCD recipe on Qwen3-0.6B-Base) under differential privacy and measure what happens to calibration,
the property such models are sold on.

It is self-contained (no eve-rlcd code is imported), but byte-compatible with eve-rlcd: the same JSONL rows, the
same prompt, the same last-token letter readout. eve-rlcd's data-v1 files and full fine-tune checkpoints work as is.

## What is in it
| module | purpose |
|---|---|
| `dpdecide/train.py` | objectives `ce`, `brier`, `rlcd` (bandit, reward c − p_a), `rlvr`; privacy `none`, `clip` (diagnostic), `dpsgd`, `labeldp`, `labeldp_raw` (control); Poisson sampling of privacy units for every arm; a `privacy.json` ledger per run |
| `dpdecide/privacy.py` | PLD/RDP accounting (dp_accounting), sigma calibration, per-unit clipping + noise, normalization by the expected batch |
| `dpdecide/labeldp.py` | k-ary randomized response once per label, epsilon split over a unit's rows |
| `dpdecide/objectives.py` | per-row losses with exact label-DP corrections (debiased outcome / target, forward-corrected CE) |
| `dpdecide/calibrate.py` | DP temperature scaling (exponential mechanism on summed Brier), enforces a context-disjoint calibration slice |
| `dpdecide/evaluate.py` | accuracy, ECE, Brier, NLL, overconfidence, reliability/resolution split, per source and primitive |
| `dpdecide/blind.py` | label-blind clipping: clip factor from the worst case over labels (per-row Gram of logit gradients), same sensitivity C; `--clip-labels blind`, and `--auto-clip-gamma` for automatic clipping (std or blind) |
| `dpdecide/audit.py` | one-run canary audit (Steinke et al. 2023; mislabel canaries for DP-SGD, label-swap canaries for label DP) and a loss-threshold membership attack |
| `scripts/run_grid.sh` | the gated GPU run plan (phases 0, 1, 1b, 2, 3) |
| `scripts/summarize.py` | one table of every run plus automatic checks of the pre-registered predictions in `PLAN.md` |
| `scripts/summarize_blind.py` | the blindA phase: best lr per arm (by val Brier after DP temperature) and checks of B1-B4 in `PLAN_BLIND.md` |
| `scripts/dry_run_blind_cpu.sh` | phase0 + blindA on CPU with a tiny model (plumbing check, about 12 minutes) |
| `scripts/dry_run_cpu.sh` | the entire run plan on CPU with a tiny random model and synthetic data (plumbing check) |
| `gates/` | the zero-GPU gates that shaped the plan: H1 toy gate, label-DP RLCD simulation, noisy-release recalibration, and (13 Nov) label-blind clipping (`blindclip_gate.py`), conditional calibration (`conditional_gate.py`, a failed prediction), automatic clipping (`auto_gate.py`, superseded); (14 Nov) the fair rerun `auto_gate_fair.py` + `PREREG_AUTO_FAIR.md` that killed blindA, and the multi-row unit counterexample `group_condition_gate.py`; reward-blind GRPO (`reward_blind_*.py`, `PREREG_REWARD_BLIND.md`) |

Privacy unit = context: a triage ticket's three questions, or several BoolQ questions on one passage, are one
person's record. On eve-rlcd's private slice that turns 32,000 rows into 29,380 units.

## Install
```
pip install torch transformers peft dp-accounting scipy numpy pytest
export PYTHONPATH=$PWD
python -m pytest -q tests          # 44 tests, about 7 minutes on CPU
bash scripts/dry_run_cpu.sh        # whole plan on CPU, about 30 minutes
```

## Data
```
mkdir -p data && for f in train val test; do
  curl -L -o data/$f.jsonl https://github.com/anthony-maio/eve-rlcd/releases/download/data-v1/$f.jsonl; done
```
`run_grid.sh phase0` checks the md5s published in eve-rlcd's README.

## Privacy scope (fixed 14 Nov 2026 after review)
- Mechanism randomness (Poisson sampling, Gaussian noise, randomized response, the exponential mechanism)
  is OS entropy by default (`--randomness secure`). `run_grid.sh` is a simulation on public data and
  defaults to `RANDOMNESS=seeded` (per-run, per-stream seeds); those ledgers say `release_private: false`.
  Before this fix every mechanism was seeded from `--seed`, which is saved with the model, and RR rows
  sharing an id reused one stream: no run before 14 Nov was a private release.
- `train_log.jsonl` holds only release-safe values (noisy-update norms, metrics on public eval data).
  Unnoised statistics of the private slice, timing and data hashes go to `private_diagnostics/`
  (operator-only, not covered).
- Selecting among runs composes them: `privacy.composed_epsilon` composes each candidate's own
  (sigma, q, steps); `summarize_blind.py` reports it and refuses a finite number if any candidate's
  record is missing or non-private.
- Release allowlists (reaudit of e9190c3): `privacy.json` and `temperature.json` carry only the fields in
  `privacy.RELEASE_LEDGER_KEYS` / `RELEASE_TEMPERATURE_KEYS`. Unit and row counts, drops and label-DP flip
  statistics go to `private_diagnostics/`. Under add/remove adjacency the unit count is private, so a
  DP-SGD release needs `--n-units-public` (a declared public count that fixes q and the steps); without
  it `release_private` is false and `release_blockers` says why.
- Slicing (reaudit of e58939c): `--start/--limit` cut rows by file position, so adding one raw unit can
  replace several units of the slice. A private release must use `--partition SALT:LO:HI` (units kept by a
  salted hash of their own key, `schema.select_partition`) with no positional slicing; otherwise
  `release_blockers` says so. The ledgers publish the partition; `summarize_blind.py` combines training
  and calibration by parallel composition only when the records prove the partitions disjoint (same
  file, unit definition and salt, non-overlapping ranges), and sequentially otherwise. A calibrated release
  with a missing `temperature.json` is refused.
- `release_private: true` means only that this run's mechanics meet the stated adjacency. It is not an
  end-to-end statement about a release pipeline (candidate selection, calibration, audits).

## Run
**14 Nov 2026: blindA is killed before any GPU run.** With std + DP temperature scaling tuned after
calibration, the toy shows no gain from label-blind clipping (`gates/PREREG_AUTO_FAIR.md`); the 13 Nov
+3.3-point result came from an under-tuned baseline. `run_grid.sh blindA` refuses to run unless
`FORCE_BLINDA=1`. Nothing in this package currently needs a GPU run.
The original plan below stays available as baseline machinery. See `PLAN.md` for its predictions, gates and kill criteria:
```
bash scripts/run_grid.sh phase0
bash scripts/run_grid.sh phase1                 # prints the DP-SGD sweep ranked by public-val Brier
B_STAR=512 EP_STAR=4 LR_STAR=1e-3 bash scripts/run_grid.sh phase1b   # use the sweep's winner
B_STAR=... bash scripts/run_grid.sh phase2
B_STAR=... bash scripts/run_grid.sh phase3
python scripts/summarize.py --runs runs --md runs/summary.md
```

## Privacy accounting, in one place
- DP-SGD: Poisson sampling of units at rate q = B/N for T = ceil(epochs/q) steps, per-unit clip C, noise
  N(0, sigma^2 C^2) on the sum, update divided by B (not the realized batch). Adjacency: add/remove one unit.
  sigma is calibrated with PLD (default) and the ledger reports epsilon under PLD and RDP.
- Label DP: randomized response on each label once, before training; everything after is post-processing.
  Pure epsilon, delta = 0, adjacency: change the labels of one unit.
- DP temperature: exponential mechanism, pure epsilon, on a slice that shares no context with the training
  slice, so the total is max(eps_train, eps_cal) by parallel composition.
- Not covered: anything a person does by hand with the private slice (looking at it, tuning on it). The run
  plan tunes only on the public val split, and probes clip norms on it.
- Research code: the Gaussian noise uses torch's PRNG (not cryptographically secure) and floating-point
  arithmetic; fine for experiments, not for deployment.
