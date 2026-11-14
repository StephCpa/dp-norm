# Label-blind clipping for DP training of calibrated decision models (13–14 Nov 2026)

> **Status, 15 Nov 2026, after four external reviews: label-blind clipping has no demonstrated practical advantage over a fair, well-ranged calibration baseline.**
> - **Retracted (14 Nov):** the 13 Nov automatic-clipping advantage in the K = 4 toy (std's lr was chosen before temperature scaling).
> - **Retracted (15 Nov):** the 14 Nov "open, small" advantage of 0.001–0.005 Brier in the binary calibration-quota screen. The standard baseline's calibrators were capped (T ≤ 4, Platt a ≥ 1/4) and every public TS fit sat on T = 4. With the caps widened and nothing else changed, the advantage vanishes or reverses (§4).
> - **Kept:** the stationarity mechanism and its conditions (§2); the reward-blind analysis with narrow conclusions (§7).
> - **Confirmed negative (15 Nov), pre-registered in b0f20ac before the run.** With wide calibrator ranges, full-pipeline tuning on dev seeds and 10 fresh seeds, blind is *worse* than the deployable standard pipeline at the primary cells (N = 36000, ε = 4): +0.00128 [+0.00095, +0.00161] Brier well-specified, +0.00238 [+0.00165, +0.00310] misspecified. The closure rule is met. Calibrating with N/32 or more of the budget costs at most a few 1e-4 Brier, less than blind loses to the worst-case-label bound (§6).
> - **Decision: no GPU.** This line closes as a mechanism note.
> - **Release protocol:** fixed in four rounds (§8). `release_private: true` covers one run's mechanics under the stated adjacency, not a full selection-and-calibration pipeline. No run so far is a private release.

Code: github.com/StephCpa/dp-norm. Pre-registrations and outputs in `gates/`.

## 1. D1 against the literature
| D1 claim (1 Nov) | Status | Where |
|---|---|---|
| DP-SGD miscalibrates; per-example clipping is the cause | **Published** | Zhang et al., arXiv 2210.08248; Bu et al., TMLR 2023 |
| DP temperature on a disjoint split repairs reliability | **Published, with the split-and-compose accounting** | Zhang et al. (DP-TS, DP-PS) |
| Corrected label DP with public inputs | **Standard** | Ghazi et al. 2021; Chowdhury, Zhou & Natarajan 2024 |
| DP-RLCD with the prompt group as the record | **Published construction** | Zhao, Januszewicz & Jung, arXiv 2610.07212 |

The 55-GPU-hour D1 grid would mostly re-measure Zhang et al. Not run.

## 2. The mechanism and what the theory says
- **Clipping as reweighting.** DP-SGD scales a unit by s = φ(‖g(x, y)‖), which depends on the label. For fixed x the fixed point minimizes the proper score under P̃(y|x) ∝ P(y|x)·s(x, y): overconfidence (Menon et al. 2020: clipping is Huberisation).
- **Label-blind clipping.** s = φ(B), B = Σ_rows max_{y'} ‖g_row(x, y')‖. The realized ‖g‖ ≤ B, so sensitivity and accounting are unchanged.
- **Prop 1 (necessity, under a rank condition).** Fix x, model probabilities p with every p_y > 0, and the parameter Jacobian J = ∂z/∂θ ∈ ℝ^{k×d}. Let Π = I − 11ᵀ/k (k × k) project logit space onto zero-sum vectors.
  - Condition (R): rank(ΠJ) = k − 1, equivalently v ↦ Jᵀv is injective on zero-sum v.
  - Under (R), for CE, E_{y~P}[s(y) Jᵀ(p − e_y)] at p = P vanishes iff s is constant on the support of P. Same for Brier, since diag(p) − ppᵀ is injective on zero-sum vectors when p > 0.
  - Without (R) the claim fails: in a constrained model that can move only some logit contrasts, a label-dependent s whose effect lies in the kernel of Jᵀ keeps the parameter gradient at zero. Label-blindness is then sufficient, not necessary.
- **Prop 2 (maximality).** A label-independent factor must satisfy ‖s·g(x, y)‖ ≤ C for every possible y, so s ≤ C / max_y ‖g(x, y)‖. Exact-blind is the largest stationarity-preserving scalar per-example factor. This is optimality within that class only.
- **Prop 3 (multi-row units).** A unit factor needs E[y_r | X_unit, A_unit] = p(x_r) for every row. Counterexample (`gates/group_condition_gate.py`): the fixed point moves from 0.500 to 0.592 (standard clipping: 0.716). A per-row budget (`--blind-budget row`) keeps 0.500.
- **Limits.** Zero expected gradient at the truth does not make the trained model calibrated (misspecification, finite samples, noise, optimizer). These are statements of claims; the bounds are not written up as a manuscript.
- **Cost.** k − 1 backward passes per row (contrast basis); one for binary.

## 3. K = 4 toy (MLP, 32k private rows plus a separate 4k calibration set)
- Clipping alone, no noise: blind removes clipping-induced miscalibration (CE ECE 0.099 → 0.005).
- Threshold clipping with noise: blind ≈ std + DP-TS.
- Automatic clipping, fair rerun (`auto_gate_fair.py`): at ε = 4, CE accuracy +0.06 points [−0.03, +0.15], CE Brier +0.0018 [+0.0006, +0.0030] (std ahead); RLCD Brier −0.0006 [−0.0028, +0.0017]. **The 13 Nov advantage is retracted.**
- That comparison gave std an extra calibration set. §4 removes that.

## 4. Calibration quota under a fixed private budget (binary toy, `calquota_screen.py`, `PREREG_CALQUOTA.md`)
**Question.** With N private units and ε fixed, train blind on all N, or train standard on N − m and spend m on a private calibrator (DP-TS or DP-Platt)? Δ(m, ε) = Brier_blind(N) − Brier_std+DPcal(N − m, m). CE, automatic clipping, same γ for both, 5 data-and-training seeds, two model setups (well-specified, misspecified).

**Pre-registered result (tuning on one seed), N = 36000, blind minus the best private baseline:**
| setup | ε = 1 | ε = 4 | ε = 8 |
|---|---|---|---|
| well-specified | −0.0012 [−0.0018, −0.0005] | −0.0028 [−0.0036, −0.0020] | −0.0009 [−0.0018, −0.0001] |
| misspecified | −0.0045 [−0.0066, −0.0024] | +0.0023 [−0.0016, +0.0061] (acc −0.87 pts) | +0.0009 [−0.0027, +0.0045] |
- Decision rule S1 (≥ 0.003 better than the best private baseline in both setups at one ε): **not met**. Kill rule (fails everywhere): not met either.
- Against std + public TS (non-private TS on public validation labels): blind is better in 4 of 12 cells.
- Std raw is 0.056–0.059 Brier worse than blind: the miscalibration is real; the question is only what fixing it costs.

**Exploratory (not pre-registered):**
- 3-seed tuning removes the misspecified ε = 4/8 losses (they came with blind picking lr 3e-2 on one seed): N = 36000, ε = 4: well −0.0031 [−0.0042, −0.0021], mis −0.0038 [−0.0057, −0.0018].
- The best baseline sat at the largest pre-registered quota (m = 4096). Larger quotas at ε = 4: well-specified shrinks to −0.0014 [−0.0026, −0.0002] at m = 16384; misspecified stays at −0.0035 to −0.0046.

**Retracted reading (15 Nov).** The 14 Nov reading ("training blind on everything beats spending private data on calibration by 0.001–0.005 Brier") does not hold. Both standard calibrators were capped: TS at T ≤ 4 and Platt at a ≥ 1/4, so neither could shrink the margin below a quarter. Every public TS fit in the screen sat on T = 4 (all 5 seeds, all 6 N = 36000 settings); unconstrained public fits are about T = 4.6–6. With TS widened to T ≤ 16 and Platt to a ≥ 1/16 (same candidate counts, same ε, same learning rates as the pre-registered run), blind minus the best private baseline becomes:

| setup | ε = 1 | ε = 4 | ε = 8 |
|---|---|---|---|
| well-specified | +0.00084 [+0.00016, +0.00151] | +0.00101 [+0.00085, +0.00117] | −0.00100 [−0.00187, −0.00014] |
| misspecified | +0.00146 [+0.00086, +0.00206] | +0.00813 [+0.00571, +0.01056] | +0.00670 [+0.00446, +0.00894] |

(positive = blind worse; reproduced to five decimals from the reaudit of df9bb65; exploratory, on the screen's own seeds.) The misspecified ε = 1 cell, the only one that cleared the 0.003 bar, reverses with the learning rate unchanged.

**What the screen could not say even before this.**
- Std's lr was tuned for N with public TS and reused for every N − m pipeline.
- Changing m changes the training set, q, steps, σ and the margins, as well as the calibration set, so Δ(m) compares two whole pipelines; it is not the causal cost of calibration rows. The confirmation decomposes it on the same trained model: privacy of calibration, a finite calibration set, and fewer training rows.
- The "best private baseline" was picked by test means: an oracle reference, not a deployable choice.
- The pre-registration and its results were in the same commit, so their order is not verifiable from git.
- The JSON kept means and contrasts only. The scripts now keep per-seed metrics, calibrator parameters and edge hits.

## 5. Prior art
- Nearest: DP-ERM with Lipschitz losses; Clipless DP-SGD (Béthune et al.); Menon et al.; DiceSGD, BAM; Bu et al.; adaptive clipping; DPSGD-F.
- No paper found with a per-example worst case over labels or a fixed-point analysis of proper scoring rules under clipping. Not a novelty claim.

## 6. CPU confirmation (pre-registered in b0f20ac, run after it was pushed): closure
| cell | D = blind − deployable std | accuracy | deployable std |
|---|---|---|---|
| well, N = 36000, ε = 4 | +0.00128 [+0.00095, +0.00161] | −0.08 pts | m = 4500, TS |
| mis, N = 36000, ε = 4 | +0.00238 [+0.00165, +0.00310] | −0.07 pts | m = 4500, TS |
- No calibrator hit a range edge in any cell.
- Secondary (no decisions): blind is worse at N = 36000 for ε = 1 and 8 too (+0.0011 to +0.0025, intervals above 0). At N = 8000 every interval crosses 0. Where D is negative there, the dev seeds picked a very small quota (62 or 250 rows) with a noisy DP temperature. Blind is worse than the oracle-best quota and than std + public TS in all 12 cells.
- Decomposition at the primary cells:
  - From m = N/32 on, privacy of calibration costs at most 2e-4 Brier, the finite calibration set about 0, and fewer training rows about 0 up to N/4.
  - Blind's loss from the worst-case-label bound (about 1.3e-3 / 2.7e-3 Brier against std at m = N/32) is larger than everything calibration costs.
  - Within this toy, the trade the plan asked about goes to standard clipping plus a small private calibrator.
- Outputs: `gates/calquota_confirm_results.json` (per-seed metrics, calibrator parameters, edge hits, steps, q) and `calquota_confirm_log.txt`.

### Design of the confirmation (as pre-registered)
- `blindA` refuses to run without `FORCE_BLINDA=1`.
- `gates/calquota_confirm.py` with `gates/PREREG_CALQUOTA_CONFIRM.md`:
  - frozen wide ranges (T ∈ [1/32, 32]; Platt a ∈ [1/32, 4], b ∈ [−3, 3]) with edge-hit checks;
  - every pipeline tuned on dev seeds 201–203; a deployable std pipeline chosen on dev data;
  - quotas up to N/2; fresh evaluation seeds 101–110;
  - the decomposition above.
- Primary decision at N = 36000, ε = 4, both setups:
  - "blind better" needs D ≤ −0.003 with the upper limit < 0;
  - "closure" needs D's lower limit > −0.003.
- Predicted: closure. Closure → a mechanism note, no GPU.
- The GPU calibrator in `dpdecide/calibrate.py` had the same [0.25, 4] grid; it is now [1/16, 16] with an `at_grid_edge` flag.

## 7. Reward-blind clipping for DP-GRPO: no advantage observed, scaling suspended
- Worst case is the balanced split, not K − 1 : 1 (reproduced).
- Blind clipping restores the unclipped (Dr. GRPO) weighting E[S] ∝ p(1−p); standard clipping gives E[√S], whose large-K limit is the arcsine objective.
- Automatic clipping erases any positive per-group advantage rescaling.
- Linearized training toy, ε = 4: standard 0.830 J vs blind 0.730. Supported only under the tested score-noise models; "no fixed point to protect" is retracted (with shared parameters the weighting moves the stationary points).

## 8. Reviews and fixes
**Review 1 (13 Nov package).** Selection over candidates composes; mechanism randomness was recomputable from the saved seed (now OS entropy, seeded = simulation); private statistics split into `private_diagnostics/`; Prop 3; fair rerun (decisive); same C and γ; RR duplicate ids; contrast basis; Props 1–2; reward-blind wording.

**Reaudit of e9190c3.** Cardinalities and RR flip statistics removed from released files (allowlists); DP-SGD schedule from a declared public count (`--n-units-public`); candidates composed per candidate (`composed_epsilon`); refusal for missing or non-private records; Prop 1 rank condition; point-estimate wording.

**Reaudit of e58939c.**
| Finding | Verdict | Fix |
|---|---|---|
| A missing `temperature.json` was skipped (8.00 → 4.00) | Holds | `budget(cands, calibrated=...)`: a declared calibrated release refuses if any candidate lacks a private temperature; an uncalibrated release is declared explicitly |
| Max (parallel composition) used without a disjointness proof | Holds | ledgers publish the partition; parallel only when both are hash partitions of the same file, unit definition and salt with non-overlapping ranges (`schema.partitions_disjoint`); otherwise the budgets add up |
| Positional row slicing breaks raw-file adjacency (one added 4-row unit replaces four) | Holds | `--partition SALT:LO:HI` keeps units by a salted hash of their own key; positional slicing or context exclusion is a release blocker; test: inserting one unit changes a hash partition by at most that unit, a positional slice by several |
| Rank condition written as JΠ | Holds | rank(ΠJ) = k − 1 |
| Doc not in the repository | Holds | this file |

**Reaudit of df9bb65.**
| Finding | Verdict | Fix |
|---|---|---|
| Both standard calibrators capped; the screen's advantage depends on the cap | Holds, reproduced | §4 retraction; wide ranges, edge flags and per-seed records in the screen; GPU calibrator grid widened with `at_grid_edge`; confirmation pre-registered |
| P0: in `--unit row` without ids, the fallback key hashed the whole row including the answer, so hash-partition membership (decided before RR) depended on the protected label | Holds | the fallback key hashes the inputs only; test: label-only neighbours get the same key and the same membership at every cut |
| P1: only train_i vs cal_i was checked, not train_i vs cal_j | Holds | parallel composition only when every training partition is disjoint from every calibration partition; the review's counterexample is a test |
| P1: blind chosen on calibrated validation metrics while its budget was uncalibrated | Holds | blind is chosen on raw validation metrics (`eval_val`, now produced by `run_grid.sh`); its calibrated metrics are not shown; B1/B2 compare blind at T = 1 with std + DP-TS |
| Ledger consistency not cross-checked | Holds | `budget` refuses a record whose `release_private`, `randomness` and `release_blockers` disagree or are missing |
| With partitions set, `run_grid.sh` still passed `--exclude-data` to calibration | Holds | exclusion is used only for positional slices |

Checks: 48 tests pass; blind CPU dry run end to end.

## Sources
- [Zhang et al., arXiv 2210.08248](https://arxiv.org/abs/2210.08248); [Bu et al., TMLR 2023](https://arxiv.org/abs/2106.07830); [Bu et al., Automatic Clipping](https://arxiv.org/abs/2206.07136)
- [Menon et al., ICLR 2020](https://research.google/pubs/can-gradient-clipping-mitigate-label-noise/); [Béthune et al.](https://arxiv.org/html/2305.16202v2); [DiceSGD](https://arxiv.org/abs/2311.14632)
- [Zhao, Januszewicz & Jung, arXiv 2610.07212](https://arxiv.org/html/2610.07212); [Chowdhury et al.](https://arxiv.org/abs/2310.19733); [Ghazi et al.](https://arxiv.org/abs/2102.06062)
- [Liu & Talwar](https://arxiv.org/abs/1811.07971); [Papernot & Steinke](https://arxiv.org/abs/2110.03620)
- [Liu et al. (Dr. GRPO)](https://arxiv.org/abs/2503.20783); [Bay & Yearick](https://www.emergentmind.com/papers/2607.00152); [Ge et al.](https://arxiv.org/abs/2601.23135)
