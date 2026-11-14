# Label-blind clipping for DP training of calibrated decision models (13–14 Nov 2026)

> **Status, 14 Nov 2026, after three external reviews.**
> - **Retracted:** the 13 Nov automatic-clipping advantage (+3.3 accuracy points, Brier 0.536 → 0.506) in the K = 4 toy. Std's learning rate had been chosen before temperature scaling. Tuned after calibration, the two tie (§3). Reproduced independently.
> - **Killed before any GPU run:** the original tier A (`blindA`).
> - **Open, small:** in a binary toy with a fixed private budget, blind clipping on all N units beat standard clipping plus a private calibrator on part of N by 0.001–0.005 Brier at no accuracy cost (§4). The pre-registered screen did not pass its decision rule; exploratory analyses are near the 0.003 threshold. A GPU pilot is not justified by the screen as pre-registered.
> - **Release protocol:** fixed in three rounds (§8). `release_private: true` covers one run's mechanics under the stated adjacency. It is not a claim about a full selection-and-calibration pipeline. No run so far is a private release.

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

**Reading.** In this binary toy, training blind on everything beats spending private data on calibration by 0.001–0.005 Brier, with no accuracy cost, and more clearly under misspecification. The size is at or below the proposed 0.003 threshold, sensitive to tuning, and shrinks when std may spend a large share of N on calibration. The K = 4 toy (§3) showed parity. Whether this transfers to a language model is untested.

## 5. Prior art
- Nearest: DP-ERM with Lipschitz losses; Clipless DP-SGD (Béthune et al.); Menon et al.; DiceSGD, BAM; Bu et al.; adaptive clipping; DPSGD-F.
- No paper found with a per-example worst case over labels or a fixed-point analysis of proper scoring rules under clipping. Not a novelty claim.

## 6. GPU
- `blindA` refuses to run without `FORCE_BLINDA=1`.
- The two-stage plan (binary CE, Qwen3-0.6B LoRA, 2 tasks, fixed total N, then confirmation seeds) is the right design if the §4 effect is worth testing. The screen as pre-registered does not justify it. If run, the toy suggests: tune on several seeds, include m up to about half of N, and power the confirmation for an effect near 0.003 Brier.

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

Checks: 45 tests pass.

## Sources
- [Zhang et al., arXiv 2210.08248](https://arxiv.org/abs/2210.08248); [Bu et al., TMLR 2023](https://arxiv.org/abs/2106.07830); [Bu et al., Automatic Clipping](https://arxiv.org/abs/2206.07136)
- [Menon et al., ICLR 2020](https://research.google/pubs/can-gradient-clipping-mitigate-label-noise/); [Béthune et al.](https://arxiv.org/html/2305.16202v2); [DiceSGD](https://arxiv.org/abs/2311.14632)
- [Zhao, Januszewicz & Jung, arXiv 2610.07212](https://arxiv.org/html/2610.07212); [Chowdhury et al.](https://arxiv.org/abs/2310.19733); [Ghazi et al.](https://arxiv.org/abs/2102.06062)
- [Liu & Talwar](https://arxiv.org/abs/1811.07971); [Papernot & Steinke](https://arxiv.org/abs/2110.03620)
- [Liu et al. (Dr. GRPO)](https://arxiv.org/abs/2503.20783); [Bay & Yearick](https://www.emergentmind.com/papers/2607.00152); [Ge et al.](https://arxiv.org/abs/2601.23135)
