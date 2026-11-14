# Label-blind clipping: pre-registration (13 Nov 2026)

> **Status 14 Nov 2026: killed before any GPU run.** The toy evidence behind B1 compared blind clipping
> against std + DP-TS with std's learning rate chosen by validation Brier BEFORE temperature scaling, so
> std picked the smallest lr in the grid. Tuned after calibration, std + DP-TS ties blind at eps = 4
> (CE +0.06 acc points [-0.03, +0.15], Brier +0.0018 in std's favor; RLCD +0.05 points, Brier -0.0006
> [-0.0028, +0.0017]); see `gates/PREREG_AUTO_FAIR.md`. Also revised after review: the multi-row unit
> condition (`gates/group_condition_gate.py`), same C and gamma for both arms, k - 1 backward passes
> (one for binary), and the selection-inclusive epsilon. The text below is the original plan.

Replaces the headline of PLAN.md. The prior-art check of 13 Nov found D1's first, second and fourth claims already published; see `claude/dp-blind-clipping.md` in the project.

## Claim
Standard DP-SGD clips each privacy unit by a factor that depends on the unit's labels, because surprising labels have large gradients. That turns a proper scoring objective (CE, Brier, RLCD) into an improper one. Instead, clip with the factor

    B = sum over rows of max over all labels y' of ||g_row(x, y')||
    factor = min(1, C/B)          threshold clipping
    factor = C/(B + gamma)        automatic clipping

The factor depends on the inputs, the model and (for RLCD) the model's own sampled actions, never on the labels.
- **Sensitivity is still C:** the realized gradient is bounded by B. Accounting is unchanged.
- **The calibrated predictor stays a fixed point:** at p = P(·|x), every logit-gradient and every RLCD reward c − π(a) has zero conditional mean.
- **Binary decisions, exactly:** under per-example normalization (automatic clipping), standard CE and RLCD drive the prediction to the majority label. The blind version keeps it at P.

## Zero-GPU evidence (gates/blindclip_gate.py, conditional_gate.py, auto_gate.py; toy MLP, 32k private rows, 3 seeds)
- **Clipping without noise:**

  | Objective | Std clip, C = median | Std clip, C = 0.1 × median | Blind clip, C = median | Blind clip, C = 0.1 × median | Non-private |
  |---|---|---|---|---|---|
  | CE | 0.099 | 0.132 | 0.005 | 0.006 | 0.008 |
  | Brier | 0.057 | 0.089 | 0.016 | 0.018 | 0.013 |
  | RLCD | 0.086 | 0.066 | 0.021 | 0.024 | 0.013 |

  - CE accuracy at the small clip: blind 0.633, std 0.606.
- **Threshold clipping with DP noise, ε=4, C = median:** blind and std + DP temperature scaling (DP-TS) are equal on Brier. CE: 0.506 vs 0.511. RLCD: 0.507 vs 0.508. **No gain at a tuned threshold.**
- **Calibration by input-norm quartile:** std + DP-TS and blind have the same profile, set by the student's misspecification. **No gain in conditional calibration.**
- **Automatic clipping with DP noise, ε=4** (std's learning rate checked down to 3e-5):

  | Objective | Accuracy, std (+TS) | Accuracy, blind | Brier, std + TS | Brier, blind (T=1) |
  |---|---|---|---|---|
  | CE | 0.597 | 0.630 | 0.536 | 0.506 |
  | RLCD | 0.612 | 0.629 | 0.522 | 0.507 |

  - Temperature scaling cannot recover std's loss.

## Tier A GPU test (`run_grid.sh blindA`)
- **Data.**
  - Binary rows of the private slice: 5,336 rows in 4,804 units (BoolQ 4,011, triage paging 1,325).
  - Test: 1,333 binary rows. Calibration: 1,315 binary rows (context-disjoint).
  - On binary rows blind clipping needs 2 logit gradients per row.
- **Arms** (CE and RLCD, ε=4):
  - Threshold clipping at C = median (probed on public val) and automatic clipping (γ = 0.01 × median).
  - Each clipping regime with std and blind clipping.
  - lr ∈ {2e-4, 1e-3}, chosen by validation Brier after each arm's own DP temperature, so std gets the full Zhang et al. pipeline.
  - Expected batch 128, 4 epochs (151 steps).
  - 16 DP runs plus 2 non-private references; about 10 GPU-hours on one 16 GB card.
- **Predictions** (checked by `scripts/summarize_blind.py`):
  - **B1.** Automatic clipping: blind beats std + DP-TS by at least 1.0 accuracy point and at least 0.005 Brier, for both CE and RLCD. (Toy: +3.3 points, −0.030.)
  - **B2.** Threshold clipping at C = median: blind and std + DP-TS within 0.005 Brier. No gain is claimed here.
  - **B3.** At T = 1, with no temperature, blind is within 0.02 ECE of non-private; std is at least 2× non-private ECE.
  - **B4.** Cost on binary rows: blind wall-clock at most 2.5× std.
- **Kill criterion.** If B1 fails, the mechanism does not transfer to a 0.6B LoRA model where it would matter. Write it up as a toy-level note and stop; do not run tier B.
- **Tier B, only if B1 holds.** Rows with k ≤ 5: 24,666 rows, at most 5 logit gradients per row. Use the same arms plus 2 more seeds, then the canary audit (phase 3) on the blind arm.

## Caveats stated up front
- **Cost scales with option count.** A k-option row needs k backward passes and k trainable-size vectors, so LoRA is required beyond small k. For RLCD only the sampled actions and the prediction matter (at most G + 1 vectors); that optimization is not implemented yet.
- **Units with several rows** use the triangle bound across rows. It is valid but conservative, so it clips more than needed.
- **Novelty is not established.** Nearest prior work:
  - Classical DP-ERM: bounded features and Lipschitz losses, label-free by construction.
  - Clipless DP-SGD (Béthune et al. 2023): a Lipschitz architecture.
  - Menon et al. 2020: clipping is Huberisation.
  - DiceSGD (ICLR 2024) and bias-aware minimisation (BAM): clipping-bias correction.
  - Global and automatic clipping (Bu et al. 2023).
  - The search of 13 Nov found no paper that uses a per-example worst case over labels.
