# D1 plan: private calibrated decision models (pre-registration, 1 Nov 2026)

Question: what does it cost to train a Jev-style typed decision model (eve-rlcd on Qwen3-0.6B-Base) under
differential privacy *without losing calibration*, which is the property these models are sold on?

## Setting
- **Data:** eve-rlcd data-v1 (md5-checked).
  - Public warmup: train rows 0..6399.
  - Private slice: rows 32000..63999, which is 32,000 rows in **29,380 privacy units**. A unit is a context: a triage ticket's three questions, or the BoolQ questions on one passage.
  - At most 4 rows per unit; 39 rows are dropped by that cap.
  - DP calibration slice: rows 24000..31999. 1,132 of its rows share a context with the private slice and are excluded automatically.
  - Canary pool: rows 6400..23999, minus the contexts of the private, calibration, val and test sets.
  - Test: the 8,000-row held-out split. Only 2 test rows share a context with the private slice.
- **DP-SGD.** Poisson sampling of units with add/remove-one-unit adjacency at δ = 1e-6; per-unit clipping; Gaussian noise; PLD calibration (RDP also reported). The update is divided by the **expected** batch size.
- **Label DP.** k-ary randomized response is applied once per label, splitting ε equally over a unit's rows. This gives pure ε label-DP and needs no accountant. The objectives are corrected exactly:
  - RLCD and RLVR: the debiased outcome.
  - Brier: the debiased target.
  - CE: forward correction.
- **DP temperature scaling.** An exponential mechanism over 61 temperatures, scored by summed Brier (sensitivity 2 × 4 rows). It runs on the disjoint calibration slice, so by parallel composition the end-to-end guarantee is max(ε_train, ε_cal) = ε.
- **Noise on this slice** (PLD, δ = 1e-6):

  | Batch B | Epochs | Steps T | σ at ε=1 | ε=4 | ε=8 |
  |---|---|---|---|---|---|
  | 128 | 2 | 460 | 0.886 | 0.603 | 0.483 |
  | 512 | 4 | 230 | 1.434 | 0.774 | 0.604 |
  | 1024 | 8 | 230 | 2.460 | 1.006 | 0.744 |

- **Label-DP cost on this slice.** The data mix matters: Banking77 and Bitext have up to 26 options, and triage units split ε three ways.

  | ε | Flip rate | Median variance inflation | Mean variance inflation | Triage | Bitext |
  |---|---|---|---|---|---|
  | 1 | 0.58 | 15 | 61 | 51 | 106 |
  | 2 | 0.38 | 3.2 | 10 | 11 | 12 |
  | 4 | 0.12 | 1.2 | 2.1 | 2.6 | 1.7 |
  | 8 | 0.015 | 1.0 | 1.1 | n/a | n/a |

  Label DP at ε=1 is not worth running; the grid uses ε ∈ {2, 4, 8}.

## Zero-GPU gate already run (gates/h1_gate.py, toy MLP, 32k private rows, 3 seeds)
H1 was "DP-SGD on the RLCD objective ends better calibrated than DP-SGD on CE". **It failed the gate.**

- **Clipping alone, without noise, makes every objective overconfident.** Test ECE:

  | Objective | Non-private | Clipped at C = median norm | Clipped at C = 0.1 × median |
  |---|---|---|---|
  | CE | 0.008 | 0.099 | 0.132 |
  | Brier | 0.013 | 0.057 | 0.089 |
  | RLCD | 0.013 | 0.086 | 0.066 |

- **Under DP-SGD at ε=4, RLCD is no better than CE.**
  - C = median norm: RLCD 0.098 vs CE 0.062.
  - C = 0.1 × median: RLCD 0.101 vs CE 0.098.
  - Full-label Brier is better than CE at the median clip (0.053) but worse at 0.1 × median (0.120): not robust.
- **DP temperature scaling repairs every arm.** ECE lands at 0.018–0.025 and Brier recovers partly. Resolution stays below non-private (0.022–0.023 vs 0.025–0.029).
- **Label DP with the corrected objective at ε=4 matches non-private:**

  | Arm | Accuracy | ECE | Brier |
  |---|---|---|---|
  | Label-DP RLCD | 0.632 | 0.017 | 0.500 |
  | Label-DP CE (forward correction) | 0.632 | 0.007 | 0.498 |
  | Non-private, for reference | 0.632–0.633 | 0.008–0.013 | 0.494–0.500 |

  At ε=1 it falls behind DP-SGD + TS (Brier 0.515–0.517 vs 0.506–0.507), because 53% of labels flip.

## Pre-registered predictions for the GPU study (checked by scripts/summarize.py)
- **P1.** Every DP-SGD arm at ε=4 is overconfident before recalibration: ECE ≥ 3× its non-private arm and overconfidence > +0.03.
- **P2.** DP temperature scaling at the same ε brings every DP-SGD arm within 0.02 ECE of its non-private arm.
- **P3 (H1 is predicted to fail).** No objective effect under DP-SGD after recalibration: |ΔECE(RLCD − CE)| < 0.015 and |ΔBrier| < 0.01.
- **P4a.** Corrected label DP at ε=4 is within 1.5 accuracy points and 0.015 ECE of non-private.
- **P4b.** Raw randomized response at ε=2 is underconfident (overconfidence < −0.05); the corrected arm is not.
- **P5.** Every private arm has lower resolution than its non-private arm, even after recalibration.
- **P6a.** The mislabel-canary one-run audit detects the non-private run (ε_lb > 1). Otherwise the audits are uninformative and P6b is void.
- **P6b.** No DP run's audit lower bound exceeds its ε. A violation means a bug: stop and debug before anything else.
- **P7.** With public inputs, corrected label DP beats DP-SGD + TS in Brier at ε=4.

## Gates and kill criteria
- **After phase0.**
  - The warmup must land near eve-rlcd's (accuracy 0.746–0.750, ECE about 0.02). If it is off by more than 0.02 accuracy, stop: the substrate is broken.
  - Record the median per-unit gradient norms. A probe median under 1e-3 means the readout is degenerate; check before continuing.
- **After phase1.**
  - The non-private LoRA RLCD must keep the warmup's calibration (ECE within 0.02 of the warmup), as eve-rlcd's full fine-tune does.
  - The full-FT n-rlcd-full must reproduce 0.808 / 0.023 / 0.267 within about 0.01. If not, the reproduction failed and every comparison is suspect.
  - From the sweep, choose B_STAR, EP_STAR and LR_STAR by final public-val Brier.
- **After phase1b.**
  - If P1 fails (DP-SGD is not overconfident at LLM scale), the toy mechanism did not transfer. That is a finding in itself; record it and continue.
  - If P2 fails, the paper is about why recalibration does not suffice. Run per-primitive DP temperatures (choice / score / noul) as the follow-up, splitting ε by parallel composition only if the slices are disjoint by context.
- **Phase 3.**
  - P6b is a hard stop.
  - If P6a fails, strengthen the canaries before drawing any privacy conclusion: repeat each canary r times and account with group privacy, or use more epochs.

## Caveats stated up front
- **Hyperparameter selection** uses only public data, so it spends no privacy. B, lr and epochs are chosen on the public val split. The clip is the median per-unit gradient norm from a 20-step probe on the public val split (phase0), never on private rows.
- **The audit uses the pure-DP form** of Steinke et al. The δ term is negligible when m·δ = 1e-3 ≪ β = 0.05.
- **Everything is in-distribution** on eve-rlcd's sources. The "private" data are public datasets standing in for private ones.

## Budget (rough; one 16 GB GPU)
| Phase | Runs | GPU-hours |
|---|---|---|
| 0 | warmup and probes | ~0.5 |
| 1 | 3 LoRA references, 1 full-FT reference, 6 sweep runs | ~12 |
| 1b | 3 DP-SGD, 2 clip-only, 2 label-DP | ~10 |
| 2 | 6 DP-SGD, 4 label-DP, 1 raw control, 10 seed reruns | ~25 |
| 3 | 5 canary runs and scoring, 5 membership attacks | ~8 |
| **Total** | | **~55** |

DP-SGD runs cost about 2× non-private because of the per-unit backward loop. For full fine-tuning on a 16 GB card, use LoRA (the default here). The per-unit accumulator adds 2.4 GB on a full 0.6B model.
