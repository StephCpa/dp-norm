# Calibration quota under a fixed private budget: CPU screen (pre-registration, 14 Nov 2026, before any run)

Question. With N private units and one ε fixed, is it better to train label-blind on all N, or to train
standard on N − m and spend m units on a private calibrator? Where does each win?

Primary contrast (never gives std an extra calibration set):
  Δ(m, ε) = Brier_blind(N, ε) − Brier_std+DPcal(N − m train, m calibrate, ε),   m ∈ {64, 256, 1024, 4096}
Training and calibration use disjoint parts of the same N, so the release is max(ε_train, ε_cal) = ε.

Setup (gates/calquota_screen.py; the existing toy, made binary to match GPU stage 1):
- K = 2, CE objective, automatic clipping (γ = 0.01 × median standard norm on public rows, same γ for both).
- Two model setups: well-specified (teacher without the ambiguity gate; the student can represent it)
  and misspecified (the toy's gated teacher).
- N ∈ {36000 (primary), 8000 (secondary)}; ε ∈ {1, 4, 8}; 5 independent data-and-training seeds (1-5).
- Baselines: std(N) raw; std(N) + public TS (argmin on the public validation set); std(N − m) + DP-TS(m);
  std(N − m) + DP-Platt(m) (exponential mechanism over a 21 × 13 (a, b) grid, Brier utility, sensitivity 2);
  blind(N) raw.
- Tuning on data seed 0 only, lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}: std by public-validation Brier after public TS,
  blind by raw public-validation Brier. Frozen per (setup, N, ε) before seeds 1-5. 96 tuning runs, reported.
- Primary metric test Brier; accuracy is a constraint. Paired over seeds, 95% t-interval (df 4).

Decision rules (thresholds proposed in the plan, fixed here):
- S1 (private-only regime, primary): blind beats the BEST private baseline (min over m and calibrator,
  chosen post hoc, which favours std) by ≥ 0.003 Brier with the interval excluding 0, accuracy no more
  than 0.5 points lower, in both model setups, at some ε. Pass → GPU stage 1 is worth running.
- S1-weak (where the advantage emerges): the set of m where Δ ≤ −0.003 with the interval below 0. Reported
  as a curve; on its own it does not justify GPU work.
- S2 (public labels available): blind vs std + public TS, same thresholds. Reported; if public in-domain
  labels exist this is the baseline that matters.
- Kill: S1 fails in both setups at every ε (N = 36000) → no GPU investment; the line becomes a short
  mechanism note. N = 8000 can only qualify a pass, not rescue a fail.

Predictions:
P1. Blind beats std(N) raw by a wide margin in both setups (clipping-induced miscalibration is real).
P2. Δ(m, ε) < 0 at m = 64 (a 64-row exponential mechanism picks a noisy temperature) and Δ ≥ 0 by
    m = 1024 at ε ∈ {4, 8}: S1-weak holds only at tiny m.
P3. S1 fails at N = 36000 for ε ∈ {4, 8}. At ε = 1 I do not know.
P4. Std + public TS beats or ties blind everywhere (S2 fails).
P5. DP-Platt ≤ DP-TS in Brier at m ≥ 1024 (two parameters, binary), worse at m = 64.

## Outcome (14 Nov 2026)
Pre-registered analysis (tuning on data seed 0), N = 36000, blind minus the best private baseline
(test Brier, 5 seeds, 95% t-interval), accuracy difference:
| setup | ε = 1 | ε = 4 | ε = 8 |
|---|---|---|---|
| well-specified | −0.0012 [−0.0018, −0.0005] | −0.0028 [−0.0036, −0.0020] | −0.0009 [−0.0018, −0.0001] |
| misspecified | **−0.0045 [−0.0066, −0.0024]** | +0.0023 [−0.0016, +0.0061], acc −0.87 pts | +0.0009 [−0.0027, +0.0045], acc −0.71 pts |
- S1: passes only for misspecified at ε = 1; the well-specified setup never reaches −0.003. **Not a pass**
  (the rule needs both setups at one ε). Not a kill either (the kill needs both to fail at every ε).
- S2 (vs std + public TS): passes in 4 of 12 cells (well ε = 4 at both N; mis ε = 1 N = 36000; mis ε = 4 N = 8000).
- P1 held (std raw is 0.056-0.059 Brier worse). P2 failed: blind beat DP-TS at m = 1024 in most cells, not
  only at m = 64. P3 held. P4 failed (see S2). P5 mixed.
- The misspecified ε = 4 and 8 losses come with blind picking lr 3e-2 on the single tuning seed; its own
  accuracy falls 0.8 points below ε = 1.

Exploratory, NOT pre-registered (labelled as such everywhere):
- Tuning on 3 seeds (TUNE_SEEDS=0,1,2): N = 36000, ε = 4: well −0.0031 [−0.0042, −0.0021], mis −0.0038
  [−0.0057, −0.0018]; ε = 8: well −0.0012, mis −0.0032; ε = 1: well −0.0020, mis −0.0038. S1 would pass at
  ε = 4 in both setups.
- The best private baseline sat at the largest pre-registered quota (m = 4096). Extending to m = 8192 and
  16384 at ε = 4 (3-seed tuning): well-specified shrinks to −0.0014 [−0.0026, −0.0002] at m = 16384 (S1
  fails again); misspecified stays at −0.0035 to −0.0046.
- Reading: in this binary toy, blind clipping is better than spending private data on calibration by
  0.001-0.005 Brier, at no accuracy cost, and the margin shrinks as std is allowed larger quotas in the
  well-specified setup. That is at or below the 0.003 practical threshold, depends on tuning noise, and
  differs from the K = 4 fair rerun (parity, std slightly ahead on CE Brier).
