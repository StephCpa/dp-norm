# Calibration quota: confirmation with frozen wide calibrator ranges (pre-registration, 15 Nov 2026)

This file and `calquota_confirm.py` are committed and pushed BEFORE the script is run on the confirmation
seeds; the results go in a later commit. Before this commit the script was run once for plumbing only, on
unrelated seeds (dev 901, eval 902-903) at N = 2000, ε = 4, well-specified; nothing from that run is used.

## Why
The screen (`PREREG_CALQUOTA.md`, df9bb65) capped the calibrators: T ≤ 4 for TS and a ≥ 1/4 for Platt. Every
public TS fit landed on T = 4. With the caps widened, the screen's small blind advantage vanished or reversed
(exploratory, reproduced in the reaudit of df9bb65 and here). Its other limits: std's lr was tuned for N and
reused for N − m; the "best private baseline" was chosen by test means; quotas stopped at m = 4096; no
per-seed records; the pre-registration and results shared one commit.

## Design (frozen)
- Toy, training, automatic clipping (same γ for both), CE, K = 2: as in `calquota_screen.py`.
- Ranges: TS T ∈ [1/32, 32] (61 log-spaced); Platt a ∈ [1/32, 4] (31 log-spaced) × b ∈ [−3, 3] (25).
  DP calibrators: exponential mechanism, Brier utility, sensitivity 2, the same ε as training on a disjoint
  part of the same N (release ε unchanged).
- Quotas: m = N/128, N/32, N/8, N/4, N/2 (N = 36000: 281, 1125, 4500, 9000, 18000).
- Tuning on dev seeds 201-203 only, public validation only, lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}:
  - blind: raw validation Brier;
  - every std pipeline (m, calibrator ∈ {TS, Platt}, lr): validation Brier after its own DP calibrator;
  - the **deployable std pipeline** is the dev-best (m, calibrator, lr). It is the primary comparator;
  - std(N) + public TS: validation Brier after public TS (reference for when public labels exist).
- Evaluation: fresh seeds 101-110, test Brier and accuracy, paired 95% t-intervals (df 9). Per seed the JSON
  keeps all metrics, calibrator parameters, edge hits, steps and q.
- Decomposition per quota m, on the same trained model: privacy of calibration (DP vs non-private fit on
  the same m rows); finite calibration set (non-private m rows vs the large public validation set); fewer
  training rows (std(N − m) vs std(N), both with public TS).

## Decision (primary cells: N = 36000, ε = 4, both setups; 2 tests)
D = Brier(blind) − Brier(deployable std).
- **Blind better:** D ≤ −0.003 with the upper limit < 0, and accuracy no more than 0.5 points lower, in both
  primary cells.
- **No practically relevant advantage (closure):** lower limit > −0.003 in both primary cells.
- Otherwise inconclusive.
- **Range check:** if the public TS fit hits a T edge in any seed of a primary cell, that cell is not
  interpretable and the decision is "inconclusive (range)".
- The other 10 cells (N ∈ {36000, 8000} × ε ∈ {1, 4, 8} × setup, minus the primaries) and the comparisons
  to std + public TS and to the oracle-best quota are reported without decisions or multiplicity claims.

## Prediction
Closure in both primary cells: D's lower limit > −0.003, and D > 0 in at least one of them. The decomposition
should show that a quota of N/32 or more costs little through privacy of calibration.

## Consequence
Closure → label-blind clipping is written up as a mechanism note (stationarity under rank conditions, the
cost in signal, the reward-blind analysis, negative results); no GPU. Blind better → the two-stage GPU plan
becomes a candidate, sized for an effect near 0.003. Inconclusive → no GPU; report as is.

## Outcome (15 Nov 2026; run after b0f20ac was pushed)
Fresh seeds 101-110, all 12 cells completed, no calibrator edge hit in any cell (range check passed).
D = Brier(blind) − Brier(deployable std), positive = blind worse, paired 95% t-interval (df 9):

| cell | D | accuracy, blind − std | deployable std (chosen on dev) |
|---|---|---|---|
| **well, N = 36000, ε = 4** | **+0.00128 [+0.00095, +0.00161]** | −0.08 pts | m = 4500, TS, lr 1e-2 |
| **mis, N = 36000, ε = 4** | **+0.00238 [+0.00165, +0.00310]** | −0.07 pts | m = 4500, TS, lr 3e-2 |

- **Decision: closure.** Both lower limits are above −0.003 (they are above 0). Prediction held, including D > 0.
- Secondary, no decisions: N = 36000 at ε = 1 and 8: D between +0.00115 and +0.00247, every interval above 0.
  N = 8000: D from −0.00663 to +0.00008, every interval crossing 0. In the cells with negative D the dev seeds
  chose a very small quota (m = 62 or 250), whose DP temperature is noisy. Against the oracle-best quota blind
  is worse in all 12 cells (+0.00059 to +0.00271), and against std + public TS in all 12 (+0.00130 to +0.00316).
- Decomposition at the primary cells (same trained model; well / mis):
  - privacy of calibration: +0.00050 / +0.00090 at m = 281 (intervals cross 0); at m ≥ 1125 at most +0.00021;
  - finite calibration set: +0.00127 / +0.00080 at m = 281; about 0 from m = 1125 on;
  - fewer training rows: about 0 up to m = 9000; +0.00101 / +0.00114 at m = N/2.
  So from m = N/32 on, the whole cost of calibrating is at most a few 1e-4 Brier, smaller than what blind
  loses to the worst-case-label bound (blind − std at m = 1125: +0.00138 / +0.00271). The prediction that a
  quota of N/32 or more costs little through privacy of calibration held.
- Consequence (pre-registered): a mechanism note, no GPU.
