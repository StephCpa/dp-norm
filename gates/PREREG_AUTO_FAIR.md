# Fair rerun of the automatic-clipping toy gate (14 Nov 2026, written before the run)

Why: review finding P1-4/P1-5. auto_gate.py chose each arm's lr by validation Brier at T = 1, then
compared std at its DP temperature. That penalizes std's overconfidence during selection, and std
picked the smallest lr in the grid (3e-4) for every objective, so std + DP-TS was never tuned as a
pipeline. Gamma also differed between arms (0.01 x each arm's own public norm statistic).

Changes (auto_gate_fair.py; same toy, data, warmup, accountant, DP temperature as auto_gate.py):
- lr grid 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 (one step higher), chosen on seed 0, evaluated on seeds 1-3.
- std + DP-TS: lr chosen by validation Brier AFTER its DP temperature (temperature fitted on the
  private calibration slice, applied to validation).
- blind: lr chosen by validation Brier at T = 1 (its intended use, no calibration split);
  blind + DP-TS also reported, chosen like std + DP-TS.
- Same gamma for both arms: 0.01 x median of the STANDARD per-example norm on public rows.
- Paired effects over seeds 1-3 (same data, same warmup): mean and 95% t-interval (t = 4.30, df 2).

Predictions:
F1. std + DP-TS selects lr >= 1e-3 for at least 2 of 3 objectives at eps = 4.
F2 (the claim that justified blindA's B1). At eps = 4, blind (T = 1) beats std + DP-TS by >= 1.0
    accuracy point AND >= 0.005 test Brier, for both CE and RLCD. I do not know whether this holds;
    the old gap (+3.3 points, -0.031 Brier for CE) was measured against an under-tuned baseline.
F3. Kill rule for the GPU tier A: if F2 fails for both CE and RLCD, blindA's B1 has no toy support.
    Then the B1 arms are not run as a performance claim; what remains is the calibration-without-a-
    split result at T = 1 (B3), which is a weaker claim, and the GPU plan is cut to that.

## Outcome (14 Nov 2026)
- F1 held: std + DP-TS selected lr = 1e-2 for every objective and eps (the old gate picked 3e-4).
  Its validation Brier after DP-TS fell monotonically from 3e-4 to 1e-2 (CE eps=4: 0.5502 -> 0.5126).
- F2 failed for CE and RLCD. eps = 4, blind (T = 1) minus std + DP-TS, seeds 1-3, 95% t-interval:
  CE    acc +0.06 pts [-0.03, +0.15], Brier +0.0018 [+0.0006, +0.0030] (blind worse)
  RLCD  acc +0.05 pts [-0.31, +0.42], Brier -0.0006 [-0.0028, +0.0017]
  Brier objective: acc +0.06 pts, Brier +0.0018 [+0.0001, +0.0034]. eps = 1 and 8: same picture.
- F3 triggered: the +3.3-point / -0.031 Brier automatic-clipping result (13 Nov) is RETRACTED; it
  came from an under-tuned baseline. blindA's B1 has no toy support and is not run.
- What survives: blind at T = 1 matches std + DP-TS (within 0.002 Brier, 0.3 acc points) without a
  calibration split. ECE at eps = 4: blind 0.022 vs std + DP-TS 0.017 (CE).
