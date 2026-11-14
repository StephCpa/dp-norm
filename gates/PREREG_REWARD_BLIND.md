# Reward-blind clipping for binary-reward GRPO: pre-registration (13 Nov 2026, written before any run)

Model (the earlier one): per-response score s_i = (R_i - p) u + xi_i, xi_i isotropic, ||xi|| ~ sigma.
Group gradient g = (1/K) sum_i A_i s_i. Advantages: "center" (Dr. GRPO / RLOO-style, A = R - mean R)
or "std" (GRPO, A = (R - mean R)/std R, population std, 0 for constant groups).
Write S = m(K-m)/K^2 for a group with m successes; E[S] = (K-1)/K * p(1-p).

Reward-blind bound B = max over non-constant R' in {0,1}^K of ||g(R')||, with the K score vectors held
fixed (enumerated exactly from the K x K Gram). Triangle bound B_tri = (1/K) sum_i max_R' |A'_i| ||s_i||.
Both are functions of the responses only, so sensitivity stays <= C and accounting is unchanged.

## Claims under test (the 13 Nov derivation sent to me) and my predictions

W1. Claim: the worst case is an extreme split (K-1 : 1). Prediction: false for centered advantages.
    ||g(R')||^2 ~ (sigma^2/K^2) sum_i A'_i^2 = sigma^2 m'(K-m')/K^3 when scores are near-orthogonal,
    which peaks at the balanced split. Predict argmax m' = K/2 in >= 95% of noise-dominated groups.

W2. Claim: blind weight = sqrt(p(1-p))/c_worst(p), "stronger suppression at extreme p", a new reweighting.
    Prediction: in the noise-dominated regime B is essentially reward-independent (mean B varies < 2%
    across p), so the blind weight is E[S] proportional to p(1-p): the UNCLIPPED (Dr. GRPO) weighting,
    objective sum p. Measured shape within 3% of p(1-p)/0.25 at every p. Standard clipping gives
    E[sqrt S], the GRPO-std / arcsine weighting. Blind clipping does not add a reweighting; it removes one.

W3. Absolute signal per unit C (automatic clipping, centered): blind/standard = 2 E[S]/E[sqrt S] <= 1,
    within 3% of that formula. Blind gives up signal on imbalanced groups. Under DP that is SNR lost.

W4. The triangle bound is looser, not tighter: B_tri/B ~ 2(K-1)/sqrt(K) (4.9 at K=8). Same shape as
    exact blind, signal smaller by that factor.

W5. With std (GRPO) advantages, sum A'^2 = K for every non-constant split, so B ~ constant and so is
    the standard norm: blind/standard within 5% of 1 at every p. Nothing to gain.

W6. Signal-dominated regime: the realized R is the worst case, blind = standard within 2% at every p.

W7. Claim: the gap widens under automatic clipping. Prediction: true for the weighting distortion.
    With a threshold at the median standard norm (p = 0.5), standard clipping's distortion relative
    to unclipped at p = 0.05 is strictly smaller than under automatic clipping; blind's is ~0 in both.

## Training toy (linearized policy-gradient model, DP-SGD, eps = 4) predictions

T1. Automatic clipping, centered advantages: standard >= blind on final mean success rate J
    (seed-mean), because blind's extra unbiasedness is worth less than its SNR loss. If blind beats
    standard by >= 0.01 in J, reward-blind clipping for reward maximization is worth a GPU arm.
    Otherwise it is dead for reward maximization; it stays a calibration-objective tool only.
T2. blind-tri < blind-exact on J.
T3. On hard prompts (initial p < 0.1), standard >= blind (arcsine weighting favors them).

## Addendum before the main training run (13 Nov 2026)
A one-seed lr probe of the first task design (psi = U w + beta e, no shared direction) showed the
task was degenerate: no direction helps most prompts, so J stayed near 0.5 for every method while
hard and easy prompts traded places. Disclosure: that probe already had standard >= blind at its
best lr (J 0.525 vs 0.522, one seed), consistent with T1. Changed the task before the main run:
psi = kappa mu + U w + beta e, with mu a shared skill direction orthogonal to the conflict subspace
U (kappa = 1, alpha = 1, beta = 0.3, sigma_xi = 10), and widened the lr grid to 0.1 ... 1000
(nine values, best per method by seed-mean J). T1-T3 unchanged.

## Addendum 2 (13 Nov 2026, after the kappa = 1 run)
kappa = 1 saturates: every DP arm reaches J >= 0.999 at its best lr, so T1 is untestable there
(ceiling tie). Below the best lr the order was std > blind > tri at every lr. A one-seed probe at
kappa = 0.1 (std_auto, blind_auto only) gave best-lr J 0.842 vs 0.739, i.e. not saturated, and
already in T1's direction (disclosed). Main run: kappa = 0.1, all methods, 3 seeds, lr grid
0.03 ... 30 (7 values). T1-T3 unchanged; kappa = 1 results kept as reward_blind_train_kappa1.*.

## Outcome (13 Nov 2026)
- W1 held (the claim is refuted): argmax is the balanced split in 100% of noise-dominated groups (K = 4, 8, 16); K-1:1 in 0%.
- W2 held: mean B flat across p (0.1805-0.1806 at K=8); blind shape = p(1-p) to 3 decimals.
- W3 held at K = 4, 8 (within 2%); missed at K = 16 (up to 3.8% vs 3% tolerance): the max over 2^K
  configurations inflates B by +2% (K=8) and +4% (K=16) at d = 4096.
- W4 held: B_tri/B = 2.97 (K=4), 4.85 (K=8), 7.23 (K=16).
- W5 held: blind/standard = 0.988, 0.977, 0.963 (K = 4, 8, 16), constant in p.
- W6 held: 1.000 at every p.  W7 held: standard distortion at p = 0.05 is 0.191 vs 0.189 (threshold), 0.251 (automatic).
- T1 held: standard > blind, 3/3 seeds at all 7 lrs; best-lr J 0.830 vs 0.730 (paired +0.103, +0.058, +0.139).
  Kill rule triggered: reward-blind clipping is dead for reward maximization.
- T2 held (tri 0.551). T3 held (hard prompts 0.696 vs 0.593).

## Revised wording after review (14 Nov 2026)
- The conclusion "dead for reward maximization" is too strong. Supported: no advantage was observed
  for reward-blind clipping under the tested score-noise models and configurations (one linearized toy,
  one task family, 3 seeds, 1,500 steps, eps = 4); scaling this sub-line is suspended.
- Retracted: "every clipping rule keeps the per-prompt optimum, so there is no fixed point to protect".
  With shared parameters and conflicting prompts the stationary points of sum_x w(p_x) grad p_x depend
  on the weighting w, so blind clipping (about the unclipped weights in the noise-dominated regime) keeps
  the stationary points of sum_x p_x, and standard clipping moves toward the arcsine surrogate's. The toy
  measured finite-budget J only, not where either converges.
- For finite K the standard weight is E[sqrt S] exactly (as measured); the arcsine objective is its
  large-K limit, not an identity at K = 8.
- The full training outputs (reward_blind_train*.json, reward_blind_weights.json and their logs) are in
  gates/ of the 14 Nov dpdecide tarball.
