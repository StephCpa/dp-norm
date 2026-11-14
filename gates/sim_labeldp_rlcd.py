"""Zero-GPU check: does RLCD (REINFORCE with reward c - pi_a) stay calibrated under label DP?

Setting: linear-softmax decision model on synthetic data with a known, well-specified P(y|x).
Arms:
  np_rlcd   : non-private RLCD (reward c - pi_a, 4 samples, leave-one-out baseline)
  rr_raw    : labels replaced once by k-ary RR (eps-label-DP), reward computed from noisy label as-is
  rr_debias : same RR labels, reward uses the unbiased estimate c~ = (1{y~=a} - q)/(p - q)
  np_ce     : non-private cross-entropy (reference)
  rr_ce_fc  : cross-entropy on RR labels with forward correction (Patrini et al.), a standard label-noise fix
Metrics on clean test labels: accuracy, top-label ECE (15 bins), multiclass Brier.
RR is applied once per record before training, so everything after is post-processing (pure eps label-DP).
"""
import numpy as np

K, D = 5, 16
N_TRAIN, N_TEST = 20000, 20000
EPOCHS, BATCH, SAMPLES = 15, 128, 4


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def make_data(rng, n, W):
    x = rng.standard_normal((n, D))
    p = softmax(x @ W.T)
    y = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return x, y, p


def rr(rng, y, eps):
    keep = np.exp(eps) / (np.exp(eps) + K - 1)
    out = y.copy()
    flip = rng.random(len(y)) > keep
    other = rng.integers(0, K - 1, len(y))
    other = other + (other >= y)
    out[flip] = other[flip]
    q = 1 / (np.exp(eps) + K - 1)
    return out, keep, q


def ece(probs, y, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def evaluate(W, x, y):
    p = softmax(x @ W.T)
    onehot = np.eye(K)[y]
    return dict(acc=(p.argmax(1) == y).mean(), ece=ece(p, y), brier=((p - onehot) ** 2).sum(1).mean())


def train_rlcd(rng, x, ylab, debias=None, lr=0.3):
    W = np.zeros((K, D))
    n = len(x)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = x[idx], ylab[idx]
            pi = softmax(xb @ W.T)
            a = (pi.cumsum(1)[:, None, :] > rng.random((len(idx), SAMPLES, 1))).argmax(2)  # (B, S)
            c = (a == yb[:, None]).astype(float)
            if debias is not None:
                p_keep, q = debias
                c = (c - q) / (p_keep - q)
            pa = np.take_along_axis(pi, a, 1)
            r = c - pa
            base = (r.sum(1, keepdims=True) - r) / (SAMPLES - 1)
            adv = r - base
            # grad of log pi_a wrt logits = e_a - pi
            g_logits = np.zeros((len(idx), K))
            for j in range(SAMPLES):
                ea = np.eye(K)[a[:, j]]
                g_logits += adv[:, j:j + 1] * (ea - pi)
            g_logits /= SAMPLES
            W += lr * g_logits.T @ xb / len(idx)
    return W


def train_ce(rng, x, ylab, T=None, lr=0.3):
    """CE; if T (noise transition matrix) given, use forward correction: loss on T^T p."""
    W = np.zeros((K, D))
    n = len(x)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = x[idx], ylab[idx]
            p = softmax(xb @ W.T)
            onehot = np.eye(K)[yb]
            if T is None:
                g = onehot - p
            else:
                pn = p @ T  # P(noisy label = j | x)
                w = onehot / pn  # d log pn_y / d pn
                dp = w @ T.T  # d/dp
                g = p * (dp - (dp * p).sum(1, keepdims=True))  # through softmax
            W += lr * g.T @ xb / len(idx)
    return W


def main():
    rng = np.random.default_rng(0)
    W_true = rng.standard_normal((K, D)) * 0.45
    xtr, ytr, ptr = make_data(rng, N_TRAIN, W_true)
    xte, yte, pte = make_data(rng, N_TEST, W_true)
    oracle = dict(acc=(pte.argmax(1) == yte).mean(), ece=ece(pte, yte),
                  brier=((pte - np.eye(K)[yte]) ** 2).sum(1).mean())
    print(f"oracle P(y|x): acc={oracle['acc']:.3f} ece={oracle['ece']:.3f} brier={oracle['brier']:.3f}")
    rows = []
    for seed in range(3):
        r = np.random.default_rng(100 + seed)
        rows.append(("np_rlcd", None, evaluate(train_rlcd(r, xtr, ytr), xte, yte)))
        rows.append(("np_ce", None, evaluate(train_ce(r, xtr, ytr), xte, yte)))
        for eps in [1, 2, 4]:
            ynoisy, p_keep, q = rr(r, ytr, eps)
            T = np.full((K, K), q)
            np.fill_diagonal(T, p_keep)
            rows.append(("rr_raw", eps, evaluate(train_rlcd(r, xtr, ynoisy), xte, yte)))
            rows.append(("rr_debias", eps, evaluate(train_rlcd(r, xtr, ynoisy, debias=(p_keep, q)), xte, yte)))
            rows.append(("rr_ce_fc", eps, evaluate(train_ce(r, xtr, ynoisy, T=T), xte, yte)))
    # aggregate
    keys = sorted({(a, e) for a, e, _ in rows}, key=lambda t: (str(t[1]), t[0]))
    print(f"{'arm':>10} {'eps':>4} {'acc':>13} {'ece':>13} {'brier':>13}")
    for a, e in keys:
        ms = [m for aa, ee, m in rows if aa == a and ee == e]
        def f(k):
            v = np.array([m[k] for m in ms])
            return f"{v.mean():.3f}±{v.std():.3f}"
        print(f"{a:>10} {str(e):>4} {f('acc'):>13} {f('ece'):>13} {f('brier'):>13}")


if __name__ == "__main__":
    main()
