"""When does a unit-level label-blind factor keep the calibrated point stationary? (review P1-3)

Exact enumeration, binary CE, one privacy unit with two rows. The row gradient is (p_r - y_r) d_r.
Row 1 has a fixed input; its label depends on row 2's input x2 in {a, b}:
    P(y1 = 1 | x2 = a) = 0.8, P(y1 = 1 | x2 = b) = 0.2, so P(y1 = 1 | x1) = 0.5.
Row 2's gradient scale depends on x2: ||d2(a)|| = 0.5, ||d2(b)|| = 2; ||d1|| = 1.
A predictor calibrated on each row's own input has p1 = 0.5. Is that stationary under each rule?
    unit : f = C / (B1 + B2), B_r = max(p_r, 1 - p_r) ||d_r||   (depends on x2)
    row  : f_1 = (C/2) / B1                                      (depends on row 1 only)
    std  : f = C / ||sum_r (p_r - y_r) d_r||  with d1 orthogonal to d2 (label-dependent)
Reports E[f (p1 - y1)] at p1 = 0.5 and the p1 at which it vanishes (the fixed point).
"""
import numpy as np

PX = {"a": 0.5, "b": 0.5}
PY1 = {"a": 0.8, "b": 0.2}
D2 = {"a": 0.5, "b": 2.0}
D1, P2, C = 1.0, 0.5, 1e-3


def drift(p1, rule):
    tot = 0.0
    for x, wx in PX.items():
        for y1 in (0, 1):
            w1 = PY1[x] if y1 else 1 - PY1[x]
            for y2 in (0, 1):
                w = wx * w1 * (P2 if y2 else 1 - P2)
                if rule == "unit":
                    f = C / (max(p1, 1 - p1) * D1 + max(P2, 1 - P2) * D2[x])
                elif rule == "row":
                    f = (C / 2) / (max(p1, 1 - p1) * D1)
                else:
                    f = min(1.0, C / np.hypot((p1 - y1) * D1, (P2 - y2) * D2[x]))
                tot += w * f * (p1 - y1)
    return tot / C


def fixed_point(rule):
    lo, hi = 0.01, 0.99
    for _ in range(80):
        mid = (lo + hi) / 2
        if drift(mid, rule) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    for rule in ("unit", "row", "std"):
        print(f"{rule:>4}: E[f (p1 - y1)]/C at p1 = 0.5 is {drift(0.5, rule):+.4f}; fixed point p1* = {fixed_point(rule):.4f}")
