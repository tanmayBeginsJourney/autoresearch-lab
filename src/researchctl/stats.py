"""Small, dependency-free statistics: t intervals, Welch / paired t-tests, Holm correction."""
from __future__ import annotations

import math


def mean(x):
    return sum(x) / len(x)


def sd(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def _betacf(a, b, x):
    # Continued fraction for the incomplete beta function (Numerical Recipes, betacf).
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_sf2(t, df):
    """Two-sided p-value for Student t."""
    return betainc(df / 2, 0.5, df / (df + t * t))


def t_quantile(p, df):
    """Inverse CDF via bisection (p in (0.5, 1))."""
    lo, hi = 0.0, 1e3
    target = 2 * (1 - p)
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def describe(x, level=0.95):
    n = len(x)
    if n == 0:
        return {"n": 0}
    m = mean(x)
    if n < 2:
        return {"n": 1, "mean": m}
    s = sd(x)
    half = t_quantile(0.5 + level / 2, n - 1) * s / math.sqrt(n)
    return {"n": n, "mean": m, "sd": s, "ci": [m - half, m + half]}


def compare(treat, ctrl, paired=False, level=0.95):
    """Difference treat - ctrl with CI and two-sided p (Welch or paired)."""
    if paired:
        if len(treat) != len(ctrl):
            raise ValueError("paired comparison needs equal lengths")
        d = [a - b for a, b in zip(treat, ctrl)]
        n = len(d)
        if n < 2:
            return {"n": n, "diff": d[0] if d else None, "test": "paired-t", "insufficient": True}
        md, sdd = mean(d), sd(d)
        se, df = sdd / math.sqrt(n), n - 1
        test = "paired-t"
    else:
        n1, n2 = len(treat), len(ctrl)
        if n1 < 2 or n2 < 2:
            return {"n": [n1, n2], "test": "welch-t", "insufficient": True}
        v1, v2 = sd(treat) ** 2 / n1, sd(ctrl) ** 2 / n2
        md, se = mean(treat) - mean(ctrl), math.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)) if v1 + v2 > 0 else n1 + n2 - 2
        test = "welch-t"
        n = [n1, n2]
    if se == 0:
        p = 0.0 if md != 0 else 1.0
        half = 0.0
    else:
        p = t_sf2(md / se, df)
        half = t_quantile(0.5 + level / 2, df) * se
    pooled = math.sqrt((sd(treat) ** 2 + sd(ctrl) ** 2) / 2) if len(treat) > 1 and len(ctrl) > 1 else float("nan")
    return {"n": n, "diff": md, "ci": [md - half, md + half], "p": p, "df": df, "test": test,
            "cohen_d": md / pooled if pooled and pooled == pooled and pooled > 0 else None}


def holm(pvals: dict, alpha: float) -> dict:
    """Holm-Bonferroni: returns {key: adjusted_p}; reject where adjusted_p <= alpha."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
