import math
import random

import pytest

from researchctl import stats

scipy = pytest.importorskip("scipy.stats")


def test_t_quantile_and_sf_match_scipy():
    for df in (1, 2, 4, 9, 30, 120):
        assert math.isclose(stats.t_quantile(0.975, df), scipy.t.ppf(0.975, df), rel_tol=1e-6)
        for t in (0.3, 1.0, 2.5, 6.0):
            assert math.isclose(stats.t_sf2(t, df), 2 * scipy.t.sf(t, df), rel_tol=1e-6, abs_tol=1e-12)


def test_welch_and_paired_match_scipy():
    rng = random.Random(0)
    a = [rng.gauss(0, 1) for _ in range(5)]
    b = [rng.gauss(0.8, 2) for _ in range(7)]
    w = stats.compare(a, b)
    ref = scipy.ttest_ind(a, b, equal_var=False)
    assert math.isclose(w["p"], ref.pvalue, rel_tol=1e-6)
    c = [x + rng.gauss(0.3, 0.1) for x in a]
    pr = stats.compare(c, a, paired=True)
    assert math.isclose(pr["p"], scipy.ttest_rel(c, a).pvalue, rel_tol=1e-6)


def test_holm():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)
