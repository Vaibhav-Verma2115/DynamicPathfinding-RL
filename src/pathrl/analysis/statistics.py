"""Summary statistics with 95% confidence intervals.

For a metric measured once per independent seed (n values), the 95% CI of the mean is

    mean +/- t_{0.975, n-1} * s / sqrt(n)

where s is the sample standard deviation and t is Student's t quantile. We use t rather
than 1.96 because n is small (5-30). With n = 5, t = 2.776, so intervals are noticeably
wider. The unit of replication is the SEED (independent map + run), never individual
episodes or steps within one seed, which are correlated.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values: Iterable[float], confidence: float = 0.95) -> dict:
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan, "n": 0}
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan, "n": 1}
    std = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1)) * std / math.sqrt(n)
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "std": std, "n": n}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """CI for a proportion (e.g. pooled success rate). Unlike p +/- z*sqrt(p(1-p)/n), it stays
    inside [0, 1] and behaves sensibly at p = 0 or 1."""
    if n == 0:
        return (np.nan, np.nan)
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(df: pd.DataFrame, by: Sequence[str], metrics: Sequence[str], confidence: float = 0.95) -> pd.DataFrame:
    """One row per group with <metric>_mean, _ci_low, _ci_high, _std and n."""
    rows = []
    for keys, group in df.groupby(list(by), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        for m in metrics:
            s = mean_ci(group[m], confidence)
            row.update({f"{m}_mean": s["mean"], f"{m}_ci_low": s["ci_low"], f"{m}_ci_high": s["ci_high"],
                        f"{m}_std": s["std"]})
            row["n"] = s["n"]
        rows.append(row)
    return pd.DataFrame(rows)
