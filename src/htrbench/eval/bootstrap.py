"""Page-level bootstrap for micro-averaged rates such as CER = sum(edits) / sum(reference characters).

Pages are the sampling unit: errors within a page are not independent, and the benchmark generalises to other pages,
not to other characters of the same pages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(df: pd.DataFrame, num_col: str, den_col: str, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(rate, lower, upper) over the rows of ``df``, one row per page."""
    num, den = df[num_col].to_numpy(dtype=float), df[den_col].to_numpy(dtype=float)
    if den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    point = float(num.sum() / den.sum())
    if len(num) < 2:
        return point, point, point
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.nanpercentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
