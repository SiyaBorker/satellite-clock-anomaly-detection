"""
detector_trend.py  --  Gradual-drift detector (no training).

Part 2 of the hybrid detector. Catches GRADUAL DRIFT, which the
autoencoder and the threshold baseline structurally cannot see: drift
never makes a single gap deviate sharply, it nudges every gap
imperceptibly, so any short window still looks locally normal.

Method
------
Over a long sliding window of the last L gaps, fit a straight line
(least-squares) and read its SLOPE. A non-zero slope means the gaps are
trending -- i.e. the clock rate is drifting. Flag when |slope| exceeds a
threshold learned from clean data.

Why slope, not value
--------------------
A drifting clock produces inter-frame gaps that are each ~1.0 s but
slowly increasing or decreasing. The VALUE stays near nominal; only the
TREND reveals the drift. Linear regression slope is the minimal statistic
that captures trend.

This is hand-codable in C: a closed-form slope over L points is a few
sums, no matrix library needed.
"""

import numpy as np
from data_loader import load_csv

NOMINAL = 1.0
L = 100            # trend window length (gaps). Long, because drift is slow.


def slope_series(gaps, L):
    """
    Compute the least-squares slope of each length-L sliding window.

    Closed-form slope for points (x=0..L-1, y=gaps):
        slope = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)
    The x-denominator is constant for fixed L, so precompute it.

    Returns slopes aligned to the LAST gap of each window (index L-1..n-1).
    """
    n = len(gaps)
    if n < L:
        return np.empty(0), np.empty(0, dtype=int)

    x = np.arange(L, dtype=np.float64)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()           # constant for fixed L

    windows = np.lib.stride_tricks.sliding_window_view(gaps, L)
    y_centered = windows - windows.mean(axis=1, keepdims=True)
    slopes = (y_centered * x_centered).sum(axis=1) / denom

    centers = np.arange(L - 1, n)
    return slopes, centers


def fit_threshold(clean_gaps, L, k=6.0):
    """Threshold = k * std of slopes on clean data."""
    slopes, _ = slope_series(clean_gaps, L)
    return k * slopes.std(), slopes.std(), np.abs(slopes).max()


def detect(gaps, L, threshold):
    """Return (flags, centers): flags True where |slope| > threshold."""
    slopes, centers = slope_series(gaps, L)
    return np.abs(slopes) > threshold, centers


if __name__ == "__main__":
    val = load_csv("dataset/val_normal.csv")
    thr, std, mx = fit_threshold(val["gaps"], L, k=6.0)
    print(f"Trend test (L={L}) fit on clean val data:")
    print(f"  normal slope std : {std:.3e} s/gap")
    print(f"  normal slope max : {mx:.3e} s/gap")
    print(f"  chosen threshold : {thr:.3e} s/gap")
    print()

    flags, centers = detect(val["gaps"], L, thr)
    print(f"False positives on clean val : {flags.sum()} / {len(flags)}")
