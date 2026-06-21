"""Trend test: least-squares slope over L gaps; flag |slope| > threshold."""
import numpy as np
from data_loader import load_csv
NOMINAL = 1.0
L = 100
def slope_series(gaps, L):
    n = len(gaps)
    if n < L: return np.empty(0), np.empty(0, dtype=int)
    x = np.arange(L, dtype=np.float64); xc = x - x.mean()
    denom = (xc**2).sum()
    win = np.lib.stride_tricks.sliding_window_view(gaps, L)
    yc = win - win.mean(axis=1, keepdims=True)
    slopes = (yc * xc).sum(axis=1) / denom
    return slopes, np.arange(L-1, n)
def fit_threshold(clean_gaps, L, k=6.0):
    s, _ = slope_series(clean_gaps, L)
    return k * s.std(), s.std(), np.abs(s).max()
def detect(gaps, L, threshold):
    s, c = slope_series(gaps, L)
    return np.abs(s) > threshold, c
if __name__ == "__main__":
    val = load_csv("dataset/val_normal.csv")
    thr, std, mx = fit_threshold(val["gaps"], L, 6.0)
    print(f"  normal slope std : {std:.3e} s/gap")
    print(f"  chosen threshold : {thr:.3e} s/gap")
    f,_ = detect(val["gaps"], L, thr)
    print(f"False positives on clean val : {f.sum()} / {len(f)}")
