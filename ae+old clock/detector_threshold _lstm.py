"""Baseline threshold detector: flag |gap - 1.0| > threshold."""
import numpy as np
from data_loader import load_csv
NOMINAL = 1.0
def fit_threshold(clean_gaps, k=6.0):
    dev = np.abs(clean_gaps - NOMINAL)
    return k * dev.std(), dev.std(), dev.max()
def detect(gaps, threshold):
    return np.abs(gaps - NOMINAL) > threshold
if __name__ == "__main__":
    val = load_csv("dataset/val_normal.csv")
    thr, std, mx = fit_threshold(val["gaps"], 6.0)
    print(f"  normal deviation std : {std*1000:.4f} ms")
    print(f"  chosen threshold (6s): {thr*1000:.4f} ms")
    print(f"False positives on clean val : {detect(val['gaps'],thr).sum()} / {len(val['gaps'])}")
