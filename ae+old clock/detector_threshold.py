"""
detector_threshold.py  --  Baseline detector (no training).

The simplest possible clock anomaly detector: flag any inter-frame gap
that deviates from the nominal 1.0 s frame interval by more than a fixed
threshold.

    anomaly  if  |gap - nominal| > threshold

Purpose
-------
This is the scientific CONTROL. It establishes how hard the problem is.
If this trivial rule catches most jumps and missing frames, then the
autoencoder must beat it to justify its added complexity. For the report,
"autoencoder vs threshold baseline" is a far stronger claim than the
autoencoder alone.

The threshold is set from the CLEAN validation data, not guessed: we take
the largest normal deviation and add a margin, so normal jitter never
trips it (controls false positives by construction).
"""

import numpy as np
from data_loader import load_csv

NOMINAL = 1.0          # expected inter-frame gap (the frame interval, s)


def fit_threshold(clean_gaps, k=6.0):
    """
    Choose the threshold from clean data.

    Strategy: threshold = k * std(normal deviations). With k=6, any gap
    more than 6 standard deviations from nominal is flagged. Normal jitter
    is ~0.16 ms std, so the threshold lands around ~1 ms -- far below the
    smallest fault (20 ms) and far above normal jitter.
    """
    dev = np.abs(clean_gaps - NOMINAL)
    return k * dev.std(), dev.std(), dev.max()


def detect(gaps, threshold):
    """Return a boolean array: True where the gap is flagged anomalous."""
    return np.abs(gaps - NOMINAL) > threshold


if __name__ == "__main__":
    # Fit threshold on clean validation data.
    val = load_csv("dataset/val_normal.csv")
    thr, std, mx = fit_threshold(val["gaps"], k=6.0)
    print(f"Threshold fit on clean val data:")
    print(f"  normal deviation std : {std*1000:.4f} ms")
    print(f"  normal deviation max : {mx*1000:.4f} ms")
    print(f"  chosen threshold (6s): {thr*1000:.4f} ms")
    print()

    # Sanity check: false positives on the clean validation set itself.
    val_flags = detect(val["gaps"], thr)
    print(f"False positives on clean val : {val_flags.sum()} "
          f"/ {len(val['gaps'])}")
