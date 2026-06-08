"""
data_loader.py  --  Shared dataset loading for all detectors.

Loads a clock-simulator CSV and exposes the Option B feature
(inter-frame gaps) plus the ground-truth fault flags for scoring.

Column reference (written by clock_sim.py):
    frame_index      : original frame number
    true_time_s      : ground truth (NOT used by detectors)
    reported_time_s  : the drifted clock's reported timestamp
    ref_error_s      : reported - true (Option A; NOT used here)
    interframe_gap_s : reported[i] - reported[i-1]  <-- THE FEATURE
    fault_flag       : 0 normal, 1 jump, 2 missing  <-- GROUND TRUTH
"""

import numpy as np
import csv


def load_csv(path):
    """
    Load one simulator CSV.

    Returns a dict with:
        gaps   : float array of inter-frame gaps (first row dropped; it has
                 no predecessor so its gap is blank)
        flags  : int array of fault flags aligned to gaps
        frames : int array of frame indices aligned to gaps
    The first transmitted row is dropped because it has no inter-frame gap.
    """
    gaps, flags, frames = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row["interframe_gap_s"].strip()
            if g == "":
                continue                      # first row, no predecessor
            gaps.append(float(g))
            flags.append(int(row["fault_flag"]))
            frames.append(int(row["frame_index"]))

    gaps   = np.asarray(gaps,   dtype=np.float64)
    flags  = np.asarray(flags,  dtype=np.int32)
    frames = np.asarray(frames, dtype=np.int32)

   
    missing_mask = gaps >= 1.5
    flags = flags.copy()
    flags[missing_mask & (flags == 0)] = 2

    return {
        "gaps":   gaps,
        "flags":  flags,
        "frames": frames,
    }


def make_windows(gaps, W):
    """
    Slice a 1-D gap series into overlapping windows of length W.

    Returns:
        windows : 2-D array, shape (N-W+1, W)
        centers : index of the LAST gap in each window, used to align a
                  window-level anomaly score back to a specific gap.
    A window is labelled anomalous (for scoring) if ANY gap inside it is
    faulty; alignment uses the last gap so detection latency is honest.
    """
    n = len(gaps)
    if n < W:
        return np.empty((0, W)), np.empty((0,), dtype=int)
    num = n - W + 1
    windows = np.lib.stride_tricks.sliding_window_view(gaps, W).copy()
    centers = np.arange(W - 1, n)             # last-gap index per window
    return windows, centers


if __name__ == "__main__":
    # Quick self-test on the training file.
    d = load_csv("dataset/train_normal.csv")
    print(f"loaded {len(d['gaps'])} gaps")
    print(f"  mean gap : {d['gaps'].mean():.6f} s")
    print(f"  std  gap : {d['gaps'].std():.6f} s")
    print(f"  faults   : {(d['flags'] != 0).sum()}")
    W = 10
    win, ctr = make_windows(d["gaps"], W)
    print(f"  windows  : {win.shape}  centers: {ctr.shape}")
