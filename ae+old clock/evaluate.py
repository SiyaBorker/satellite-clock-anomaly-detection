"""
evaluate.py  --  Evaluation harness.

Scores the three detectors against ground-truth fault_flag on every faulty
test file, broken down BY FAULT TYPE.

  1. threshold  : flag |gap - 1.0| > thr     (jumps, missing)
  2. autoencoder: flag recon-error > thr      (jumps, missing)
  3. trend      : flag |slope| > thr          (drift)

Run from inside the detector/ folder:
    python evaluate.py
Requires ae_weights.npz (run detector_autoencoder.py first).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np

from data_loader import load_csv, make_windows
import detector_threshold as thr_mod
import detector_trend as trend_mod

TEST_FILES = [
    "test_jumps_only",
    "test_missing_only",
    "test_mixed_1",
    "test_mixed_2",
]

# drift_bias per file (from the dataset spec in clock_sim.py)
DRIFT_BIAS = {
    "test_jumps_only":   +1e-6,
    "test_missing_only": -2e-6,
    "test_mixed_1":      +3e-6,
    "test_mixed_2":      -3e-6,
}

TOL = 5      # detection counted if a flag fires within +/-TOL gaps of the fault
W   = 10     # autoencoder window length
L   = 100    # trend window length


def detected_within(flag_positions, fault_positions, tol):
    """For each fault position, was there ANY detector flag within +/-tol?"""
    if len(fault_positions) == 0:
        return 0
    flagged = np.where(flag_positions)[0]
    if len(flagged) == 0:
        return 0
    detected = 0
    for fp in fault_positions:
        if np.any(np.abs(flagged - fp) <= tol):
            detected += 1
    return detected


def load_ae():
    """Load trained AE weights; numpy forward pass (no TensorFlow needed)."""
    z = np.load("ae_weights.npz")
    W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
    NOMINAL = float(z["NOMINAL"])
    SCALE = float(z["SCALE"])
    THRESHOLD = float(z["THRESHOLD"])

    def recon_error(gaps):
        windows, centers = make_windows(gaps, W)
        X = (windows - NOMINAL) / SCALE
        h = np.maximum(0.0, X @ W1 + b1)   # ReLU
        out = h @ W2 + b2                  # linear
        err = ((out - X) ** 2).mean(axis=1)
        return err, centers

    return recon_error, THRESHOLD


def main():
    # ---- Fit thresholds on clean validation data -----------------------
    val = load_csv("dataset/val_normal.csv")
    thr_threshold, _, _ = thr_mod.fit_threshold(val["gaps"], k=6.0)
    trend_threshold, _, _ = trend_mod.fit_threshold(val["gaps"], L, k=6.0)
    ae_recon, ae_threshold = load_ae()

    # ---- Detection rate by fault type ----------------------------------
    print("=" * 72)
    print("DETECTION RATE BY FAULT TYPE  (recall %)")
    print("=" * 72)
    print(f"{'file':18s} {'detector':12s} {'jumps':>8s} {'missing':>9s} {'FP rate':>10s}")
    print("-" * 72)

    for name in TEST_FILES:
        d = load_csv(f"dataset/{name}.csv")
        gaps, flags = d["gaps"], d["flags"]
        jump_pos = np.where(flags == 1)[0]
        miss_pos = np.where(flags == 2)[0]
        normal_pos = np.where(flags == 0)[0]
        n_normal = len(normal_pos)

        # Threshold detector (per-gap)
        thr_flags = thr_mod.detect(gaps, thr_threshold)
        thr_j = detected_within(thr_flags, jump_pos, TOL)
        thr_m = detected_within(thr_flags, miss_pos, TOL)
        thr_fp = thr_flags[normal_pos].sum() / max(n_normal, 1)

        # Autoencoder (window, aligned to last gap)
        ae_err, ae_centers = ae_recon(gaps)
        ae_flag_full = np.zeros(len(gaps), dtype=bool)
        ae_flag_full[ae_centers] = ae_err > ae_threshold
        ae_j = detected_within(ae_flag_full, jump_pos, TOL)
        ae_m = detected_within(ae_flag_full, miss_pos, TOL)
        ae_fp = ae_flag_full[normal_pos].sum() / max(n_normal, 1)

        def pct(num, den):
            return f"{100.0*num/den:6.1f}" if den else "   n/a"

        print(f"{name:18s} {'threshold':12s} "
              f"{pct(thr_j,len(jump_pos)):>8s} {pct(thr_m,len(miss_pos)):>9s} "
              f"{100*thr_fp:9.4f}%")
        print(f"{'':18s} {'autoencoder':12s} "
              f"{pct(ae_j,len(jump_pos)):>8s} {pct(ae_m,len(miss_pos)):>9s} "
              f"{100*ae_fp:9.4f}%")

    # ---- Trend test (drift detection) ----------------------------------
    print(f"\n{'='*72}")
    print("TREND TEST  (drift detection)")
    print("=" * 72)
    print("Drift is a whole-file property (the drift_bias), not a per-gap")
    print("label. Reports whether the trend test fires somewhere in the file")
    print("and what fraction of gaps it flags as drifting.\n")
    print(f"{'file':18s} {'drift_bias':>12s} {'fired?':>8s} {'%flagged':>10s}")
    print("-" * 52)

    for name in TEST_FILES:
        d = load_csv(f"dataset/{name}.csv")
        flags_t, _ = trend_mod.detect(d["gaps"], L, trend_threshold)
        fired = "YES" if flags_t.any() else "no"
        pct_flagged = 100.0 * flags_t.sum() / max(len(flags_t), 1)
        print(f"{name:18s} {DRIFT_BIAS[name]:+12.0e} {fired:>8s} {pct_flagged:9.1f}%")


if __name__ == "__main__":
    main()
