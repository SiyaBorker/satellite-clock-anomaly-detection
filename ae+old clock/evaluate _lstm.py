"""
evaluate.py  --  Evaluation harness (threshold + trend + LSTM autoencoder).

Scores three detectors against ground-truth fault_flag, broken down by the
five fault types, on every test file.

Fault flags: 1 jump, 2 missing, 3 slow-onset, 4 oscillatory, 5 freqstep.

Run from inside detector/ (after generating data and training the LSTM):
    python clock_sim.py
    python detector_lstm.py
    python evaluate.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loader import load_csv
import detector_threshold as thr_mod
import detector_trend as trend_mod
from detector_lstm import make_sequences, recon_error

FAULT_NAMES = {1: "jump", 2: "missing", 3: "onset", 4: "oscill", 5: "freqstep"}

TEST_FILES = [
    "test_jumps",
    "test_missing",
    "test_onset",
    "test_oscillatory",
    "test_freqstep",
    "test_mixed_1",
    "test_mixed_2",
]

TOL = 8        # detection counted if a flag fires within +/-TOL gaps
L   = 100      # trend window


def detected_within(flag_bool, fault_pos, tol):
    """For each fault position, did any detector flag fire within +/-tol?"""
    if len(fault_pos) == 0:
        return None      # fault type absent in this file
    flagged = np.where(flag_bool)[0]
    if len(flagged) == 0:
        return 0
    return sum(np.any(np.abs(flagged - fp) <= tol) for fp in fault_pos)


def load_lstm():
    meta = np.load("lstm_meta.npz")
    NOMINAL = float(meta["NOMINAL"]); SCALE = float(meta["SCALE"])
    THRESHOLD = float(meta["THRESHOLD"]); SEQ_LEN = int(meta["SEQ_LEN"])
    model = keras.models.load_model("lstm_ae.keras")

    def flags_for(gaps):
        seqs, centers = make_sequences((gaps - NOMINAL) / SCALE, SEQ_LEN, 1)
        err = recon_error(model, seqs)
        full = np.zeros(len(gaps), dtype=bool)
        full[centers] = err > THRESHOLD
        return full

    return flags_for


def main():
    val = load_csv("dataset/val_normal.csv")
    thr_t, _, _ = thr_mod.fit_threshold(val["gaps"], k=6.0)
    trend_t, _, _ = trend_mod.fit_threshold(val["gaps"], L, k=6.0)
    lstm_flags_for = load_lstm()

    # detector flag-functions: name -> (flags_bool_for_gaps)
    def thr_flags(gaps):
        return thr_mod.detect(gaps, thr_t)

    def trend_flags(gaps):
        f, centers = trend_mod.detect(gaps, L, trend_t)
        full = np.zeros(len(gaps), dtype=bool)
        full[centers] = f
        return full

    detectors = [
        ("threshold", thr_flags),
        ("trend",     trend_flags),
        ("lstm-AE",   lstm_flags_for),
    ]

    print("=" * 90)
    print("DETECTION RATE BY FAULT TYPE  (recall %)   '.' = fault type absent in file")
    print("=" * 90)
    head = f"{'file':16s} {'detector':10s}"
    for k in FAULT_NAMES:
        head += f"{FAULT_NAMES[k]:>9s}"
    head += f"{'FP rate':>10s}"
    print(head)
    print("-" * 90)

    for name in TEST_FILES:
        d = load_csv(f"dataset/{name}.csv")
        gaps, flags = d["gaps"], d["flags"]
        normal_pos = np.where(flags == 0)[0]
        n_normal = len(normal_pos)
        pos_by_type = {k: np.where(flags == k)[0] for k in FAULT_NAMES}

        for di, (dname, fn) in enumerate(detectors):
            fb = fn(gaps)
            row = f"{name if di==0 else '':16s} {dname:10s}"
            for k in FAULT_NAMES:
                got = detected_within(fb, pos_by_type[k], TOL)
                if got is None:
                    row += f"{'.':>9s}"
                else:
                    tot = len(pos_by_type[k])
                    row += f"{100.0*got/tot:8.1f}%"
            fp = fb[normal_pos].sum() / max(n_normal, 1)
            row += f"{100*fp:9.4f}%"
            print(row)
        print("-" * 90)


if __name__ == "__main__":
    main()
