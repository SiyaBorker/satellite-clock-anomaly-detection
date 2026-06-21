import numpy as np
from tensorflow import keras
from data_loader import load_csv
import detector_threshold as thr
import detector_trend as trend

SEQ_LEN = 20
TOL = 8          # a fault counts as detected if a flag fires within +/-TOL gaps
FAULTS = {1: "jump", 2: "missing", 3: "onset", 4: "oscill", 5: "freqstep"}

FILES = [
    "test_jumps", "test_missing", "test_onset",
    "test_oscillatory", "test_freqstep",
    "test_mixed_1", "test_mixed_2",
]



def make_seq_eval(gaps):
    X, centers = [], []
    for i in range(len(gaps) - SEQ_LEN + 1):
        X.append(gaps[i:i + SEQ_LEN])
        centers.append(i + SEQ_LEN - 1)        # align score to last gap
    return np.array(X).reshape(-1, SEQ_LEN, 1), np.array(centers)


# Did any detector flag fire within TOL gaps of each fault of this type?
def recall(flag_positions, fault_positions):
    if len(fault_positions) == 0:
        return None                            # this fault type absent here
    fired = np.where(flag_positions)[0]
    if len(fired) == 0:
        return 0.0
    hits = sum(np.any(np.abs(fired - fp) <= TOL) for fp in fault_positions)
    return 100.0 * hits / len(fault_positions)


# ---- Calibrate threshold + trend on clean validation data ----------------
val = load_csv("dataset/val_normal.csv")
thr_t, _, _ = thr.fit_threshold(val["gaps"], k=6)
trend_t, _, _ = trend.fit_threshold(val["gaps"], 100, k=6)

# ---- Load the trained LSTM -----------------------------------------------
meta = np.load("lstm_meta.npz")
nominal = float(meta["NOMINAL"])
scale = float(meta["SCALE"])
lstm_t = float(meta["THRESHOLD"])
model = keras.models.load_model("lstm_ae.keras")


def lstm_detect(gaps):
    """Return a per-gap boolean flag array from the LSTM."""
    X, centers = make_seq_eval((gaps - nominal) / scale)
    err = np.mean((model.predict(X, verbose=0) - X) ** 2, axis=(1, 2))
    flags = np.zeros(len(gaps), dtype=bool)
    flags[centers] = err > lstm_t
    return flags


# ---- Evaluate -------------------------------------------------------------
print("=" * 88)
print("DETECTION RATE BY FAULT TYPE (%)   '.' = type absent    FP = false positives on normal")
print("=" * 88)
header = f"{'file':16s} {'detector':10s}"
for k in FAULTS:
    header += f"{FAULTS[k]:>9s}"
header += f"{'FP rate':>10s}"
print(header)
print("-" * 88)

for file in FILES:
    d = load_csv(f"dataset/{file}.csv")
    gaps, truth = d["gaps"], d["flags"]
    normal_pos = np.where(truth == 0)[0]
    fault_pos = {k: np.where(truth == k)[0] for k in FAULTS}

    # Compute each detector's per-gap flags
    thr_flags = thr.detect(gaps, thr_t)

    trend_f, trend_centers = trend.detect(gaps, 100, trend_t)
    trend_flags = np.zeros(len(gaps), dtype=bool)
    trend_flags[trend_centers] = trend_f

    lstm_flags = lstm_detect(gaps)

    detectors = [("threshold", thr_flags),
                 ("trend", trend_flags),
                 ("lstm-AE", lstm_flags)]

    for di, (name, flags) in enumerate(detectors):
        row = f"{file if di == 0 else '':16s} {name:10s}"
        for k in FAULTS:
            r = recall(flags, fault_pos[k])
            row += f"{'.':>9s}" if r is None else f"{r:8.1f}%"
        fp = 100.0 * flags[normal_pos].sum() / max(len(normal_pos), 1)
        row += f"{fp:9.4f}%"
        print(row)
    print("-" * 88)