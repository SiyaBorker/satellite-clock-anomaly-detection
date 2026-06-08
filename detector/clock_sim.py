"""
clock_sim.py  --  Spacecraft Clock Simulator (dataset generator)
================================================================
Generates the ML dataset: a stream of reported timestamps from a drifting
spacecraft clock, with faults injected at known positions.

Physical effects modelled (each toggleable in PHYSICS below)
------------------------------------------------------------
  - Steady drift + random-walk wander : oscillator rate is not perfect and
    wanders slowly (gradual-drift fault when large).
  - Temperature cycle                 : clock RATE varies sinusoidally with
    the orbital thermal cycle (sunlight/eclipse). Gives normal data a
    smooth, learnable structure.
  - Correlated (colored) readout jitter: per-frame timing noise is an AR(1)
    process (each sample depends on the last), not independent white noise.
    Real oscillators have colored noise; this is structure a model can learn.
  - Timestamp quantization            : the clock reports time in discrete
    ticks (it cannot report a continuous real number). Rounds the reported
    timestamp to the nearest QUANTUM.

Injected faults
---------------
  - jump    (flag 1) : sudden step in the clock offset at a random frame.
  - missing (flag 2) : a frame is dropped (omitted from the CSV); shows up
    as a doubled inter-frame gap on the next surviving row.

Output
------
  ./dataset/*.csv          one row per TRANSMITTED frame
  ./dataset/manifest.csv   index of files + roles

Run
---
    pip install numpy
    python clock_sim.py            # generates the full dataset
"""

import os
import csv
import numpy as np

# =====================================================================
# PHYSICS  --  the normal-clock model shared by every file
# =====================================================================
# All effects below describe NORMAL behaviour. Faults are added on top,
# per-file, via the DATASET_FILES spec.

PHYSICS = dict(
    DURATION_SEC   = 86400.0,    # 24 h per file
    FRAME_INTERVAL = 1.0,        # nominal seconds between frames (1 Hz)

    # --- steady drift + random-walk wander ---------------------------
    DRIFT_RATE_RW_SIGMA = 5e-7,  # random-walk step on the rate (s/s per frame)

    # --- temperature cycle (orbital thermal effect on rate) ----------
    TEMP_CYCLE_ON     = True,
    TEMP_AMPLITUDE    = 3e-6,    # peak rate change from thermal cycle (s/s)
    ORBIT_PERIOD_SEC  = 5400.0,  # 90-minute low-Earth orbit

    # --- correlated (colored) readout jitter, AR(1) ------------------
    JITTER_ON     = True,
    JITTER_RHO    = 0.95,        # AR(1) memory: 0 = white noise, →1 = slow wander
    JITTER_SIGMA  = 1e-4,        # innovation std of the jitter process (s)

    # --- timestamp quantization --------------------------------------
    QUANTIZE_ON  = False,
    QUANTUM      = 1e-3,         # clock reports in 1 ms ticks (when ON)

    # --- jump fault magnitude ----------------------------------------
    JUMP_SIZE_RANGE = (0.020, 0.150),  # seconds
    JUMP_SIGN_RANDOM = True,
)

DATASET_DIR = "dataset"

# Each entry: (filename, seed, num_jumps, missing_prob, drift_bias)
#   num_jumps    : sudden jumps to inject (0 = none)
#   missing_prob : per-frame probability of a dropped frame (0 = none)
#   drift_bias   : steady drift rate added to the wander (s/s); sign and
#                  magnitude vary per file so the detector sees a range.
# Clean files (training / validation) have num_jumps=0 and missing_prob=0.
DATASET_FILES = [
    # name,                seed, n_jumps, miss_prob, drift_bias
    ("train_normal",       101,  0,       0.0,       +2e-6),   # clean
    ("val_normal",         202,  0,       0.0,       -1e-6),   # clean
    ("test_jumps_only",    301,  6,       0.0,       +1e-6),
    ("test_missing_only",  302,  0,       0.0015,    -2e-6),
    ("test_mixed_1",       303,  5,       0.0010,    +3e-6),
    ("test_mixed_2",       304,  4,       0.0008,    -3e-6),
]


# =====================================================================
# SIMULATION
# =====================================================================

def simulate(seed, num_jumps, missing_prob, drift_bias, phys=PHYSICS):
    """
    Simulate one clock run.

    Returns dict with arrays (true_time, reported_time, ref_error,
    fault_flag) and lists (injected_jumps, missing_indices).
    """
    rng      = np.random.default_rng(seed)
    dt       = phys["FRAME_INTERVAL"]
    n_frames = int(phys["DURATION_SEC"] / dt)

    # Jump frames: random, avoiding first/last 10%.
    if num_jumps > 0:
        jump_frames = set(rng.choice(
            np.arange(int(0.1 * n_frames), int(0.9 * n_frames)),
            size=num_jumps, replace=False).tolist())
    else:
        jump_frames = set()

    true_time     = np.zeros(n_frames)
    reported_time = np.zeros(n_frames)
    fault_flag    = np.zeros(n_frames, dtype=int)

    drift_rate        = drift_bias        # steady bias; wander accumulates on top
    accumulated_drift = 0.0
    jitter_state      = 0.0               # AR(1) colored-jitter state

    injected_jumps  = []
    missing_indices = []

    for i in range(n_frames):
        t_true       = i * dt
        true_time[i] = t_true

        # --- drift rate: steady bias + random-walk wander -------------
        drift_rate += rng.normal(0.0, phys["DRIFT_RATE_RW_SIGMA"])

        # --- temperature cycle: sinusoidal rate modulation ------------
        if phys["TEMP_CYCLE_ON"]:
            rate_now = drift_rate + phys["TEMP_AMPLITUDE"] * np.sin(
                2.0 * np.pi * t_true / phys["ORBIT_PERIOD_SEC"])
        else:
            rate_now = drift_rate

        # --- accumulate offset ----------------------------------------
        accumulated_drift += rate_now * dt

        # --- correlated (colored) readout jitter, AR(1) ---------------
        if phys["JITTER_ON"]:
            jitter_state = (phys["JITTER_RHO"] * jitter_state
                            + rng.normal(0.0, phys["JITTER_SIGMA"]))
            jitter = jitter_state
        else:
            jitter = rng.normal(0.0, phys["JITTER_SIGMA"])

        # --- jump fault -----------------------------------------------
        if i in jump_frames:
            mag  = rng.uniform(*phys["JUMP_SIZE_RANGE"])
            sign = rng.choice([-1.0, 1.0]) if phys["JUMP_SIGN_RANDOM"] else 1.0
            jump = sign * mag
            accumulated_drift += jump
            fault_flag[i] = 1
            injected_jumps.append((i, t_true, jump))

        # --- missing frame --------------------------------------------
        if fault_flag[i] == 0 and missing_prob > 0 and rng.random() < missing_prob:
            fault_flag[i] = 2
            missing_indices.append(i)

        # --- reported timestamp ---------------------------------------
        t_report = t_true + accumulated_drift + jitter

        # --- timestamp quantization (discrete clock ticks) ------------
        if phys["QUANTIZE_ON"]:
            q = phys["QUANTUM"]
            t_report = round(t_report / q) * q

        reported_time[i] = t_report

    ref_error = reported_time - true_time
    return dict(
        true_time=true_time, reported_time=reported_time,
        ref_error=ref_error, fault_flag=fault_flag,
        injected_jumps=injected_jumps, missing_indices=missing_indices,
    )


# =====================================================================
# CSV OUTPUT  (missing frames omitted, as on real hardware)
# =====================================================================

def write_csv(sim, path):
    """
    One row per TRANSMITTED frame. Missing frames are omitted; their
    anomaly survives as a doubled inter-frame gap on the next row.

    Columns:
        frame_index      original frame number
        true_time_s      ground truth (analysis only; not seen onboard)
        reported_time_s  the clock's reported timestamp
        ref_error_s      reported - true (analysis only)
        interframe_gap_s reported[i] - reported[i-1]  <-- detector feature
        fault_flag       0 normal, 1 jump, 2 missing
    """
    missing = set(sim["missing_indices"])
    prev = None
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "true_time_s", "reported_time_s",
                    "ref_error_s", "interframe_gap_s", "fault_flag"])
        for i in range(len(sim["true_time"])):
            if i in missing:
                continue
            tt = sim["true_time"][i]
            rt = sim["reported_time"][i]
            re = sim["ref_error"][i]
            ff = sim["fault_flag"][i]
            gap = "" if prev is None else f"{rt - prev:.6f}"
            w.writerow([i, f"{tt:.6f}", f"{rt:.6f}", f"{re:.6f}", gap, int(ff)])
            prev = rt
            rows += 1
    return rows


# =====================================================================
# DATASET GENERATION
# =====================================================================

def generate_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  GENERATING DATASET  ->  ./{DATASET_DIR}/")
    print(f"  temp_cycle={PHYSICS['TEMP_CYCLE_ON']}  "
          f"jitter={PHYSICS['JITTER_ON']}(rho={PHYSICS['JITTER_RHO']})  "
          f"quantize={PHYSICS['QUANTIZE_ON']}({PHYSICS['QUANTUM']*1000:.0f}ms)")
    print(f"{'='*60}")

    summary = []
    for (name, seed, n_jumps, miss_prob, drift_bias) in DATASET_FILES:
        sim = simulate(seed, n_jumps, miss_prob, drift_bias)
        path = os.path.join(DATASET_DIR, f"{name}.csv")
        rows = write_csv(sim, path)

        clean = (n_jumps == 0 and miss_prob == 0.0)
        kind  = "CLEAN" if clean else "FAULTY"
        n_j   = len(sim["injected_jumps"])
        n_m   = len(sim["missing_indices"])
        print(f"[{kind:6s}] {name:18s} seed={seed} "
              f"jumps={n_j:2d} missing={n_m:3d} -> {rows} rows")
        summary.append((name, kind, rows, n_j, n_m))

    # Manifest
    with open(os.path.join(DATASET_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "kind", "rows", "n_jumps", "n_missing", "role"])
        for (name, kind, rows, n_j, n_m) in summary:
            role = ("training" if name.startswith("train")
                    else "validation" if name.startswith("val") else "test")
            w.writerow([f"{name}.csv", kind, rows, n_j, n_m, role])

    print(f"{'='*60}")
    print(f"  Done. {len(DATASET_FILES)} files + manifest in ./{DATASET_DIR}/")


if __name__ == "__main__":
    generate_dataset()
