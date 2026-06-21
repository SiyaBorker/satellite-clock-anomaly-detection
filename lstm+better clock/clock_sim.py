"""
clock_sim.py  --  Spacecraft Clock Simulator (dataset generator)
================================================================
Generates the ML dataset: a stream of reported timestamps from a drifting
spacecraft clock, with faults injected at known positions.

Normal-clock physics (each toggleable in PHYSICS)
-------------------------------------------------
  - steady drift + random-walk wander : oscillator rate is imperfect and
    wanders slowly.
  - temperature cycle                 : clock RATE varies sinusoidally with
    the orbital thermal cycle (sunlight/eclipse). Smooth, learnable.
  - correlated (colored) jitter       : per-frame timing noise is AR(1)
    (each sample depends on the last), like real oscillator noise.
  - timestamp quantization            : optional discrete-tick rounding.

Fault types (each is physically grounded)
-----------------------------------------
  flag 1  jump            : sudden step in the clock offset (radiation upset,
                            counter glitch). Instantaneous.
  flag 2  missing frame   : a frame is dropped (telemetry loss). Shows up as
                            a doubled inter-frame gap on the next row.
  flag 3  slow-onset jump : a step smeared over N frames (a ramp). Models a
                            glitch that develops over several seconds rather
                            than instantly. Threshold-hard, pattern-shaped.
  flag 4  oscillatory     : a transient sinusoidal wobble in the offset over
                            a burst of frames (resonance / thermal transient
                            / control-loop instability). Pure pattern fault.
  flag 5  frequency step  : the drift RATE changes abruptly and stays changed
                            (oscillator mode change / ageing step). Between a
                            jump and gradual drift.

Output
------
  ./dataset/*.csv          one row per TRANSMITTED frame
  ./dataset/manifest.csv   index of files + roles

Run
---
    pip install numpy
    python clock_sim.py
"""

import os
import csv
import numpy as np

# =====================================================================
# PHYSICS  --  the normal-clock model shared by every file
# =====================================================================

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
    JITTER_RHO    = 0.95,        # AR(1) memory: 0 = white, ->1 = slow wander
    JITTER_SIGMA  = 1e-4,        # innovation std of the jitter process (s)

    # --- timestamp quantization --------------------------------------
    QUANTIZE_ON  = False,
    QUANTUM      = 1e-3,         # clock reports in 1 ms ticks (when ON)

    # --- fault magnitudes (realistic ranges) -------------------------
    JUMP_SIZE_RANGE      = (0.020, 0.150),  # instantaneous jump (s)
    JUMP_SIGN_RANDOM     = True,

    ONSET_SIZE_RANGE     = (0.020, 0.150),  # slow-onset total step (s)
    ONSET_FRAMES_RANGE   = (5, 30),         # frames over which it ramps

    OSC_AMP_RANGE        = (0.005, 0.030),  # oscillation amplitude (s)
    OSC_FRAMES_RANGE     = (20, 60),        # burst length (frames)
    OSC_CYCLES_RANGE     = (2, 5),          # full sine cycles in the burst

    FREQSTEP_RANGE       = (1e-6, 5e-6),    # rate change magnitude (s/s)
    FREQSTEP_SIGN_RANDOM = True,
)

DATASET_DIR = "dataset"

# Per-file fault spec. Each entry:
#   (filename, seed, drift_bias, faults_dict)
# faults_dict keys (all optional, default 0 / 0.0):
#   n_jumps, missing_prob, n_onset, n_osc, n_freqstep
# Clean files have an empty faults dict.
DATASET_FILES = [
    # ---- CLEAN : training / validation ------------------------------
    ("train_normal",      101, +2e-6, {}),
    ("val_normal",        202, -1e-6, {}),

    # ---- TEST : single fault type, many events (statistical power) ---
    ("test_jumps",        301, +1e-6, dict(n_jumps=40)),
    ("test_missing",      302, -2e-6, dict(missing_prob=0.0015)),
    ("test_onset",        303, +1e-6, dict(n_onset=40)),   # slow-onset jumps
    ("test_oscillatory",  304, -1e-6, dict(n_osc=40)),     # sinusoidal wobble
    ("test_freqstep",     305, +1e-6, dict(n_freqstep=20)),# rate steps

    # ---- TEST : mixed, realistic combination ------------------------
    ("test_mixed_1",      306, +3e-6, dict(n_jumps=15, missing_prob=0.0008,
                                           n_onset=10, n_osc=10)),
    ("test_mixed_2",      307, -3e-6, dict(n_jumps=10, missing_prob=0.0010,
                                           n_onset=10, n_osc=10, n_freqstep=5)),
]


# =====================================================================
# SIMULATION
# =====================================================================

def _pick_frames(rng, n_frames, count, used, margin=0.1, spacing=80):
    """
    Pick `count` start-frames in [margin, 1-margin] of the run that do not
    collide with already-used frames (keeps faults from overlapping so each
    stays a clean, labelled event).
    """
    lo, hi = int(margin * n_frames), int((1 - margin) * n_frames)
    chosen = []
    attempts = 0
    while len(chosen) < count and attempts < count * 200:
        attempts += 1
        f = int(rng.integers(lo, hi))
        if all(abs(f - u) >= spacing for u in used) and \
           all(abs(f - c) >= spacing for c in chosen):
            chosen.append(f)
    return sorted(chosen)


def simulate(seed, drift_bias, faults, phys=PHYSICS):
    """
    Simulate one clock run.

    `faults` is a dict with any of:
        n_jumps, missing_prob, n_onset, n_osc, n_freqstep
    Returns dict of arrays + lists of injected events (for reporting).
    """
    rng      = np.random.default_rng(seed)
    dt       = phys["FRAME_INTERVAL"]
    n_frames = int(phys["DURATION_SEC"] / dt)

    n_jumps      = faults.get("n_jumps", 0)
    missing_prob = faults.get("missing_prob", 0.0)
    n_onset      = faults.get("n_onset", 0)
    n_osc        = faults.get("n_osc", 0)
    n_freqstep   = faults.get("n_freqstep", 0)

    used = []  # frames already claimed by a fault (avoid overlap)

    # --- pre-plan the structured faults (need start frames) ----------
    jump_frames = _pick_frames(rng, n_frames, n_jumps, used); used += jump_frames

    # onset: (start, total_step, ramp_frames)
    onset_starts = _pick_frames(rng, n_frames, n_onset, used); used += onset_starts
    onset_events = []
    for s in onset_starts:
        mag  = rng.uniform(*phys["ONSET_SIZE_RANGE"])
        sign = rng.choice([-1.0, 1.0])
        nf   = int(rng.integers(*phys["ONSET_FRAMES_RANGE"]))
        onset_events.append((s, sign * mag, nf))

    # oscillatory: (start, amp, burst_frames, cycles)
    osc_starts = _pick_frames(rng, n_frames, n_osc, used); used += osc_starts
    osc_events = []
    for s in osc_starts:
        amp    = rng.uniform(*phys["OSC_AMP_RANGE"])
        nf     = int(rng.integers(*phys["OSC_FRAMES_RANGE"]))
        cycles = int(rng.integers(*phys["OSC_CYCLES_RANGE"]))
        osc_events.append((s, amp, nf, cycles))

    # frequency step: (frame, rate_delta)
    fs_starts = _pick_frames(rng, n_frames, n_freqstep, used); used += fs_starts
    fs_events = []
    for s in fs_starts:
        mag  = rng.uniform(*phys["FREQSTEP_RANGE"])
        sign = rng.choice([-1.0, 1.0]) if phys["FREQSTEP_SIGN_RANDOM"] else 1.0
        fs_events.append((s, sign * mag))
    fs_map = dict(fs_events)

    # Build per-frame additive contributions for onset & oscillatory so the
    # main loop can just look them up.
    onset_add = np.zeros(n_frames)   # cumulative offset added by onsets
    onset_flag = np.zeros(n_frames, dtype=bool)
    for (s, total, nf) in onset_events:
        for k in range(nf):
            idx = s + k
            if idx >= n_frames:
                break
            onset_add[idx:] += total / nf      # ramp: add a slice each frame
            onset_flag[idx] = True

    osc_add = np.zeros(n_frames)
    osc_flag = np.zeros(n_frames, dtype=bool)
    for (s, amp, nf, cycles) in osc_events:
        for k in range(nf):
            idx = s + k
            if idx >= n_frames:
                break
            osc_add[idx] += amp * np.sin(2.0 * np.pi * cycles * k / nf)
            osc_flag[idx] = True

    true_time     = np.zeros(n_frames)
    reported_time = np.zeros(n_frames)
    fault_flag    = np.zeros(n_frames, dtype=int)

    drift_rate        = drift_bias
    accumulated_drift = 0.0
    jitter_state      = 0.0

    injected = dict(jumps=[], onset=onset_events, osc=osc_events,
                    freqstep=fs_events, missing=[])

    for i in range(n_frames):
        t_true       = i * dt
        true_time[i] = t_true

        # --- drift rate: steady bias + random-walk wander -------------
        drift_rate += rng.normal(0.0, phys["DRIFT_RATE_RW_SIGMA"])

        # --- frequency-step fault: permanent rate change --------------
        if i in fs_map:
            drift_rate += fs_map[i]
            fault_flag[i] = 5

        # --- temperature cycle ----------------------------------------
        if phys["TEMP_CYCLE_ON"]:
            rate_now = drift_rate + phys["TEMP_AMPLITUDE"] * np.sin(
                2.0 * np.pi * t_true / phys["ORBIT_PERIOD_SEC"])
        else:
            rate_now = drift_rate

        accumulated_drift += rate_now * dt

        # --- colored jitter (AR1) -------------------------------------
        if phys["JITTER_ON"]:
            jitter_state = (phys["JITTER_RHO"] * jitter_state
                            + rng.normal(0.0, phys["JITTER_SIGMA"]))
            jitter = jitter_state
        else:
            jitter = rng.normal(0.0, phys["JITTER_SIGMA"])

        # --- instantaneous jump ---------------------------------------
        if i in jump_frames:
            mag  = rng.uniform(*phys["JUMP_SIZE_RANGE"])
            sign = rng.choice([-1.0, 1.0]) if phys["JUMP_SIGN_RANDOM"] else 1.0
            accumulated_drift += sign * mag
            fault_flag[i] = 1
            injected["jumps"].append((i, sign * mag))

        # --- slow-onset & oscillatory (label if not already faulted) --
        if onset_flag[i] and fault_flag[i] == 0:
            fault_flag[i] = 3
        if osc_flag[i] and fault_flag[i] == 0:
            fault_flag[i] = 4

        # --- missing frame --------------------------------------------
        if fault_flag[i] == 0 and missing_prob > 0 and rng.random() < missing_prob:
            fault_flag[i] = 2
            injected["missing"].append(i)

        # --- reported timestamp ---------------------------------------
        t_report = (t_true + accumulated_drift + jitter
                    + onset_add[i] + osc_add[i])

        if phys["QUANTIZE_ON"]:
            q = phys["QUANTUM"]
            t_report = round(t_report / q) * q

        reported_time[i] = t_report

    ref_error = reported_time - true_time
    missing_indices = injected["missing"]
    return dict(
        true_time=true_time, reported_time=reported_time,
        ref_error=ref_error, fault_flag=fault_flag,
        injected=injected, missing_indices=missing_indices,
    )


# =====================================================================
# CSV OUTPUT  (missing frames omitted, as on real hardware)
# =====================================================================

def write_csv(sim, path):
    """
    One row per TRANSMITTED frame. Missing frames omitted; their anomaly
    survives as a doubled inter-frame gap on the next row.

    Columns:
        frame_index, true_time_s, reported_time_s, ref_error_s,
        interframe_gap_s, fault_flag
    fault_flag: 0 normal, 1 jump, 2 missing, 3 slow-onset, 4 oscillatory,
                5 frequency-step
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

FAULT_NAMES = {1: "jump", 2: "missing", 3: "onset", 4: "oscillatory",
               5: "freqstep"}


def generate_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"{'='*64}")
    print(f"  GENERATING DATASET  ->  ./{DATASET_DIR}/")
    print(f"  temp_cycle={PHYSICS['TEMP_CYCLE_ON']}  "
          f"jitter={PHYSICS['JITTER_ON']}(rho={PHYSICS['JITTER_RHO']})  "
          f"quantize={PHYSICS['QUANTIZE_ON']}")
    print(f"{'='*64}")

    summary = []
    for (name, seed, drift_bias, faults) in DATASET_FILES:
        sim = simulate(seed, drift_bias, faults)
        path = os.path.join(DATASET_DIR, f"{name}.csv")
        rows = write_csv(sim, path)

        # count faults by type
        ff = sim["fault_flag"]
        counts = {k: int((ff == k).sum()) for k in FAULT_NAMES}
        clean = sum(counts.values()) == 0
        kind = "CLEAN" if clean else "FAULTY"
        desc = " ".join(f"{FAULT_NAMES[k]}={counts[k]}"
                        for k in FAULT_NAMES if counts[k] > 0) or "none"
        print(f"[{kind:6s}] {name:16s} seed={seed} -> {rows} rows | {desc}")
        summary.append((name, kind, rows, counts))

    # Manifest
    with open(os.path.join(DATASET_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "kind", "rows", "jump", "missing", "onset",
                    "oscillatory", "freqstep", "role"])
        for (name, kind, rows, counts) in summary:
            role = ("training" if name.startswith("train")
                    else "validation" if name.startswith("val") else "test")
            w.writerow([f"{name}.csv", kind, rows, counts[1], counts[2],
                        counts[3], counts[4], counts[5], role])

    print(f"{'='*64}")
    print(f"  Done. {len(DATASET_FILES)} files + manifest in ./{DATASET_DIR}/")


if __name__ == "__main__":
    generate_dataset()
