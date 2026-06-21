# Phase 1 — Onboard Clock Drift Simulator

Real-time simulator. Reference clock vs. drifting OBC clock, with four
oscillator error sources (bias, temperature, aging, random walk), live
temperature control (manual / ramp / orbital), variable simulation speed,
and an SPS-status toggle.

## Install

```
pip install PyQt6 pyqtgraph
```

## Run

```
python main.py
```

## Architecture

- `reference_clock.py` — ground-truth time
- `oscillator.py` — four error components, returns ppm offset
- `obc_clock.py` — integrates offset into accumulated error
- `temperature.py` — Manual / Ramp / Orbital modes
- `sps_status.py` — display-only flag this phase
- `sim_worker.py` — model + tick loop, runs in its own `QThread`,
  emits a `tick(dict)` signal each step
- `ui.py` — PyQt6 window: controls strip, readouts panel, three plots
- `main.py` — entry point; wires worker thread + window

The worker thread owns all model state and is the only writer. The UI thread
reads snapshots from `tick` signals and pushes control commands back through
mutex-protected setters.

## Controls

- **Temperature slider**: −40 to +85 °C. Drives the value in Manual mode;
  in Ramp/Orbital modes it becomes a read-only live indicator of the
  computed temperature.
- **Mode dropdown**: Manual / Ramp / Orbital. Ramp and orbital parameters
  are constants in `temperature.py` (edit `RampParams` / `OrbitalParams`
  to change rate, period, amplitude).
- **Speed slider**: 1× to 1000×. Uses dt scaling — timer fires every 100 ms
  wall-clock; each tick advances sim time by `0.1 × speed` seconds.
- **SPS button**: toggles status flag. No correction yet.
- **Pause / Reset**: pause freezes time; reset zeros sim time, error, and
  random-walk state but preserves oscillator and temperature parameters.

## Tuning oscillator behavior

Edit defaults in `oscillator.py` (`OscillatorParams`):

- `constant_bias_ppm` — fixed manufacturing offset
- `turnover_temp_c` — apex of the parabolic temperature curve
- `temp_coeff_k` — ppm per (°C)² away from turnover
- `aging_ppm_per_day` — long-term linear drift
- `rw_step_sigma_ppm` — Gaussian sigma of the per-tick RW increment
  (scaled internally by √dt so RW variance grows linearly in sim time)

Here's the full setup, step by step. I'll assume you're on Windows since that's most common for this kind of work — call out if you're on Mac or Linux and I'll adjust.

**1. Get the files into one folder**

Download all the files I shared and put them in a single folder. Something like:

```
C:\Users\YourName\Documents\clock_sim\
    main.py
    sim_worker.py
    ui.py
    oscillator.py
    reference_clock.py
    obc_clock.py
    temperature.py
    sps_status.py
    README.md
```

The exact path doesn't matter, but all eight `.py` files must sit in the same folder — they import each other by filename.

**2. Check Python is installed**

Open a terminal (on Windows: press Win+R, type `cmd`, hit Enter). Type:

```
python --version
```

You should see Python 3.10 or higher. If you get "not recognized" or a version below 3.10, install Python 3.11 or 3.12 from python.org first — and during install, **tick the "Add Python to PATH" checkbox**, that one's easy to miss.

**3. Navigate to the folder**

In the terminal:

```
cd C:\Users\YourName\Documents\clock_sim
```

(Use your actual path. Tip: on Windows you can drag the folder onto the terminal window to paste its path.)

**4. Create a virtual environment**

A virtual environment is an isolated Python installation just for this project — it keeps PyQt6 and pyqtgraph from polluting your system Python or clashing with other projects. From inside the `clock_sim` folder:

```
python -m venv venv
```

This creates a `venv` subfolder containing a private Python. Takes about 10 seconds.

**5. Activate the environment**

On Windows (cmd):
```
venv\Scripts\activate
```

On Windows (PowerShell):
```
venv\Scripts\Activate.ps1
```

On Mac/Linux:
```
source venv/bin/activate
```

You'll know it worked because your prompt now has `(venv)` at the front:

```
(venv) C:\Users\YourName\Documents\clock_sim>
```

If PowerShell complains about script execution being disabled, run this once and try again: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

**6. Install the two libraries**

```
pip install PyQt6 pyqtgraph
```

This pulls in PyQt6 (the GUI framework, ~50 MB) and pyqtgraph (the live plotting library, much smaller, depends on numpy which it'll grab automatically). Takes a minute or two depending on your connection.

If it succeeds, run `pip list` and you should see PyQt6, pyqtgraph, and numpy in the list.

**7. Run the simulator**

Still in the same terminal, with `(venv)` showing:

```
python main.py
```

**Where the output shows up**

A window opens on your desktop — that *is* the output. There's no terminal output by design (the terminal stays empty while the sim runs). The window has:

- A control strip across the top: temperature slider, mode dropdown, speed slider, SPS button, Pause, Reset.
- A readouts panel on the left: reference time, OBC time, error, temperature, SPS status, and the four frequency-offset components.
- Three live plots stacked on the right: clock error, temperature, frequency offset.

Numbers and plots update ten times a second. Move the temperature slider and watch the temperature plot respond immediately; the error plot will start bending within a second or two.

To close: just close the window (X button), or press Ctrl+C in the terminal.

**Next time you want to run it**

You only do steps 4 and 6 once. After that, every session is just:

```
cd C:\Users\YourName\Documents\clock_sim
venv\Scripts\activate
python main.py
```

**If something goes wrong**

- `ModuleNotFoundError: No module named 'PyQt6'` — you forgot to activate the venv, or you installed the libraries into a different Python. Re-activate (step 5) and reinstall (step 6).
- Window opens but plots are blank — give it a couple of seconds, the first ~10 ticks fill the buffer. If still blank, check the terminal for tracebacks.
- Window doesn't open at all and the terminal shows an error mentioning `xcb` or `Qt platform plugin` (Linux only) — install: `sudo apt install libxcb-cursor0 libxcb-xinerama0`.
- On Mac, if PyQt6 install fails — make sure you're on Python 3.10+ and try `pip install --upgrade pip` first.

Tell me what you see when you run it and I can help debug from there.