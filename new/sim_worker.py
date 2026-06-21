"""
Simulation worker.

Owns the model objects and runs the tick loop in its own QThread. Emits a
`tick` signal each step carrying a snapshot dict the UI thread renders. The
UI never touches model state directly; it asks the worker via thread-safe
setters (which use a lock).

Speed control uses dt scaling: the timer fires at a fixed wall-clock interval
and each tick advances sim time by `tick_interval * speed_multiplier`.
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QMutex, QMutexLocker
import time

from reference_clock import ReferenceClock
from oscillator import Oscillator, OscillatorParams
from obc_clock import OBCClock
from temperature import TemperatureModule, TempMode
from sps_status import SPSStatus


TICK_INTERVAL_S = 0.1   # wall-clock seconds between ticks


class SimulationWorker(QObject):
    tick = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.reference = ReferenceClock()
        self.oscillator = Oscillator(OscillatorParams())
        self.obc = OBCClock()
        self.temperature = TemperatureModule()
        self.sps = SPSStatus()

        self._mutex = QMutex()
        self._running = True       # paused vs running
        self._stop = False         # thread shutdown
        self._speed_multiplier = 1.0

    # ---------- thread-safe setters called by UI ----------
    def set_speed(self, multiplier: float) -> None:
        with QMutexLocker(self._mutex):
            self._speed_multiplier = max(0.1, float(multiplier))

    def set_manual_temp(self, value_c: float) -> None:
        with QMutexLocker(self._mutex):
            self.temperature.set_manual(value_c)

    def set_temp_mode(self, mode: TempMode) -> None:
        with QMutexLocker(self._mutex):
            self.temperature.set_mode(mode, self.reference.sim_elapsed_s)

    def toggle_sps(self) -> bool:
        with QMutexLocker(self._mutex):
            return self.sps.toggle()

    def set_sps(self, available: bool) -> None:
        with QMutexLocker(self._mutex):
            self.sps.set(available)

    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False

    def resume(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = True

    def reset(self) -> None:
        with QMutexLocker(self._mutex):
            self.reference.reset()
            self.oscillator.reset()
            self.obc.reset()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stop = True

    # ---------- main loop, runs on the worker thread ----------
    def run(self) -> None:
        next_fire = time.monotonic()
        while True:
            with QMutexLocker(self._mutex):
                if self._stop:
                    return
                running = self._running
                speed = self._speed_multiplier

            if running:
                dt_sim = TICK_INTERVAL_S * speed
                self.reference.advance(dt_sim)
                temp_c = self.temperature.current(self.reference.sim_elapsed_s)
                offsets = self.oscillator.step(
                    temp_c, self.reference.sim_elapsed_s, dt_sim
                )
                self.obc.integrate(offsets["total_ppm"], dt_sim)

                ref_now = self.reference.now
                obc_now = self.obc.time_at(ref_now)

                self.tick.emit({
                    "sim_elapsed_s": self.reference.sim_elapsed_s,
                    "ref_now": ref_now,
                    "obc_now": obc_now,
                    "error_ms": self.obc.accumulated_error_ms,
                    "temperature_c": temp_c,
                    "offsets_ppm": offsets,
                    "sps_available": self.sps.available,
                    "temp_mode": self.temperature.mode,
                    "speed_multiplier": speed,
                })

            # Sleep until the next scheduled fire (wall clock)
            next_fire += TICK_INTERVAL_S
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We fell behind; resync rather than firing back-to-back
                next_fire = time.monotonic()
