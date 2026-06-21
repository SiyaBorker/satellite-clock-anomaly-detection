"""
Temperature module.

Three modes:
    MANUAL  - returns the user-set value, ignores sim time
    RAMP    - T_start + rate * elapsed
    ORBITAL - T_mid + amplitude * sin(2*pi * elapsed / period)

`current(sim_elapsed_s)` returns the temperature for this tick.
The slider in the UI always reflects the resulting current value (read-only in
ramp/orbital modes), so the user can see what the profile is producing.
"""

from dataclasses import dataclass
from enum import Enum
import math


class TempMode(Enum):
    MANUAL = "manual"
    RAMP = "ramp"
    ORBITAL = "orbital"


@dataclass
class RampParams:
    t_start_c: float = 25.0
    rate_c_per_s: float = 0.1     # degrees per simulated second


@dataclass
class OrbitalParams:
    t_mid_c: float = 0.0
    amplitude_c: float = 60.0     # +/- this around t_mid
    period_s: float = 5400.0      # ~90 minute LEO orbit by default


class TemperatureModule:
    def __init__(self):
        self.mode = TempMode.MANUAL
        self._manual_value_c = 25.0
        self.ramp = RampParams()
        self.orbital = OrbitalParams()
        self._mode_entry_sim_s = 0.0     # sim time when current mode was entered

    def set_manual(self, value_c: float) -> None:
        self._manual_value_c = value_c

    def set_mode(self, mode: TempMode, sim_elapsed_s: float) -> None:
        """Switch mode; ramp/orbital are measured from this entry time."""
        self.mode = mode
        self._mode_entry_sim_s = sim_elapsed_s

    def current(self, sim_elapsed_s: float) -> float:
        if self.mode is TempMode.MANUAL:
            return self._manual_value_c
        elapsed = sim_elapsed_s - self._mode_entry_sim_s
        if self.mode is TempMode.RAMP:
            return self.ramp.t_start_c + self.ramp.rate_c_per_s * elapsed
        if self.mode is TempMode.ORBITAL:
            return self.orbital.t_mid_c + self.orbital.amplitude_c * \
                math.sin(2.0 * math.pi * elapsed / self.orbital.period_s)
        return self._manual_value_c
