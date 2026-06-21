"""
Oscillator model.

Produces a frequency offset in ppm composed of four independent components:
    bias       : constant manufacturing offset
    temperature: parabolic curve around turnover temperature
    aging      : linear drift in ppm per day
    random_walk: stateful Gaussian-increment noise

`step()` advances the random-walk state by one tick of size dt_sim_s and
returns a dict with all four components plus the total. Components are
returned separately so the UI can show what's driving the drift.
"""

import random
from dataclasses import dataclass


@dataclass
class OscillatorParams:
    constant_bias_ppm: float = 2.0
    turnover_temp_c: float = 25.0
    temp_coeff_k: float = 0.04          # ppm per (deg C)^2 away from turnover
    aging_ppm_per_day: float = 0.5
    rw_step_sigma_ppm: float = 0.02     # Gaussian sigma of per-tick RW increment


class Oscillator:
    def __init__(self, params: OscillatorParams | None = None):
        self.params = params or OscillatorParams()
        self._rw_offset_ppm = 0.0

    def reset(self) -> None:
        self._rw_offset_ppm = 0.0

    def step(self, temperature_c: float, sim_elapsed_s: float,
             dt_sim_s: float) -> dict:
        """
        Advance the random-walk state by one tick, then return the four
        component offsets and their total.
        """
        p = self.params

        bias = p.constant_bias_ppm

        dT = temperature_c - p.turnover_temp_c
        temp = -p.temp_coeff_k * dT * dT

        aging = p.aging_ppm_per_day * (sim_elapsed_s / 86400.0)

        # The RW increment is per *tick*, not per second. Scale sigma with
        # sqrt(dt) so the random-walk variance grows linearly with sim time
        # regardless of how big a sim step we take this tick.
        rw_increment = random.gauss(0.0, p.rw_step_sigma_ppm) * (dt_sim_s ** 0.5)
        self._rw_offset_ppm += rw_increment
        rw = self._rw_offset_ppm

        total = bias + temp + aging + rw

        return {
            "bias_ppm": bias,
            "temp_ppm": temp,
            "aging_ppm": aging,
            "rw_ppm": rw,
            "total_ppm": total,
        }
