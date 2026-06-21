"""OBC clock: accumulates frequency-offset integral as time error."""

from datetime import datetime, timedelta


class OBCClock:
    def __init__(self):
        self._accumulated_error_s = 0.0

    def integrate(self, offset_ppm: float, dt_sim_s: float) -> None:
        """Add the time-error contribution of `offset_ppm` over `dt_sim_s`."""
        # offset_ppm * 1e-6 = fractional frequency error (sec of error per sec of run)
        self._accumulated_error_s += offset_ppm * 1e-6 * dt_sim_s

    def reset(self) -> None:
        self._accumulated_error_s = 0.0

    @property
    def accumulated_error_s(self) -> float:
        return self._accumulated_error_s

    @property
    def accumulated_error_ms(self) -> float:
        return self._accumulated_error_s * 1000.0

    def time_at(self, ref_now: datetime) -> datetime:
        """Return OBC time given the current reference time."""
        return ref_now + timedelta(seconds=self._accumulated_error_s)
