"""Reference clock: ground-truth time. No drift, no noise."""

from datetime import datetime, timedelta


class ReferenceClock:
    def __init__(self):
        self._start_wall = datetime.now()
        self._sim_elapsed_s = 0.0

    def advance(self, dt_sim_s: float) -> None:
        """Advance simulated time by dt_sim_s (independent of wall clock)."""
        self._sim_elapsed_s += dt_sim_s

    def reset(self) -> None:
        self._start_wall = datetime.now()
        self._sim_elapsed_s = 0.0

    @property
    def sim_elapsed_s(self) -> float:
        return self._sim_elapsed_s

    @property
    def now(self) -> datetime:
        """Reference 'wall' time = start + simulated elapsed."""
        return self._start_wall + timedelta(seconds=self._sim_elapsed_s)
