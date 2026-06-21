"""SPS status: just a boolean flag for this phase, no correction logic."""


class SPSStatus:
    def __init__(self):
        self._available = True

    def toggle(self) -> bool:
        self._available = not self._available
        return self._available

    def set(self, available: bool) -> None:
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    @property
    def label(self) -> str:
        return "AVAILABLE" if self._available else "UNAVAILABLE"
