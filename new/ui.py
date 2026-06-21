"""
Main window UI.

Layout:
    Top strip      : controls (temp slider+mode, speed, SPS, pause, reset)
    Left column    : numeric readouts (times, error, temp, SPS, offset comps)
    Right column   : three pyqtgraph plots (error, temperature, freq offset)

All state updates arrive via the worker's `tick` signal on the main thread.
"""

from collections import deque
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QComboBox, QFrame, QSizePolicy,
)
import pyqtgraph as pg

from temperature import TempMode


PLOT_BUFFER_SIZE = 1000

# Dark, restrained palette
COL_BG = "#11151a"
COL_PANEL = "#171c23"
COL_TEXT = "#dde3e6"
COL_MUTED = "#8893a1"
COL_REF = "#45c9b8"
COL_ERR = "#e98a5a"
COL_TEMP = "#e3a83f"
COL_OFFSET = "#7bd88f"


def _styled_label(text: str, *, size: int = 12, color: str = COL_TEXT,
                  mono: bool = False, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    weight = "500" if bold else "400"
    family = ('ui-monospace, "Cascadia Code", Consolas, monospace'
              if mono else "-apple-system, Segoe UI, sans-serif")
    lbl.setStyleSheet(
        f"color: {color}; font-size: {size}px; font-weight: {weight}; "
        f"font-family: {family};"
    )
    return lbl


def _make_plot(title: str, y_label: str, color: str) -> tuple:
    pw = pg.PlotWidget()
    pw.setBackground(COL_PANEL)
    pw.setTitle(title, color=COL_MUTED, size="10pt")
    pw.setLabel("left", y_label, color=COL_MUTED)
    pw.setLabel("bottom", "Sim time (s)", color=COL_MUTED)
    pw.showGrid(x=True, y=True, alpha=0.15)
    pw.getAxis("left").setTextPen(COL_MUTED)
    pw.getAxis("bottom").setTextPen(COL_MUTED)
    curve = pw.plot(pen=pg.mkPen(color=color, width=2))
    return pw, curve


class MainWindow(QMainWindow):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setWindowTitle("Onboard Clock Drift Simulator — Phase 1")
        self.resize(1180, 720)
        self.setStyleSheet(f"QMainWindow {{ background: {COL_BG}; }}")

        # Plot buffers
        self._t_buf = deque(maxlen=PLOT_BUFFER_SIZE)
        self._err_buf = deque(maxlen=PLOT_BUFFER_SIZE)
        self._temp_buf = deque(maxlen=PLOT_BUFFER_SIZE)
        self._offset_buf = deque(maxlen=PLOT_BUFFER_SIZE)

        self._suppress_temp_signal = False  # for read-only slider sync

        self._build_ui()
        self.worker.tick.connect(self._on_tick)

    # ------------- UI construction -------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        root.addWidget(self._build_controls())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_readouts(), stretch=1)
        body.addWidget(self._build_plots(), stretch=3)
        root.addLayout(body)

    def _build_controls(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border-radius: 8px; "
            f"border: 1px solid #2a323c; }}"
        )
        lay = QGridLayout(bar)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setHorizontalSpacing(20)

        # Temperature
        lay.addWidget(_styled_label("Temperature", color=COL_MUTED, size=11), 0, 0)
        self.temp_value_label = _styled_label("25.0 °C", mono=True, size=12)
        lay.addWidget(self.temp_value_label, 0, 1)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(-40, 85)
        self.temp_slider.setValue(25)
        self.temp_slider.setMinimumWidth(220)
        self.temp_slider.valueChanged.connect(self._on_temp_slider)
        lay.addWidget(self.temp_slider, 1, 0, 1, 2)

        # Mode
        lay.addWidget(_styled_label("Mode", color=COL_MUTED, size=11), 0, 2)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Ramp", "Orbital"])
        self.mode_combo.setStyleSheet(
            f"QComboBox {{ background: #1e252d; color: {COL_TEXT}; "
            f"padding: 4px 8px; border: 1px solid #2a323c; border-radius: 4px; }}"
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        lay.addWidget(self.mode_combo, 1, 2)

        # Speed
        lay.addWidget(_styled_label("Speed", color=COL_MUTED, size=11), 0, 3)
        self.speed_value_label = _styled_label("1×", mono=True, size=12)
        lay.addWidget(self.speed_value_label, 0, 4)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 1000)
        self.speed_slider.setValue(1)
        self.speed_slider.setMinimumWidth(180)
        self.speed_slider.valueChanged.connect(self._on_speed)
        lay.addWidget(self.speed_slider, 1, 3, 1, 2)

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)
        self.sps_btn = self._make_button("SPS: AVAILABLE", primary=True)
        self.sps_btn.clicked.connect(self._on_sps_toggle)
        btn_box.addWidget(self.sps_btn)
        self.pause_btn = self._make_button("Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        btn_box.addWidget(self.pause_btn)
        self.reset_btn = self._make_button("Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_box.addWidget(self.reset_btn)
        btn_widget = QWidget()
        btn_widget.setLayout(btn_box)
        lay.addWidget(btn_widget, 0, 5, 2, 1, alignment=Qt.AlignmentFlag.AlignRight)
        lay.setColumnStretch(5, 1)

        return bar

    def _make_button(self, text: str, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        bg = "rgba(69,201,184,0.12)" if primary else "#1e252d"
        border = COL_REF if primary else "#2a323c"
        fg = COL_REF if primary else COL_TEXT
        btn.setStyleSheet(
            f"QPushButton {{ background: {bg}; color: {fg}; "
            f"border: 1px solid {border}; border-radius: 5px; "
            f"padding: 6px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ border-color: {COL_MUTED}; }}"
        )
        return btn

    def _build_readouts(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border-radius: 8px; "
            f"border: 1px solid #2a323c; }}"
        )
        panel.setMinimumWidth(290)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        def row(label_text: str, color: str = COL_TEXT, size: int = 14) -> QLabel:
            v.addWidget(_styled_label(label_text.upper(), color=COL_MUTED,
                                      size=10, bold=True))
            val = _styled_label("—", mono=True, size=size, color=color)
            v.addWidget(val)
            return val

        self.ref_time_label = row("Reference time", COL_REF, size=15)
        self.obc_time_label = row("OBC time", COL_ERR, size=15)
        self.error_label    = row("Clock error", COL_ERR, size=15)
        self.temp_label     = row("Temperature", COL_TEMP, size=15)
        self.sps_label      = row("SPS status", COL_REF, size=13)

        v.addSpacing(6)
        v.addWidget(_styled_label("Frequency offset components",
                                  color=COL_MUTED, size=10, bold=True))

        comp_grid = QGridLayout()
        comp_grid.setHorizontalSpacing(8)
        comp_grid.setVerticalSpacing(4)
        def comp_row(r: int, name: str) -> QLabel:
            comp_grid.addWidget(
                _styled_label(name, color=COL_MUTED, size=11), r, 0
            )
            val = _styled_label("— ppm", mono=True, size=12)
            comp_grid.addWidget(val, r, 1, alignment=Qt.AlignmentFlag.AlignRight)
            return val
        self.total_ppm_label = comp_row(0, "Total")
        self.bias_ppm_label  = comp_row(1, "Bias")
        self.tcomp_ppm_label = comp_row(2, "Temperature")
        self.aging_ppm_label = comp_row(3, "Aging")
        self.rw_ppm_label    = comp_row(4, "Random walk")
        v.addLayout(comp_grid)
        v.addStretch(1)

        return panel

    def _build_plots(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border-radius: 8px; "
            f"border: 1px solid #2a323c; }}"
        )
        v = QVBoxLayout(wrap)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        self.err_plot,    self.err_curve    = _make_plot(
            "Clock error", "ms", COL_ERR
        )
        self.temp_plot,   self.temp_curve   = _make_plot(
            "Temperature", "°C", COL_TEMP
        )
        self.offset_plot, self.offset_curve = _make_plot(
            "Frequency offset", "ppm", COL_OFFSET
        )
        for p in (self.err_plot, self.temp_plot, self.offset_plot):
            p.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)
            v.addWidget(p)
        return wrap

    # ------------- control handlers -------------
    def _on_temp_slider(self, value: int) -> None:
        if self._suppress_temp_signal:
            return
        # Slider drives manual value only when in manual mode
        if self.mode_combo.currentIndex() == 0:
            self.worker.set_manual_temp(float(value))
        self.temp_value_label.setText(f"{value:.1f} °C")

    def _on_speed(self, value: int) -> None:
        self.worker.set_speed(float(value))
        self.speed_value_label.setText(f"{value}×")

    def _on_mode(self, idx: int) -> None:
        mode = [TempMode.MANUAL, TempMode.RAMP, TempMode.ORBITAL][idx]
        self.worker.set_temp_mode(mode)
        self.temp_slider.setEnabled(mode is TempMode.MANUAL)

    def _on_sps_toggle(self) -> None:
        available = self.worker.toggle_sps()
        self.sps_btn.setText(f"SPS: {'AVAILABLE' if available else 'UNAVAILABLE'}")

    def _on_pause(self) -> None:
        if self.pause_btn.text() == "Pause":
            self.worker.pause()
            self.pause_btn.setText("Resume")
        else:
            self.worker.resume()
            self.pause_btn.setText("Pause")

    def _on_reset(self) -> None:
        # Worker-side: zero sim time, error, random-walk state
        self.worker.reset()

        # Plot buffers
        self._t_buf.clear()
        self._err_buf.clear()
        self._temp_buf.clear()
        self._offset_buf.clear()
        for curve in (self.err_curve, self.temp_curve, self.offset_curve):
            curve.setData([], [])

        # Controls back to defaults. Setting these fires their valueChanged /
        # currentIndexChanged signals which in turn push the values into the
        # worker — so the worker's temperature, speed, and mode all sync up.
        self.mode_combo.setCurrentIndex(0)        # Manual
        self.temp_slider.setEnabled(True)
        self.temp_slider.setValue(25)             # 25 °C
        self.speed_slider.setValue(1)             # 1×

        # SPS and Pause: force state explicitly and update button labels
        self.worker.set_sps(True)
        self.sps_btn.setText("SPS: AVAILABLE")
        self.worker.resume()
        self.pause_btn.setText("Pause")

    # ------------- tick handler (main thread) -------------
    @pyqtSlot(dict)
    def _on_tick(self, snap: dict) -> None:
        ref = snap["ref_now"]
        obc = snap["obc_now"]
        ref_str = ref.strftime("%H:%M:%S.") + f"{ref.microsecond // 1000:03d}"
        obc_str = obc.strftime("%H:%M:%S.") + f"{obc.microsecond // 1000:03d}"
        self.ref_time_label.setText(ref_str)
        self.obc_time_label.setText(obc_str)

        err_ms = snap["error_ms"]
        if abs(err_ms) < 1000:
            self.error_label.setText(f"{err_ms:+.2f} ms")
        else:
            self.error_label.setText(f"{err_ms/1000:+.3f} s")

        temp = snap["temperature_c"]
        self.temp_label.setText(f"{temp:+.2f} °C")

        # Keep slider in sync when ramp/orbital is driving temperature
        if snap["temp_mode"] is not TempMode.MANUAL:
            self._suppress_temp_signal = True
            clamped = max(self.temp_slider.minimum(),
                          min(self.temp_slider.maximum(), int(round(temp))))
            self.temp_slider.setValue(clamped)
            self.temp_value_label.setText(f"{temp:+.1f} °C")
            self._suppress_temp_signal = False

        self.sps_label.setText("AVAILABLE" if snap["sps_available"]
                               else "UNAVAILABLE")

        off = snap["offsets_ppm"]
        self.total_ppm_label.setText(f"{off['total_ppm']:+.3f} ppm")
        self.bias_ppm_label .setText(f"{off['bias_ppm']:+.3f} ppm")
        self.tcomp_ppm_label.setText(f"{off['temp_ppm']:+.3f} ppm")
        self.aging_ppm_label.setText(f"{off['aging_ppm']:+.4f} ppm")
        self.rw_ppm_label   .setText(f"{off['rw_ppm']:+.4f} ppm")

        t = snap["sim_elapsed_s"]
        self._t_buf.append(t)
        self._err_buf.append(err_ms)
        self._temp_buf.append(temp)
        self._offset_buf.append(off["total_ppm"])
        # pyqtgraph's setData wants sequences; deque is fine
        ts = list(self._t_buf)
        self.err_curve.setData(ts, list(self._err_buf))
        self.temp_curve.setData(ts, list(self._temp_buf))
        self.offset_curve.setData(ts, list(self._offset_buf))
