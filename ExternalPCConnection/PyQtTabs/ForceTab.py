import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt


class ForceTab(QWidget):
    """Live force sensor plot with an adjustable thumb/pinch threshold overlay."""

    SENSOR_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def __init__(self, thresholds, window_size=500, parent=None):
        super().__init__(parent)
        self.thresholds = thresholds
        self.window_size = window_size
        self.data_buffer = np.zeros((self.window_size, 5))
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.plot = pg.PlotWidget(title="Force Sensors (Analog Read)")
        self.plot.addLegend()
        self.curves = [self.plot.plot(pen=(i, 5), name=n) for i, n in enumerate(self.SENSOR_NAMES)]

        self.thumb_thresh_line = pg.InfiniteLine(
            pos=self.thresholds.force_thumb, angle=0,
            pen=pg.mkPen('r', style=Qt.PenStyle.DashLine),
            label="Thumb Threshold", labelOpts={'color': 'r', 'position': 0.95})
        self.pinch_thresh_line = pg.InfiniteLine(
            pos=self.thresholds.force_pinch, angle=0,
            pen=pg.mkPen('y', style=Qt.PenStyle.DashLine),
            label="Pinch Threshold (Idx+Mid sum)", labelOpts={'color': 'y', 'position': 0.85})
        self.plot.addItem(self.thumb_thresh_line)
        self.plot.addItem(self.pinch_thresh_line)
        layout.addWidget(self.plot)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Thumb Thresh:"))
        sld_thumb = QSlider(Qt.Orientation.Horizontal)
        sld_thumb.setRange(0, 1000)
        sld_thumb.setValue(int(self.thresholds.force_thumb))
        sld_thumb.valueChanged.connect(self._on_thumb_slider)
        slider_row.addWidget(sld_thumb)

        slider_row.addWidget(QLabel("Pinch Thresh:"))
        sld_pinch = QSlider(Qt.Orientation.Horizontal)
        sld_pinch.setRange(0, 1000)
        sld_pinch.setValue(int(self.thresholds.force_pinch))
        sld_pinch.valueChanged.connect(self._on_pinch_slider)
        slider_row.addWidget(sld_pinch)
        layout.addLayout(slider_row)

    def _on_thumb_slider(self, value):
        self.thresholds.force_thumb = value
        self.thumb_thresh_line.setPos(value)

    def _on_pinch_slider(self, value):
        self.thresholds.force_pinch = value
        self.pinch_thresh_line.setPos(value)

    def on_new_data(self, force_values):
        """force_values: iterable of 5 floats (thumb, index, middle, ring, pinky)."""
        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
        self.data_buffer[-1, :] = force_values
        for i in range(5):
            self.curves[i].setData(self.data_buffer[:, i])