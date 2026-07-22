import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt


class FlexTab(QWidget):
    """Live flex sensor plot with an adjustable 'bent' threshold overlay."""

    SENSOR_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def __init__(self, thresholds, window_size=500, parent=None):
        super().__init__(parent)
        self.thresholds = thresholds
        self.window_size = window_size
        self.data_buffer = np.zeros((self.window_size, 5))
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.plot = pg.PlotWidget(title="Flex Sensors (Analog Read)")
        self.plot.addLegend()
        self.curves = [self.plot.plot(pen=(i, 5), name=n) for i, n in enumerate(self.SENSOR_NAMES)]

        self.bent_thresh_line = pg.InfiniteLine(
            pos=self.thresholds.flex_bent, angle=0,
            pen=pg.mkPen('r', style=Qt.PenStyle.DashLine),
            label="Bent Threshold", labelOpts={'color': 'r', 'position': 0.95})
        self.plot.addItem(self.bent_thresh_line)
        layout.addWidget(self.plot)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Bent Thresh:"))
        sld_flex = QSlider(Qt.Orientation.Horizontal)
        sld_flex.setRange(0, 1023)
        sld_flex.setValue(int(self.thresholds.flex_bent))
        sld_flex.valueChanged.connect(self._on_bent_slider)
        slider_row.addWidget(sld_flex)
        layout.addLayout(slider_row)

    def _on_bent_slider(self, value):
        self.thresholds.flex_bent = value
        self.bent_thresh_line.setPos(value)

    def on_new_data(self, flex_values):
        """flex_values: iterable of 5 floats (thumb, index, middle, ring, pinky)."""
        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
        self.data_buffer[-1, :] = flex_values
        for i in range(5):
            self.curves[i].setData(self.data_buffer[:, i])