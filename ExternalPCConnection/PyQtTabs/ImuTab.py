import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class ImuTab(QWidget):
    """Live IMU quaternion and linear acceleration plots."""

    def __init__(self, window_size=500, parent=None):
        super().__init__(parent)
        self.window_size = window_size
        self.data_buffer = np.zeros((self.window_size, 7))  # qw, qx, qy, qz, ax, ay, az
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.accel_plot = pg.PlotWidget(title="Linear Acceleration (x, y, z)")
        self.accel_plot.addLegend()
        self.accel_curves = [self.accel_plot.plot(pen=(i, 3), name=n) for i, n in enumerate(["Ax", "Ay", "Az"])]

        self.quat_plot = pg.PlotWidget(title="Quaternion (w, x, y, z)")
        self.quat_plot.addLegend()
        self.quat_curves = [self.quat_plot.plot(pen=(i, 4), name=n) for i, n in enumerate(["Qw", "Qx", "Qy", "Qz"])]

        layout.addWidget(self.accel_plot)
        layout.addWidget(self.quat_plot)

    def on_new_data(self, imu_values):
        """imu_values: iterable of 7 floats [qw, qx, qy, qz, ax, ay, az]."""
        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
        self.data_buffer[-1, :] = imu_values
        for i in range(4):
            self.quat_curves[i].setData(self.data_buffer[:, i])
        for i in range(3):
            self.accel_curves[i].setData(self.data_buffer[:, i + 4])