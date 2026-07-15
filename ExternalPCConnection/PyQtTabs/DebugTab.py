import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QGroupBox, QLabel, QTextEdit)
from PyQt6.QtCore import Qt


class DebugTab(QWidget):
    """Sensor status indicators, calibration readout, system log, and error history."""

    def __init__(self, window_size=500, parent=None):
        super().__init__(parent)
        self.window_size = window_size
        self.error_val_buffer = np.zeros(self.window_size)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 1. Status Indicators
        status_group = QGroupBox("Sensor Status")
        status_layout = QHBoxLayout()
        self.ind_imu = self._create_indicator("IMU")
        self.ind_ff = self._create_indicator("Force/Flex")
        status_layout.addWidget(self.ind_imu)
        status_layout.addWidget(self.ind_ff)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 2. Calibration Data
        calib_group = QGroupBox("BNO055 Calibration (0=Uncalibrated, 3=Fully Calibrated)")
        calib_layout = QGridLayout()
        self.lbl_cal_sys = QLabel("Sys: --")
        self.lbl_cal_gyro = QLabel("Gyro: --")
        self.lbl_cal_accel = QLabel("Accel: --")
        self.lbl_cal_mag = QLabel("Mag: --")
        for lbl in [self.lbl_cal_sys, self.lbl_cal_gyro, self.lbl_cal_accel, self.lbl_cal_mag]:
            lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        calib_layout.addWidget(self.lbl_cal_sys, 0, 0)
        calib_layout.addWidget(self.lbl_cal_gyro, 0, 1)
        calib_layout.addWidget(self.lbl_cal_accel, 1, 0)
        calib_layout.addWidget(self.lbl_cal_mag, 1, 1)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # 3. Text Log
        log_group = QGroupBox("System Log & Errors")
        log_layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        log_layout.addWidget(self.text_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 4. Error Graph
        err_group = QGroupBox("Cumulative Errors Over Time")
        err_layout = QVBoxLayout()
        self.err_plot = pg.PlotWidget()
        self.err_curve = self.err_plot.plot(pen='r', name="Errors")
        err_layout.addWidget(self.err_plot)
        err_group.setLayout(err_layout)
        layout.addWidget(err_group)

    @staticmethod
    def _create_indicator(text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #7f8c8d; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        return lbl

    @staticmethod
    def _set_indicator(label, active, reason=""):
        base_text = label.text().split(' ')[0]
        if active:
            label.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            label.setText(f"{base_text} (ACTIVE)")
        else:
            label.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            label.setText(f"{base_text} ({reason})")

    def set_ff_status(self, active, reason=""):
        self._set_indicator(self.ind_ff, active, reason)

    def set_imu_status(self, active, reason=""):
        self._set_indicator(self.ind_imu, active, reason)

    def set_calibration(self, sys_c, gyro_c, accel_c, mag_c):
        self.lbl_cal_sys.setText(f"Sys: {sys_c}")
        self.lbl_cal_gyro.setText(f"Gyro: {gyro_c}")
        self.lbl_cal_accel.setText(f"Accel: {accel_c}")
        self.lbl_cal_mag.setText(f"Mag: {mag_c}")

    def append_log(self, time_str, msg, is_error):
        if is_error:
            self.text_log.append(f"<span style='color:red;'>[{time_str}] ERROR: {msg}</span>")
        else:
            self.text_log.append(f"[{time_str}] {msg}")

    def update_error_count(self, error_count):
        self.error_val_buffer = np.roll(self.error_val_buffer, -1)
        self.error_val_buffer[-1] = error_count
        self.err_curve.setData(self.error_val_buffer)