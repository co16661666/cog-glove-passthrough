import sys
import time
import queue
import threading
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QCheckBox,
                             QTabWidget, QToolBar, QPushButton, QSlider, QLabel, 
                             QHBoxLayout, QTextEdit, QStatusBar, QGridLayout, QGroupBox)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph.opengl as gl          # for 3D plots
import pyqtgraph as pg

# Utility imports
from TcpServer import TcpServer
from DataManager import DataManager
from Utility.ThresholdStore import ThresholdStore
from Utility.SerialThread import SerialThread
import Utility.SerialThread # to access constants
from Utility.LoggerThread import LoggerThread

COM_PORT = 'COM16'  # <-- CHANGE THIS TO YOUR TEENSY PORT
BAUD_RATE = 115200


# --- Mock Inference Thread Example ---
class GraspInferenceThread(threading.Thread):
    """
    Rule-based grasp detector. A grasp is flagged only when all three
    signals agree for the current frame:
      1. Force  - thumb fingertip force exceeds its own threshold AND
                  (index + middle) fingertip force, summed, exceeds a
                  pinch threshold.
      2. Proximity - thumb/index/middle fingertips are each close to the
                     cube AND close to one another (tripod pinch geometry).
      3. Flex   - thumb/index/middle flex sensors read as "bent".
    """

    # ff payload column layout: ["Thumb","Index","Middle","Ring","Pinky"]
    FORCE_THUMB, FORCE_INDEX, FORCE_MIDDLE = 0, 1, 2
    FLEX_THUMB, FLEX_INDEX, FLEX_MIDDLE = 5, 6, 7

    # keypoints_list indices, per LANDMARK_NAMES in LeapConnector.py
    IDX_THUMB_TIP, IDX_INDEX_TIP, IDX_MIDDLE_TIP = 4, 8, 12

    def __init__(self, data_manager, thresholds):
        super().__init__()
        self.dm = data_manager
        self.thresholds = thresholds
        self.running = True
        self.daemon = True

        # Caches - ff / hp / cube arrive independently and at different
        # rates, so we hold the latest of each rather than requiring all
        # three to land in the same tick.
        self.latest_force = None   # 10 values: [f_thumb..f_pinky, flex_thumb..flex_pinky]
        self.latest_hands = None   # list of hand dicts (LeapConnector.get_latest_hands())
        self.latest_cube = None    # [tx, ty, tz, rx, ry, rz]

        self.latest_prediction = False
        self.frame_id = 0

    def run(self):
        while self.running:
            updated = False

            try:
                while True:
                    _, *ff = self.dm.subscribers['inf_ff'].get_nowait()
                    self.latest_force = ff
                    updated = True
            except queue.Empty:
                pass

            try:
                while True:
                    _, hands = self.dm.subscribers['inf_hp'].get_nowait()
                    self.latest_hands = hands
                    updated = True
            except queue.Empty:
                pass

            try:
                while True:
                    _, cube = self.dm.subscribers['inf_cube'].get_nowait()  # (timestamp, [tx,ty,tz,rx,ry,rz])
                    self.latest_cube = cube
                    updated = True
            except queue.Empty:
                pass

            if not updated:
                time.sleep(0.005)
                continue

            prev = self.latest_prediction
            self.latest_prediction = self._evaluate_grasp()
            self.frame_id += 1

            if self.latest_prediction != prev:
                self.dm.broadcast_grasp('GRASPED' if self.latest_prediction else 'RELEASED')
                self.dm.log_event(f"Grasp {'GRASPED' if self.latest_prediction else 'RELEASED'}")

    def _evaluate_grasp(self) -> bool:
        if self.latest_force is None or self.latest_hands is None or self.latest_cube is None:
            return False

        thumb_tip = index_tip = middle_tip = None
        for hand in self.latest_hands:
            kp = hand.get("keypoints_list")
            if not kp or len(kp) <= self.IDX_MIDDLE_TIP:
                continue
            raw_thumb = np.array(kp[self.IDX_THUMB_TIP])
            raw_index = np.array(kp[self.IDX_INDEX_TIP])
            raw_middle = np.array(kp[self.IDX_MIDDLE_TIP])

            break  # only using the first tracked hand for now

        if thumb_tip is None or index_tip is None or middle_tip is None:
            return False

        # 1. Force
        thumb_force = self.latest_force[self.FORCE_THUMB]
        pinch_force = self.latest_force[self.FORCE_INDEX] + self.latest_force[self.FORCE_MIDDLE]
        force_ok = (thumb_force > self.thresholds.force_thumb and pinch_force > self.thresholds.force_pinch)

        # 2. Proximity
        cube_pos = np.array(self.latest_cube[:3])
        cube_dist_ok = all(
            np.linalg.norm(tip - cube_pos) < self.thresholds.cube_proximity
            for tip in (thumb_tip, index_tip, middle_tip)
        )
        finger_dist_ok = (
            np.linalg.norm(thumb_tip - index_tip) < self.thresholds.finger_proximity and
            np.linalg.norm(thumb_tip - middle_tip) < self.thresholds.finger_proximity and
            np.linalg.norm(index_tip - middle_tip) < self.thresholds.finger_proximity
        )
        proximity_ok = cube_dist_ok and finger_dist_ok

        # 3. Flex
        flex_ok = (
            self.latest_force[self.FLEX_THUMB] > self.thresholds.flex_bent and
            self.latest_force[self.FLEX_INDEX] > self.thresholds.flex_bent and
            self.latest_force[self.FLEX_MIDDLE] > self.thresholds.flex_bent
        )

        return bool(force_ok and proximity_ok and flex_ok)

    def stop(self):
        self.running = False

# --- 3D Plotting ---
class HandCubeTab(QWidget):
    """3D viewer for hand landmarks and the tracked cube, with visibility toggles."""

    # Standard 21-point hand landmark indices (per LANDMARK_NAMES in LeapConnector.py)
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    CUBE_HALF_EXTENT = 0.02  # meters

    def __init__(self, data_manager, thresholds, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.thresholds = thresholds
        self.show_fingertips_only = False
        self.show_cube = True
        self.show_hand = True
        self.latest_tripod = None # finger tips
        self.latest_cube_pos = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.chk_show_hand = QCheckBox("Show Hand")
        self.chk_show_hand.setChecked(True)
        self.chk_show_hand.stateChanged.connect(self._on_hand_toggle)
        controls.addWidget(self.chk_show_hand)

        self.chk_fingertips_only = QCheckBox("Fingertips Only")
        self.chk_fingertips_only.stateChanged.connect(self._on_fingertips_toggle)
        controls.addWidget(self.chk_fingertips_only)

        self.chk_show_cube = QCheckBox("Show Cube")
        self.chk_show_cube.setChecked(True)
        self.chk_show_cube.stateChanged.connect(self._on_cube_toggle)
        controls.addWidget(self.chk_show_cube)

        controls.addStretch()
        layout.addLayout(controls)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=0.5)
        grid = gl.GLGridItem()
        grid.setSize(1, 1)
        grid.setSpacing(0.05, 0.05)
        self.view.addItem(grid)
        layout.addWidget(self.view)

        self.hand_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0.2, 0.8, 1.0, 1.0), size=8)
        self.view.addItem(self.hand_scatter)

        # 12 edges of the cube, drawn as a wireframe via GLLinePlotItem(mode='lines')
        self.cube_lines = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0.4, 0.1, 1.0), width=2, mode='lines')
        self.view.addItem(self.cube_lines)

        thresh_group = QGroupBox("Proximity Thresholds")
        thresh_layout = QGridLayout()

        thresh_layout.addWidget(QLabel("Cube Proximity Threshold:"), 0, 0)
        self.sld_cube_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sld_cube_thresh.setRange(1, 500)  # millimeters
        self.sld_cube_thresh.setValue(int(self.thresholds.cube_proximity * 1000))
        self.sld_cube_thresh.valueChanged.connect(self._on_cube_thresh_slider)
        thresh_layout.addWidget(self.sld_cube_thresh, 0, 1)
        self.lbl_cube_thresh_val = QLabel(f"{self.thresholds.cube_proximity:.3f} m")
        thresh_layout.addWidget(self.lbl_cube_thresh_val, 0, 2)
        thresh_layout.addWidget(QLabel("Min Tip-to-Cube Dist:"), 0, 3)
        self.lbl_cube_dist = QLabel("--")
        thresh_layout.addWidget(self.lbl_cube_dist, 0, 4)

        thresh_layout.addWidget(QLabel("Finger Proximity Threshold:"), 1, 0)
        self.sld_finger_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sld_finger_thresh.setRange(1, 500)  # millimeters
        self.sld_finger_thresh.setValue(int(self.thresholds.finger_proximity * 1000))
        self.sld_finger_thresh.valueChanged.connect(self._on_finger_thresh_slider)
        thresh_layout.addWidget(self.sld_finger_thresh, 1, 1)
        self.lbl_finger_thresh_val = QLabel(f"{self.thresholds.finger_proximity:.3f} m")
        thresh_layout.addWidget(self.lbl_finger_thresh_val, 1, 2)
        thresh_layout.addWidget(QLabel("Max Finger-to-Finger Dist:"), 1, 3)
        self.lbl_finger_dist = QLabel("--")
        thresh_layout.addWidget(self.lbl_finger_dist, 1, 4)

        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)

    def _on_cube_thresh_slider(self, value_mm):
        self.thresholds.cube_proximity = value_mm / 1000.0
        self.lbl_cube_thresh_val.setText(f"{self.thresholds.cube_proximity:.3f} m")

    def _on_finger_thresh_slider(self, value_mm):
        self.thresholds.finger_proximity = value_mm / 1000.0
        self.lbl_finger_thresh_val.setText(f"{self.thresholds.finger_proximity:.3f} m")

    def _on_hand_toggle(self, state):
        self.show_hand = bool(state)
        self.hand_scatter.setVisible(self.show_hand)

    def _on_fingertips_toggle(self, state):
        self.show_fingertips_only = bool(state)

    def _on_cube_toggle(self, state):
        self.show_cube = bool(state)
        self.cube_lines.setVisible(self.show_cube)

    def _update_threshold_readout(self):
        if self.latest_tripod is None:
            return

        thumb = np.array(self.latest_tripod['thumb'])
        index = np.array(self.latest_tripod['index'])
        middle = np.array(self.latest_tripod['middle'])

        finger_dist = max(
            np.linalg.norm(thumb - index),
            np.linalg.norm(thumb - middle),
            np.linalg.norm(index - middle),
        )
        finger_ok = finger_dist < self.thresholds.finger_proximity
        self.lbl_finger_dist.setText(f"{finger_dist:.3f} m")
        self.lbl_finger_dist.setStyleSheet(f"color: {'#2ecc71' if finger_ok else '#e74c3c'}; font-weight: bold;")

        if self.latest_cube_pos is not None:
            cube_dist = max(
                np.linalg.norm(thumb - self.latest_cube_pos),
                np.linalg.norm(index - self.latest_cube_pos),
                np.linalg.norm(middle - self.latest_cube_pos),
            )
            cube_ok = cube_dist < self.thresholds.cube_proximity
            self.lbl_cube_dist.setText(f"{cube_dist:.3f} m")
            self.lbl_cube_dist.setStyleSheet(f"color: {'#2ecc71' if cube_ok else '#e74c3c'}; font-weight: bold;")

    @staticmethod
    def _cube_corners(tx, ty, tz, rx, ry, rz, half_extent):
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        h = half_extent
        local_corners = np.array([
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ])
        return (R @ local_corners.T).T + np.array([tx, ty, tz])

    @staticmethod
    def _cube_edges(corners):
        edge_idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        pts = []
        for i, j in edge_idx:
            pts.append(corners[i])
            pts.append(corners[j])
        return np.array(pts)

    def update_view(self):
        # --- Hand points ---
        hp_queue = self.dm.subscribers['gui_hp']
        latest_hands = None
        while not hp_queue.empty():
            _, hands = hp_queue.get()
            latest_hands = hands

        if latest_hands is not None and self.show_hand:
            pts = []
            for hand in latest_hands:
                kp = hand.get("keypoints_list")
                if not kp:
                    continue
                indices = self.FINGERTIP_INDICES if self.show_fingertips_only else range(len(kp))
                for idx in indices:
                    if idx < len(kp):
                        pts.append(kp[idx])
            if pts:
                self.hand_scatter.setData(pos=np.array(pts))

        # --- Cube ---
        cube_queue = self.dm.subscribers['gui_cube']
        latest_cube = None
        while not cube_queue.empty():
            _, cube = cube_queue.get()  # (timestamp, [tx,ty,tz,rx,ry,rz])
            latest_cube = cube

        if latest_cube is not None:
            self.latest_cube_pos = np.array(latest_cube[:3])
            if self.show_cube:
                corners = self._cube_corners(*latest_cube, self.CUBE_HALF_EXTENT) # type: ignore
                self.cube_lines.setData(pos=self._cube_edges(corners))

        self._update_threshold_readout()

# --- Main PyQt6 GUI ---
class MainWindow(QMainWindow):
    def __init__(self, serial_thread, thresholds):
        super().__init__()
        self.serial_thread = serial_thread
        self.dm = serial_thread.dm
        self.thresholds = thresholds
        
        self.setWindowTitle("Teensy Sensor Interface")
        self.resize(1100, 800)

        # Plot Data Buffers
        self.window_size = 500
        self.ff_data_buffer = np.zeros((self.window_size, 10))
        self.imu_data_buffer = np.zeros((self.window_size, 7))
        
        # Error tracking
        self.error_count = 0
        self.error_time_buffer = np.zeros(self.window_size)
        self.error_val_buffer = np.zeros(self.window_size)

        # Status tracking
        self.last_ff_time = 0
        self.last_imu_time = 0

        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(33)

    def setup_ui(self):
        # Persistent Global Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Initializing...")
        self.error_label = QLabel("Errors: 0")
        self.error_label.setStyleSheet("color: red; font-weight: bold;")
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.error_label)

        # Quick-Menu Bar
        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)

        btn_debug = QPushButton("DEBUG MODE (IDLE)")
        btn_debug.clicked.connect(lambda: self.serial_thread.send_command(Utility.SerialThread.CMD_DEBUG))
        toolbar.addWidget(btn_debug)

        btn_calib = QPushButton("REQUEST CALIB")
        btn_calib.clicked.connect(lambda: self.serial_thread.send_command(Utility.SerialThread.CMD_CALIB))
        toolbar.addWidget(btn_calib)

        btn_stream = QPushButton("START STREAM")
        btn_stream.clicked.connect(lambda: self.serial_thread.send_command(Utility.SerialThread.CMD_STREAM))
        toolbar.addWidget(btn_stream)

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setup_flex_tab()
        self.setup_force_tab()
        self.setup_imu_tab()
        self.setup_hand_cube_tab()
        self.setup_motor_tab()
        self.setup_debug_tab()

    def setup_force_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.force_plot = pg.PlotWidget(title="Force Sensors (Analog Read)")
        self.force_plot.addLegend()
        self.force_curves = [self.force_plot.plot(pen=(i, 5), name=n) for i, n in enumerate(["Thumb", "Index", "Middle", "Ring", "Pinky"])]

        self.thumb_thresh_line = pg.InfiniteLine(pos=self.thresholds.force_thumb, angle=0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine),
                                                   label="Thumb Threshold", labelOpts={'color': 'r', 'position': 0.95})
        self.pinch_thresh_line = pg.InfiniteLine(pos=self.thresholds.force_pinch, angle=0, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine),
                                                   label="Pinch Threshold (Idx+Mid sum)", labelOpts={'color': 'y', 'position': 0.85})
        self.force_plot.addItem(self.thumb_thresh_line)
        self.force_plot.addItem(self.pinch_thresh_line)
        layout.addWidget(self.force_plot)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Thumb Thresh:"))
        sld_thumb = QSlider(Qt.Orientation.Horizontal)
        sld_thumb.setRange(0, 1000)
        sld_thumb.setValue(int(self.thresholds.force_thumb))
        sld_thumb.valueChanged.connect(self._on_force_thumb_slider)
        slider_row.addWidget(sld_thumb)

        slider_row.addWidget(QLabel("Pinch Thresh:"))
        sld_pinch = QSlider(Qt.Orientation.Horizontal)
        sld_pinch.setRange(0, 1000)
        sld_pinch.setValue(int(self.thresholds.force_pinch))
        sld_pinch.valueChanged.connect(self._on_force_pinch_slider)
        slider_row.addWidget(sld_pinch)
        layout.addLayout(slider_row)

        self.tabs.addTab(tab, "Force Graphs")

    def _on_force_thumb_slider(self, value):
        self.thresholds.force_thumb = value
        self.thumb_thresh_line.setPos(value)

    def _on_force_pinch_slider(self, value):
        self.thresholds.force_pinch = value
        self.pinch_thresh_line.setPos(value)

    def setup_flex_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.flex_plot = pg.PlotWidget(title="Flex Sensors (Analog Read)")
        self.flex_plot.addLegend()
        self.flex_curves = [self.flex_plot.plot(pen=(i, 5), name=n) for i, n in enumerate(["Thumb", "Index", "Middle", "Ring", "Pinky"])]

        self.bent_thresh_line = pg.InfiniteLine(pos=self.thresholds.flex_bent, angle=0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine),
                                                  label="Bent Threshold", labelOpts={'color': 'r', 'position': 0.95})
        self.flex_plot.addItem(self.bent_thresh_line)
        layout.addWidget(self.flex_plot)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Bent Thresh:"))
        sld_flex = QSlider(Qt.Orientation.Horizontal)
        sld_flex.setRange(0, 1023)
        sld_flex.setValue(int(self.thresholds.flex_bent))
        sld_flex.valueChanged.connect(self._on_flex_bent_slider)
        slider_row.addWidget(sld_flex)
        layout.addLayout(slider_row)

        self.tabs.addTab(tab, "Flex Graphs")

    def _on_flex_bent_slider(self, value):
        self.thresholds.flex_bent = value
        self.bent_thresh_line.setPos(value)

    def setup_imu_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.accel_plot = pg.PlotWidget(title="Linear Acceleration (x, y, z)")
        self.accel_plot.addLegend()
        self.accel_curves = [self.accel_plot.plot(pen=(i, 3), name=n) for i, n in enumerate(["Ax", "Ay", "Az"])]
        
        self.quat_plot = pg.PlotWidget(title="Quaternion (w, x, y, z)")
        self.quat_plot.addLegend()
        self.quat_curves = [self.quat_plot.plot(pen=(i, 4), name=n) for i, n in enumerate(["Qw", "Qx", "Qy", "Qz"])]

        layout.addWidget(self.accel_plot)
        layout.addWidget(self.quat_plot)
        self.tabs.addTab(tab, "IMU Data")
    
    def setup_hand_cube_tab(self):
        self.hand_cube_tab = HandCubeTab(self.dm, self.thresholds)
        self.tabs.addTab(self.hand_cube_tab, "Hand & Cube")

    def setup_motor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Vibration Motor Control (Placeholder)"))
        for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(f))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 255)
            slider.valueChanged.connect(lambda val, finger=f: self.dm.log_event(f"Motor {finger} -> {val}"))
            h_layout.addWidget(slider)
            layout.addLayout(h_layout)
        self.tabs.addTab(tab, "Motors")

    def setup_debug_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 1. Status Indicators Group
        status_group = QGroupBox("Sensor Status")
        status_layout = QHBoxLayout()
        self.ind_imu = self.create_indicator("IMU")
        self.ind_ff = self.create_indicator("Force/Flex")
        status_layout.addWidget(self.ind_imu)
        status_layout.addWidget(self.ind_ff)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 2. Calibration Data Group
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

        self.tabs.addTab(tab, "Debug & Logs")

    def create_indicator(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #7f8c8d; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        return lbl

    def set_indicator(self, label, active, reason=""):
        if active:
            label.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            label.setText(f"{label.text().split(' ')[0]} (ACTIVE)")
        else:
            label.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            label.setText(f"{label.text().split(' ')[0]} ({reason})")

    def update_plots(self):
        curr_time = time.time()
        
        # --- Handle Text Logs & Errors ---
        log_queue = self.dm.subscribers['gui_log']
        while not log_queue.empty():
            t_stamp, msg, is_error = log_queue.get()
            time_str = datetime.fromtimestamp(t_stamp).strftime('%H:%M:%S.%f')[:-3]
            
            if is_error:
                self.error_count += 1
                self.error_label.setText(f"Errors: {self.error_count}")
                self.text_log.append(f"<span style='color:red;'>[{time_str}] ERROR: {msg}</span>")
            else:
                self.text_log.append(f"[{time_str}] {msg}")
                self.status_label.setText(f"Status: {msg}")
                
        # Update Error Graph buffer
        self.error_val_buffer = np.roll(self.error_val_buffer, -1)
        self.error_val_buffer[-1] = self.error_count
        self.err_curve.setData(self.error_val_buffer)

        # --- Handle Calibration ---
        calib_queue = self.dm.subscribers['gui_calib']
        while not calib_queue.empty():
            _, sys_c, gyro_c, accel_c, mag_c = calib_queue.get()
            self.lbl_cal_sys.setText(f"Sys: {sys_c}")
            self.lbl_cal_gyro.setText(f"Gyro: {gyro_c}")
            self.lbl_cal_accel.setText(f"Accel: {accel_c}")
            self.lbl_cal_mag.setText(f"Mag: {mag_c}")

        # --- Handle Force/Flex ---
        ff_queue = self.dm.subscribers['gui_ff']
        ff_updates = 0
        latest_ff_data = None
        while not ff_queue.empty():
            data = ff_queue.get() 
            self.last_ff_time = data[0]
            latest_ff_data = np.array(data[1:])
            self.ff_data_buffer = np.roll(self.ff_data_buffer, -1, axis=0)
            self.ff_data_buffer[-1, :] = latest_ff_data
            ff_updates += 1

        if ff_updates > 0:
            for i in range(5): self.force_curves[i].setData(self.ff_data_buffer[:, i])
            for i in range(5): self.flex_curves[i].setData(self.ff_data_buffer[:, i+5])

            # Debug Indicator Logic (Check for zeros)
            if np.all(latest_ff_data == 0):
                self.set_indicator(self.ind_ff, False, "ALL ZEROS")
            else:
                self.set_indicator(self.ind_ff, True)
        elif curr_time - self.last_ff_time > 2.0:
            self.set_indicator(self.ind_ff, False, "NO DATA")

        # --- Handle IMU ---
        imu_queue = self.dm.subscribers['gui_imu']
        imu_updates = 0
        latest_imu_data = None
        while not imu_queue.empty():
            data = imu_queue.get()
            self.last_imu_time = data[0]
            latest_imu_data = np.array(data[1:])
            self.imu_data_buffer = np.roll(self.imu_data_buffer, -1, axis=0)
            self.imu_data_buffer[-1, :] = latest_imu_data
            imu_updates += 1

        if imu_updates > 0:
            for i in range(4): self.quat_curves[i].setData(self.imu_data_buffer[:, i])
            for i in range(3): self.accel_curves[i].setData(self.imu_data_buffer[:, i+4])

            # Debug Indicator Logic (Check for zeros)
            if np.all(latest_imu_data == 0):
                self.set_indicator(self.ind_imu, False, "ALL ZEROS")
            else:
                self.set_indicator(self.ind_imu, True)
        elif curr_time - self.last_imu_time > 2.0:
            self.set_indicator(self.ind_imu, False, "NO DATA")

        # --- Handle 3D Plot ---
        self.hand_cube_tab.update_view()


    def closeEvent(self, event): # type: ignore
        self.dm.log_event("Closing application, cleaning up threads...")
        self.serial_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    dm = DataManager()
    thresholds = ThresholdStore()

    ser_thread = SerialThread(COM_PORT, BAUD_RATE, dm)
    log_thread = LoggerThread(dm)
    inf_thread = GraspInferenceThread(dm, thresholds)

    ser_thread.start()
    log_thread.start()
    inf_thread.start()

    tcp_thread = TcpServer("127.0.0.1", 65432, dm)
    tcp_thread.start()

    window = MainWindow(ser_thread, thresholds)
    window.show()
    
    app.exec()

    tcp_thread.running = False
    
    log_thread.save_and_stop()
    inf_thread.stop()
    
    ser_thread.join()
    log_thread.join()
    inf_thread.join()