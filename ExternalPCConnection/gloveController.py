import sys
import time
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow,QTabWidget, QToolBar,
                                QPushButton, QLabel, QStatusBar)
from PyQt6.QtCore import QTimer

# Utility imports
from TcpServer import TcpServer
from DataManager import DataManager
from Utility.ThresholdStore import ThresholdStore
from Utility.SerialThread import SerialThread
import Utility.SerialThread # to access constants
from Utility.LoggerThread import LoggerThread
from classification.GraspInferenceThread import GraspInferenceThread

# Tab widgets
from PyQtTabs.ForceTab import ForceTab
from PyQtTabs.FlexTab import FlexTab
from PyQtTabs.ImuTab import ImuTab
from PyQtTabs.HandCubeTab import HandCubeTab
from PyQtTabs.MotorTab import MotorTab
from PyQtTabs.DebugTab import DebugTab


COM_PORT = 'COM16'  # <-- CHANGE THIS TO YOUR TEENSY PORT
BAUD_RATE = 115200

# --- Main PyQt6 GUI ---
class MainWindow(QMainWindow):
    """
    Owns the toolbar, status bar, and the tab container. Each tab is a
    self-contained widget (see PyQtTabs/) responsible for its own layout
    and rendering. This window's remaining job is:
      1. Draining the DataManager queues that feed more than one tab
         (force+flex share a single 'gui_ff' message, and the debug tab's
         status indicators depend on force/flex/imu freshness), and
      2. Handing the parsed data off to the relevant tab(s).
    Queues used exclusively by a single tab (e.g. hand/cube data) are
    drained by that tab itself - see HandCubeTab.update_view().
    """
 
    def __init__(self, serial_thread, thresholds):
        super().__init__()
        self.serial_thread = serial_thread
        self.dm = serial_thread.dm
        self.thresholds = thresholds
 
        self.setWindowTitle("Teensy Sensor Interface")
        self.resize(1100, 800)
 
        # Error/status tracking (shared across debug tab + status bar)
        self.error_count = 0
        self.last_ff_time = 0
        self.last_imu_time = 0
 
        self._setup_ui()
 
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_all)
        self.timer.start(33)
 
    def _setup_ui(self):
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
 
        self.flex_tab = FlexTab(self.thresholds)
        self.force_tab = ForceTab(self.thresholds)
        self.imu_tab = ImuTab()
        self.hand_cube_tab = HandCubeTab(self.dm, self.thresholds)
        self.motor_tab = MotorTab(self.dm)
        self.debug_tab = DebugTab()
 
        self.tabs.addTab(self.flex_tab, "Flex Graphs")
        self.tabs.addTab(self.force_tab, "Force Graphs")
        self.tabs.addTab(self.imu_tab, "IMU Data")
        self.tabs.addTab(self.hand_cube_tab, "Hand & Cube")
        self.tabs.addTab(self.motor_tab, "Motors")
        self.tabs.addTab(self.debug_tab, "Debug & Logs")
 
    def _update_all(self):
        curr_time = time.time()
 
        # --- Handle Text Logs & Errors ---
        log_queue = self.dm.subscribers['gui_log']
        while not log_queue.empty():
            t_stamp, msg, is_error = log_queue.get()
            time_str = datetime.fromtimestamp(t_stamp).strftime('%H:%M:%S.%f')[:-3]
 
            if is_error:
                self.error_count += 1
                self.error_label.setText(f"Errors: {self.error_count}")
            else:
                self.status_label.setText(f"Status: {msg}")
 
            self.debug_tab.append_log(time_str, msg, is_error)
 
        self.debug_tab.update_error_count(self.error_count)
 
        # --- Handle Calibration ---
        calib_queue = self.dm.subscribers['gui_calib']
        while not calib_queue.empty():
            _, sys_c, gyro_c, accel_c, mag_c = calib_queue.get()
            self.debug_tab.set_calibration(sys_c, gyro_c, accel_c, mag_c)
 
        # --- Handle Force/Flex (single wire message: 5 force + 5 flex values) ---
        ff_queue = self.dm.subscribers['gui_ff']
        ff_updates = 0
        latest_ff_data = None
        while not ff_queue.empty():
            data = ff_queue.get()
            self.last_ff_time = data[0]
            latest_ff_data = np.array(data[1:])
            ff_updates += 1
 
        if ff_updates > 0 and latest_ff_data is not None:
            self.force_tab.on_new_data(latest_ff_data[:5])
            self.flex_tab.on_new_data(latest_ff_data[5:])
 
            if np.all(latest_ff_data == 0):
                self.debug_tab.set_ff_status(False, "ALL ZEROS")
            else:
                self.debug_tab.set_ff_status(True)
        elif curr_time - self.last_ff_time > 2.0:
            self.debug_tab.set_ff_status(False, "NO DATA")
 
        # --- Handle IMU ---
        imu_queue = self.dm.subscribers['gui_imu']
        imu_updates = 0
        latest_imu_data = None
        while not imu_queue.empty():
            data = imu_queue.get()
            self.last_imu_time = data[0]
            latest_imu_data = np.array(data[1:])
            imu_updates += 1
 
        if imu_updates > 0:
            self.imu_tab.on_new_data(latest_imu_data)
 
            if np.all(latest_imu_data == 0):
                self.debug_tab.set_imu_status(False, "ALL ZEROS")
            else:
                self.debug_tab.set_imu_status(True)
        elif curr_time - self.last_imu_time > 2.0:
            self.debug_tab.set_imu_status(False, "NO DATA")
 
        # --- Handle 3D Plot (drains its own queues internally) ---
        self.hand_cube_tab.update_view()
 
    def closeEvent(self, event):  # type: ignore
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