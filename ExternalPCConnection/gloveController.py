import sys
import time
import struct
import queue
import threading
from datetime import datetime
import numpy as np
import serial

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTabWidget, QToolBar, QPushButton, QSlider, QLabel, 
                             QHBoxLayout, QTextEdit, QStatusBar, QGridLayout, QGroupBox)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

from TcpServer import TcpServer
from DataManager import DataManager

# --- Constants & Formats ---
COM_PORT = 'COM16'  # <-- CHANGE THIS TO YOUR TEENSY PORT
BAUD_RATE = 115200

# Struct formats based on __attribute__((__packed__)) C++ structs
# IMU: 4 floats (quat), 3 floats (accel) -> 28 bytes
FMT_IMU = '<7f'  
# FF: 5 uint16 (force), 5 uint16 (flex) -> 20 bytes
FMT_FF = '<10H'  
# CALIB: 4 uint8 -> 4 bytes
FMT_CALIB = '<4B'

# Packet Constants
START_BIT = 0xAA
END_BIT = 0x55
PACKET_IMU = 0
PACKET_FF = 1
PACKET_CALIB = 2

# Commands to send to Teensy (matching your C++ handleIdleState logic)
CMD_DEBUG = b'\x00'
CMD_CALIB = b'\x01'
CMD_STREAM = b'\x02'

# --- Serial Reader Thread ---
class SerialThread(threading.Thread):
    def __init__(self, port, baud, data_manager):
        super().__init__()
        self.port = port
        self.baud = baud
        self.dm = data_manager
        self.running = True
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.dm.log_event(f"Connected to {self.port}")
        except Exception as e:
            self.dm.log_event(f"Failed to connect to {self.port}: {e}", is_error=True)
            return

        state = "WAIT_START"
        packet_type = None

        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    if state == "WAIT_START":
                        if self.ser.read(1)[0] == START_BIT:
                            state = "READ_TYPE"
                        else:
                            # We read garbage data while waiting for a start bit
                            pass 
                    
                    elif state == "READ_TYPE":
                        packet_type = self.ser.read(1)[0]
                        if packet_type in [PACKET_IMU, PACKET_FF, PACKET_CALIB]:
                            state = "READ_PAYLOAD"
                        else:
                            self.dm.log_event(f"Unknown packet type: {packet_type}", is_error=True)
                            state = "WAIT_START"
                    
                    elif state == "READ_PAYLOAD":
                        if packet_type == PACKET_IMU:
                            payload = self.ser.read(28)
                            if len(payload) == 28:
                                data = struct.unpack(FMT_IMU, payload)
                                self.dm.broadcast_imu((time.time(), *data))
                            else:
                                self.dm.log_event("IMU payload incomplete.", is_error=True)
                        
                        elif packet_type == PACKET_FF:
                            payload = self.ser.read(20)
                            if len(payload) == 20:
                                data = struct.unpack(FMT_FF, payload)
                                self.dm.broadcast_ff((time.time(), *data))
                            else:
                                self.dm.log_event("FF payload incomplete.", is_error=True)
                        
                        elif packet_type == PACKET_CALIB:
                            payload = self.ser.read(4)
                            if len(payload) == 4:
                                data = struct.unpack(FMT_CALIB, payload)
                                self.dm.broadcast_calib((time.time(), *data))
                            else:
                                self.dm.log_event("CALIB payload incomplete.", is_error=True)
                        
                        state = "WAIT_END"

                    elif state == "WAIT_END":
                        if self.ser.read(1)[0] == END_BIT:
                            state = "WAIT_START"
                        else:
                            self.dm.log_event("Framing Error: Missing END_BIT.", is_error=True)
                            state = "WAIT_START"
                else:
                    time.sleep(0.001)
            except Exception as e:
                self.dm.log_event(f"Serial read error: {e}", is_error=True)
                time.sleep(0.1) # Prevent CPU pegging on severe errors
                
    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(cmd)
            cmd_name = {CMD_DEBUG: "DEBUG/IDLE", CMD_CALIB: "CALIB", CMD_STREAM: "STREAM"}.get(cmd, "UNKNOWN")
            self.dm.log_event(f"Sent Command: {cmd_name}")

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()


# --- Data Logger Thread ---
class LoggerThread(threading.Thread):
    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        self.running = True
        self.imu_history = []
        self.ff_history = []

    def run(self):
        while self.running:
            while not self.dm.subscribers['log_imu'].empty():
                self.imu_history.append(self.dm.subscribers['log_imu'].get())
            while not self.dm.subscribers['log_ff'].empty():
                self.ff_history.append(self.dm.subscribers['log_ff'].get())
            time.sleep(0.1)

    def save_and_stop(self):
        self.running = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.imu_history:
            np.save(f"imu_data_{timestamp}.npy", np.array(self.imu_history))
        if self.ff_history:
            np.save(f"ff_data_{timestamp}.npy", np.array(self.ff_history))


# --- Mock Inference Thread Example ---
class GraspInferenceThread(threading.Thread):
    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        self.running = True

    def run(self):
        while self.running:
            try:
                data = self.dm.subscribers['inf_ff'].get(timeout=1.0) 
            except queue.Empty:
                pass

    def stop(self):
        self.running = False


# --- Main PyQt6 GUI ---
class MainWindow(QMainWindow):
    def __init__(self, serial_thread):
        super().__init__()
        self.serial_thread = serial_thread
        self.dm = serial_thread.dm
        
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
        btn_debug.clicked.connect(lambda: self.serial_thread.send_command(CMD_DEBUG))
        toolbar.addWidget(btn_debug)

        btn_calib = QPushButton("REQUEST CALIB")
        btn_calib.clicked.connect(lambda: self.serial_thread.send_command(CMD_CALIB))
        toolbar.addWidget(btn_calib)

        btn_stream = QPushButton("START STREAM")
        btn_stream.clicked.connect(lambda: self.serial_thread.send_command(CMD_STREAM))
        toolbar.addWidget(btn_stream)

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setup_flex_tab()
        self.setup_force_tab()
        self.setup_imu_tab()
        self.setup_motor_tab()
        self.setup_debug_tab()

    def setup_flex_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.flex_plot = pg.PlotWidget(title="Flex Sensors (Analog Read)")
        self.flex_plot.addLegend()
        self.flex_curves = [self.flex_plot.plot(pen=(i, 5), name=n) for i, n in enumerate(["Thumb", "Index", "Middle", "Ring", "Pinky"])]
        layout.addWidget(self.flex_plot)
        self.tabs.addTab(tab, "Flex Graphs")

    def setup_force_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.force_plot = pg.PlotWidget(title="Force Sensors (Analog Read)")
        self.force_plot.addLegend()
        self.force_curves = [self.force_plot.plot(pen=(i, 5), name=n) for i, n in enumerate(["Thumb", "Index", "Middle", "Ring", "Pinky"])]
        layout.addWidget(self.force_plot)
        self.tabs.addTab(tab, "Force Graphs")

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


    def closeEvent(self, event): # type: ignore
        self.dm.log_event("Closing application, cleaning up threads...")
        self.serial_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    dm = DataManager()
    ser_thread = SerialThread(COM_PORT, BAUD_RATE, dm)
    log_thread = LoggerThread(dm)
    inf_thread = GraspInferenceThread(dm)

    ser_thread.start()
    log_thread.start()
    inf_thread.start()

    tcp_thread = TcpServer("127.0.0.1", 65432)
    tcp_thread.start()

    window = MainWindow(ser_thread)
    window.show()
    
    app.exec()

    tcp_thread.running = False
    
    log_thread.save_and_stop()
    inf_thread.stop()
    
    ser_thread.join()
    log_thread.join()
    inf_thread.join()