import queue
import time

# --- Data Management (Pub/Sub Queues) ---
class DataManager:
    """Manages routing of parsed serial data to different consumers."""
    def __init__(self):
        self.subscribers = {
            # PyQt6 GUI
            'gui_imu': queue.Queue(),
            'gui_ff': queue.Queue(),
            'gui_calib': queue.Queue(),
            'gui_log': queue.Queue(), # For errors and text logs
            'gui_hp': queue.Queue(), # hand points
            'gui_cube': queue.Queue(),

            # Debug log
            'log_imu': queue.Queue(),
            'log_ff': queue.Queue(),
            'log_cube': queue.Queue(),

            # Grasp inference
            'inf_ff': queue.Queue(),
            'inf_hp': queue.Queue(),
            'inf_cube': queue.Queue(),

            # Grasp detected
            'gui_grasp': queue.Queue(),
            'tcp_grasp': queue.Queue()
        }

    def broadcast_imu(self, data):
        self.subscribers['gui_imu'].put(data)
        self.subscribers['log_imu'].put(data)

    def broadcast_ff(self, data):
        self.subscribers['gui_ff'].put(data)
        self.subscribers['log_ff'].put(data)
        self.subscribers['inf_ff'].put(data)
        
    def broadcast_calib(self, data):
        self.subscribers['gui_calib'].put(data)

    def broadcast_hp(self, data):
        self.subscribers['gui_hp'].put(data)
        self.subscribers['inf_hp'].put(data)

    def broadcast_cube(self, data):
        self.subscribers['gui_cube'].put(data)
        self.subscribers['inf_cube'].put(data)

    def broadcast_grasp(self, data):
        self.subscribers['gui_grasp'].put(data)
        self.subscribers['tcp_grasp'].put(data)

    def log_event(self, msg, is_error=False):
        """Sends logs and error flags to the GUI."""
        self.subscribers['gui_log'].put((time.time(), msg, is_error))