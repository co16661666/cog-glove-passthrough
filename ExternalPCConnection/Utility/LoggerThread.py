import numpy as np
import threading
import time
from datetime import datetime

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