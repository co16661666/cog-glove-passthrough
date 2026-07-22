import serial
import threading
import struct
import time

# --- Constants & Formats ---
# Packet Constants
START_BIT = 0xAA
END_BIT = 0x55
PACKET_IMU = 0
PACKET_FF = 1
PACKET_CALIB = 2

# Commands to send to Teensy
# RETURN = 0,
# DEBUG = 1,
# CALIBRATION = 2,
# DATA_STREAM = 3,
# TRIGGER_HAPTIC = 4
CMD_RETURN = b'\x00'
CMD_DEBUG = b'\x01'
CMD_CALIB = b'\x02'
CMD_STREAM = b'\x03'

# Struct formats based on __attribute__((__packed__)) C++ structs
# IMU: 4 floats (quat), 3 floats (accel) -> 28 bytes
FMT_IMU = '<7f'  
# FF: 5 uint16 (force), 5 uint16 (flex) -> 20 bytes
FMT_FF = '<10H'  
# CALIB: 4 uint8 -> 4 bytes
FMT_CALIB = '<4B'
# HAPTIC: 4 uint8 -> 4 bytes
FMT_HAPTIC = '<2B'

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
            cmd_name = {CMD_RETURN: 'CMD_RETURN', CMD_DEBUG: "DEBUG", CMD_CALIB: "CALIB", CMD_STREAM: "STREAM"}.get(cmd, "UNKNOWN")
            self.dm.log_event(f"Sent Command: {cmd_name}")

    def handle_grasped(self, active_motors, patterns):
        for finger, is_enabled in active_motors.items():
            if is_enabled:
                self.send_command(struct.pack(FMT_HAPTIC, finger.value, patterns[finger])) # driver, command

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()