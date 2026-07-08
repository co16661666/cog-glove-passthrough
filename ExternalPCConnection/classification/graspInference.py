import os
import numpy as np

# Point Python to CUDA bin folder
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)

# Point Python to cuDNN bin folder
cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.8\bin\12.8"
if os.path.exists(cudnn_path):
    os.add_dll_directory(cudnn_path)

import onnxruntime as ort

import serial
from serial.tools import list_ports
import threading

print("Available ports:\n")
ports = list(list_ports.comports())
for p in ports:
    print(p.device)

delimiter = ','.encode()

# Load model
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = ort.InferenceSession(r"C:\Users\liong\Documents\cog-glove\classification\grasp_modelBETTER.onnx", providers=providers)
input_name = sess.get_inputs()[0].name

# Flex1,Flex2,Flex3,Flex4,Flex5,Force1,Force2,Force3,Force4,Force5

class GraspInference(threading.Thread):
    def __init__(self, baud=921600, port='COM10'):
        super().__init__()
        self.ser = serial.Serial(port=port, baudrate=baud)
        self.delimiter = ','.encode()
        self.latest_prediction = None
        self.running = True
        self.daemon = True

        self.frame_id = 0
        self.ser.reset_input_buffer()

    def run(self):
        while self.running:
            if self.ser.in_waiting > 0:
                values = self.ser.readline().split(self.delimiter)

                float_values = []
                for v in values:
                    try:
                        float_values.append(float(v.decode().strip()))
                    except ValueError:
                        print(f"Invalid value: {v.decode().strip()}")

                if len(float_values) != 10:
                    print("Invalid num values:", len(float_values))
                    continue

                X_test = np.array(float_values, dtype=np.float32).reshape(1, 10)

                pred_onx = sess.run(None, {input_name: X_test})[0]
                print("Prediction successful:", pred_onx)

                if isinstance(pred_onx, np.ndarray):
                    self.latest_prediction = pred_onx[0][0]
                    self.frame_id += 1

    def is_grasped(self):
        if self.latest_prediction is None:
            return False
        
        if self.latest_prediction > 0.5:
            return True
        else:
            return False
        
    def stop(self):
        self.running = False