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

ports = list(list_ports.comports())
for p in ports:
    print(p.device)

ser = serial.Serial(
    port='COM10',
    baudrate=921600
)

delimiter = ','.encode()

# Load model
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = ort.InferenceSession("grasp_modelBETTER.onnx", providers=providers)
input_name = sess.get_inputs()[0].name

# Flex1,Flex2,Flex3,Flex4,Flex5,Force1,Force2,Force3,Force4,Force5
print(ser.readline().decode())

while(True):
    values = ser.readline().split(delimiter)

    float_values = []
    for v in values:
        float_values.append(float(v.decode().strip()))

    if len(float_values) != 10:
        print("Invalid num values:", len(float_values))
        continue

    X_test = np.array(float_values, dtype=np.float32).reshape(1, 10)

    pred_onx = sess.run(None, {input_name: X_test})[0]
    print("Prediction successful:", pred_onx)

    if isinstance(pred_onx, np.ndarray):
        if pred_onx[0][0] > 0.5:
            print("True")
        else:
            print("False")