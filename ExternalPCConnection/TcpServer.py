import struct
import sys
import traceback

from collections import deque
import socket
import threading
import queue

import json
import time
import csv
from datetime import datetime

latest_timestamp = 0
running = True

send_queue = queue.Queue()

use_inference = True
use_hands = True

try:
    from classification.graspInference import GraspInference

    predictor = GraspInference()
    predictor.start()

except Exception as e:
    use_inference = False
    print(f"GraspInference initialization error: {e}")

try:
    from HandTracking.LeapConnector import LeapTrackingThread

    leap_thread = LeapTrackingThread()
    leap_thread.start()

except Exception as e:
    use_hands = False
    print(f"Hand tracking initialization error: {e}")

def data_manager(client_alive):
    print(f"Data manager started")
    global running
    global latest_timestamp
    latest_grasp_index = 0
    latest_hand_index = 0

    while running:
        if use_inference == True:
            if latest_grasp_index < predictor.frame_id:
                latest_grasp_index = predictor.frame_id

                if predictor.latest_prediction == True:
                    send_queue.put((latest_timestamp, [1.0]))
                else:
                    send_queue.put((latest_timestamp, [0.0]))

        if use_hands == True:
            if latest_hand_index < leap_thread.get_latest_frame_id():
                latest_hand_index = leap_thread.get_latest_frame_id()

                send_queue.put((latest_timestamp, leap_thread.get_latest_hands_flattened()))

    time.sleep(0.001)


def receiver_thread(client_socket, addr):
    """Continuously receives timestamp headers and bit-packed float messages from Unity."""
    print(f"Receiver started for {addr}")
    global running
    global latest_timestamp
    
    # 12 bytes total: 8 bytes for ulong (timestamp) + 4 bytes for int (float count)
    HEADER_FORMAT = "<Qi" 
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    while running:
        try:
            # 1. Read the complete header block
            header_bytes = b''
            while len(header_bytes) < HEADER_SIZE:
                remaining = client_socket.recv(HEADER_SIZE - len(header_bytes))
                if not remaining:
                    print("Connection closed by client during header reception")
                    return
                header_bytes += remaining

            # Unpack the little-endian header values
            timestamp, float_count = struct.unpack(HEADER_FORMAT, header_bytes)
            latest_timestamp = timestamp

            # 2. Read the compressed package of bits (the float data array)
            payload_size = float_count * 4  # 4 bytes per float
            payload_bytes = b''
            while len(payload_bytes) < payload_size:
                remaining = client_socket.recv(payload_size - len(payload_bytes))
                if not remaining:
                    print("Connection closed by client during payload reception")
                    return
                payload_bytes += remaining

            # Unpack the package of bits back into a native float list
            float_format = f"<{float_count}f"
            received_floats = list(struct.unpack(float_format, payload_bytes))

        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected.")
            break
        except Exception as e:
            print(f"Receiver error: {e}")
            traceback.print_exc()
            break
    
    print(f"Receiver for {addr} closing.")

def sender_thread(client_socket, addr):
    """Continuously pulls packages from the queue to send out to Unity.
    
    Expects data in the queue to be a tuple or list: (timestamp, [float_1, float_2, ...])
    """
    print(f"Sender started for {addr}")
    global running

    while running:
        try:
            # Blocks until an item is available in the queue
            queue_item = send_queue.get() 
            
            # Unpack queue format: tuple of (int/long timestamp, list of floats)
            timestamp, float_list = queue_item
            float_count = len(float_list)

            # 1. Pack the 12-byte header (ulong timestamp, int count)
            header_bytes = struct.pack("<Qi", timestamp, float_count)

            # 2. Convert float list straight into a compressed bit package 
            payload_bytes = struct.pack(f"<{float_count}f", *float_list)

            # 3. Transmit entire sequence cleanly
            client_socket.sendall(header_bytes + payload_bytes)
            send_queue.task_done()
            
        except queue.Empty:
            continue
        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected (pipe broken).")
            break
        except Exception as e:
            print(f"Sender error: {e}")
            traceback.print_exc()
            break

    print(f"Sender for {addr} closing.")

def handle_client(client_socket, addr):
    print(f'Got a connection from {str(addr)}')

    client_alive = threading.Event()
    client_alive.set()

    # Send an initial welcome message
    # client_socket.send(b'Server says connected') # interferes with corner processing in Unity

    # 1. Start the Receiver thread
    rx_thread = threading.Thread(target=receiver_thread, args=(client_socket, addr))
    rx_thread.start()

    # 2. Start the Sender thread
    tx_thread = threading.Thread(target=sender_thread, args=(client_socket, addr))
    tx_thread.start()

    dm_thread = threading.Thread(target=data_manager, args=(client_alive,))
    dm_thread.start()

    # Wait for both threads to finish (which happens only when an error occurs)
    rx_thread.join()
    tx_thread.join()

    client_alive.clear() 
    dm_thread.join()

    client_socket.close()
    print(f"Connection with {addr} closed.")

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.settimeout(60)

# Get local machine name
host = "127.0.0.1"
port = 65432

# Bind the socket to a public host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(5)
print('Server is listening on port %s...' % port)

try:
    while running:
        try:
            # Accept a connection
            client_socket, addr = server_socket.accept()

            # Create a new thread to handle the client
            client_thread = threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True)
            client_thread.start()

        except socket.timeout:
            continue

except Exception as e:
    print(f"program ended: {e}")
    running = False

finally:
    server_socket.close()