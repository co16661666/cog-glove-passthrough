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

send_queue = queue.Queue(maxsize=1000)
def safe_put(item):
    while True:
        try:
            send_queue.put(item, block=False)
            break
        except queue.Full:
            try:
                # Drop the oldest stale frame to clear space
                send_queue.get_nowait()
                send_queue.task_done()
            except queue.Empty:
                pass

use_inference = True
use_hands = True

predictor = None
try:
    from classification.graspInference import GraspInference

    predictor = GraspInference()
    predictor.start()

except Exception as e:
    use_inference = False
    print(f"GraspInference initialization error: {e}")

leap_thread = None
try:
    from HandTracking.LeapConnector import LeapTrackingThread

    leap_thread = LeapTrackingThread()
    leap_thread.start()

except Exception as e:
    use_hands = False
    print(f"Hand tracking initialization error: {e}")

class HandDataManager(threading.Thread):
    def __init__(self, shutdown_event):
        super().__init__()
        self.latest_grasp_index = 0
        self.latest_hand_index = 0
        self.shutdown_event = shutdown_event
        self.daemon = True

    def run(self):
        global latest_timestamp
        print(f"Data manager started")
        while not self.shutdown_event.is_set():
            if use_inference and predictor is not None:
                if self.latest_grasp_index < predictor.frame_id:
                    self.latest_grasp_index = predictor.frame_id
                    val = [1.0] if predictor.latest_prediction else [0.0]
                    safe_put((latest_timestamp, val))

            if use_hands and leap_thread is not None:
                if self.latest_hand_index < leap_thread.get_latest_frame_id():
                    self.latest_hand_index = leap_thread.get_latest_frame_id()
                    safe_put((latest_timestamp, leap_thread.get_latest_hands_flattened()))

            time.sleep(0.006)

class TCPReceiverThread(threading.Thread):
    def __init__(self, client_socket, addr, shutdown_event):
        super().__init__()
        self.client_socket = client_socket
        self.addr = addr
        # 12 bytes total: 8 bytes for ulong (timestamp) + 4 bytes for int (float count)
        self.header_format = "<Qi" 
        self.header_size = struct.calcsize(self.header_format)
        self.shutdown_event = shutdown_event
        self.daemon = True

    def run(self):
        global latest_timestamp
        print(f"Receiver started for {self.addr}")
        while not self.shutdown_event.is_set():
            try:
                # 1. Read the complete header block
                header_bytes = b''
                while len(header_bytes) < self.header_size:
                    try:
                        remaining = self.client_socket.recv(self.header_size - len(header_bytes))
                        if not remaining:
                            print("Connection closed by client during header reception")
                            return
                        header_bytes += remaining
                    except socket.timeout:
                        # Allow the loop to continue and check shutdown_event
                        if self.shutdown_event.is_set():
                            return
                        continue

                # Unpack the little-endian header values
                timestamp, float_count = struct.unpack(self.header_format, header_bytes)
                latest_timestamp = timestamp

                # 2. Read the compressed package of bits (the float data array)
                payload_size = float_count * 4  # 4 bytes per float
                payload_bytes = b''
                while len(payload_bytes) < payload_size:
                    try:
                        remaining = self.client_socket.recv(payload_size - len(payload_bytes))
                        if not remaining:
                            print("Connection closed by client during payload reception")
                            return
                        payload_bytes += remaining
                    except socket.timeout:
                        if self.shutdown_event.is_set():
                            return
                        continue

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
        
        print(f"Receiver for {self.addr} closing.")

class TCPSenderThread(threading.Thread):
    def __init__(self, client_socket, addr, shutdown_event):
        super().__init__()
        self.client_socket = client_socket
        self.client_socket.settimeout(0.1)
        self.addr = addr
        self.shutdown_event = shutdown_event
        self.daemon = True

    def run(self):
        """
        Continuously pulls packages from the queue to send out to Unity.
        Expects data in the queue to be a tuple or list: (timestamp, [float_1, float_2, ...])
        """
        print(f"Sender started for {self.addr}")
        while not self.shutdown_event.is_set():
            try:
                # Blocks until an item is available in the queue
                queue_item = send_queue.get(timeout=0.1) 
                
                # Unpack queue format: tuple of (int/long timestamp, list of floats)
                timestamp, float_list = queue_item
                float_count = len(float_list)

                # 1. Pack the 12-byte header (ulong timestamp, int count)
                header_bytes = struct.pack("<Qi", timestamp, float_count)

                # 2. Convert float list straight into a compressed bit package 
                payload_bytes = struct.pack(f"<{float_count}f", *float_list)

                # 3. Transmit entire sequence cleanly
                self.client_socket.sendall(header_bytes + payload_bytes)
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

        print(f"Sender for {self.addr} closing.")

class TcpServer(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        # Create a socket object
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.settimeout(0.1)

        # Get local machine name
        self.host = host # "127.0.0.1"
        self.port = port # 65432

        # Bind the socket to a public host and port
        self.server_socket.bind((host, port))

        # Listen for incoming connections
        self.server_socket.listen(5)
        print('Server is listening on port %s...' % port)
        self.running = True
        self.daemon = True

    def handle_client(self, client_socket, addr):
        print(f'Got a connection from {str(addr)}')

        # Clear queue
        with send_queue.mutex:
            send_queue.queue.clear()

        shutdown_event = threading.Event()

        # Send an initial welcome message
        # client_socket.send(b'Server says connected') # interferes with corner processing in Unity

        # 1. Start the Receiver thread
        rx_thread = TCPReceiverThread(client_socket, addr, shutdown_event)
        rx_thread.start()

        # 2. Start the Sender thread
        tx_thread = TCPSenderThread(client_socket, addr, shutdown_event)
        tx_thread.start()

        dm_thread = HandDataManager(shutdown_event)
        dm_thread.start()

        while self.running:
            if not rx_thread.is_alive(): break
            if not tx_thread.is_alive(): break
            time.sleep(0.1)
        
        shutdown_event.set()

        rx_thread.join(timeout=1.0)
        tx_thread.join(timeout=1.0)
        dm_thread.join(timeout=1.0)
        client_socket.close()
        print(f"Connection with {addr} closed.")

    def run(self):
        try:
            while self.running:
                try:
                    # Accept a connection
                    client_socket, addr = self.server_socket.accept()

                    # Create a new thread to handle the client
                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True)
                    client_thread.start()

                except socket.timeout:
                    continue

        except Exception as e:
            print(f"program ended: {e}")
            self.running = False

        finally:
            self.server_socket.close()

if __name__ == '__main__':
    server = TcpServer("127.0.0.1", 65432)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C).")
    finally:
        if use_hands and leap_thread is not None:
            leap_thread.stop()
        if use_inference and predictor is not None:
            predictor.stop()

        server.running = False # Ensure the server loop terminates
        # Any other cleanup for the main server goes here