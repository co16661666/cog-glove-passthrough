import leap
import threading
import time
import queue
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# MediaPipe-style landmark indices (for reference)
# ---------------------------------------------------------------------------
#  0: wrist
#  1: thumb_cmc   2: thumb_mcp   3: thumb_ip    4: thumb_tip
#  5: index_mcp   6: index_pip   7: index_dip   8: index_tip
#  9: mid_mcp    10: mid_pip    11: mid_dip    12: mid_tip
# 13: ring_mcp   14: ring_pip   15: ring_dip   16: ring_tip
# 17: pinky_mcp  18: pinky_pip  19: pinky_dip  20: pinky_tip
# ---------------------------------------------------------------------------

# Ordered label list for the 21 landmarks (index == landmark id)
LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",  "thumb_mcp",  "thumb_ip",   "thumb_tip",
    "index_mcp",  "index_pip",  "index_dip",  "index_tip",
    "mid_mcp",    "mid_pip",    "mid_dip",    "mid_tip",
    "ring_mcp",   "ring_pip",   "ring_dip",   "ring_tip",
    "pinky_mcp",  "pinky_pip",  "pinky_dip",  "pinky_tip",
]


def _vec3(v) -> tuple[float, float, float]:
    """Convert a Leap Vector to a plain (x, y, z) tuple."""
    return (v.x, v.y, v.z)


def _extract_21_keypoints(hand) -> dict[str, tuple[float, float, float]]:
    """
    Return the 21 hand landmarks as a dict {landmark_name: (x, y, z)}.

    Leap bone layout per digit (4 bones, 0-indexed):
        bones[0] = Metacarpal   prev_joint → next_joint
        bones[1] = Proximal     prev_joint → next_joint
        bones[2] = Intermediate prev_joint → next_joint
        bones[3] = Distal       prev_joint → next_joint (= fingertip)

    Joint derivation
    ----------------
    Wrist          : hand.arm.next_joint
    Thumb CMC      : digits[0].bones[0].prev_joint
    Thumb MCP      : digits[0].bones[0].next_joint   (= bones[1].prev_joint)
    Thumb IP       : digits[0].bones[1].next_joint   (= bones[2].prev_joint)
    Thumb TIP      : digits[0].bones[3].next_joint
    Finger MCP     : digits[n].bones[0].next_joint   (n = 1..4)
    Finger PIP     : digits[n].bones[1].next_joint
    Finger DIP     : digits[n].bones[2].next_joint
    Finger TIP     : digits[n].bones[3].next_joint
    """
    kp: dict[str, tuple[float, float, float]] = {}

    # ── 0: Wrist ──────────────────────────────────────────────────────────
    kp["wrist"] = _vec3(hand.arm.next_joint)

    # ── 1-4: Thumb (digit index 0) ─────────────────────────────────────────
    thumb_bones = hand.digits[0].bones
    kp["thumb_cmc"] = _vec3(thumb_bones[0].prev_joint)   # base of metacarpal
    kp["thumb_mcp"] = _vec3(thumb_bones[0].next_joint)   # knuckle
    kp["thumb_ip"]  = _vec3(thumb_bones[1].next_joint)   # interphalangeal
    kp["thumb_tip"] = _vec3(thumb_bones[3].next_joint)   # fingertip

    # ── 5-20: Index / Middle / Ring / Pinky (digit indices 1-4) ────────────
    FINGER_PREFIXES = ["index", "mid", "ring", "pinky"]
    for digit_idx, prefix in enumerate(FINGER_PREFIXES, start=1):
        bones = hand.digits[digit_idx].bones
        kp[f"{prefix}_mcp"] = _vec3(bones[0].next_joint)  # 1st knuckle
        kp[f"{prefix}_pip"] = _vec3(bones[1].next_joint)  # 2nd knuckle
        kp[f"{prefix}_dip"] = _vec3(bones[2].next_joint)  # 3rd knuckle
        kp[f"{prefix}_tip"] = _vec3(bones[3].next_joint)  # fingertip

    return kp


class LeapTrackingThread(threading.Thread):

    def __init__(self):
        super().__init__()
        self.listener = self.MyListener(self)
        self.connection = leap.Connection()
        self.connection.add_listener(self.listener)
        self.running = True
        self.daemon = True  # Automatically dies when the main script ends

        # Shared data storage
        self.latest_hands_data: list[dict] = []
        self.frame_id = 0

        self.data_queue = queue.Queue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filepath = f"leap_data_{timestamp}.bin"

        self.writer_thread = threading.Thread(target=self._disk_writer_worker, daemon=True)
        self.writer_thread.start()

    class MyListener(leap.Listener):
        def __init__(self, outer_thread):
            super().__init__()
            self.outer = outer_thread

        def on_connection_event(self, event):
            print("[Leap] Connected to Ultraleap Service.")

        def on_device_event(self, event):
            try:
                with event.device.open():
                    info = event.device.get_info()
            except leap.LeapCannotOpenDeviceError:
                info = event.device.get_info()
            print(f"[Leap] Found device {info.serial}")

        def on_tracking_event(self, event):
            timestamp = event.tracking_frame_id
            current_hands = []

            for hand in event.hands:
                hand_type = "left" if str(hand.type) == "HandType.Left" else "right"

                # All 21 landmarks as a named dict
                keypoints = _extract_21_keypoints(hand)

                # Ordered list of (x, y, z) tuples – index matches LANDMARK_NAMES
                keypoints_list = [keypoints[name] for name in LANDMARK_NAMES]

                current_hands.append({
                    "id":             hand.id,
                    "type":           hand_type,
                    "palm_position":  _vec3(hand.palm.position),
                    # dict access:  data["keypoints"]["index_tip"]
                    "keypoints":      keypoints,
                    # index access: data["keypoints_list"][8]  → index tip
                    "keypoints_list": keypoints_list
                })

                flat_row = [float(timestamp), float(hand.id), 0.0 if hand_type == "left" else 1.0]
                for xyz in keypoints_list:
                    flat_row.extend(xyz)
                self.outer.data_queue.put(flat_row)

            self.outer.latest_hands_data = current_hands
            self.outer.frame_id += 1

    def _disk_writer_worker(self):
        """Dedicated thread to handle binary file serialization safely."""
        all_records = []
        batch_size = 120  # Flush roughly every 1 second at 120Hz
        
        while self.running or not self.data_queue.empty():
            try:
                record = self.data_queue.get(timeout=0.5)
                all_records.append(record)
                self.data_queue.task_done()
                
                if len(all_records) >= batch_size:
                    self._flush_to_disk(all_records)
                    all_records.clear()
            except queue.Empty:
                continue
                
        if all_records:
            self._flush_to_disk(all_records)

    def _flush_to_disk(self, records):
        """Appends rows to binary file instantly with zero string parsing overhead."""
        # Convert batch to a continuous float64 block
        arr = np.array(records, dtype=np.float64)
        with open(self.output_filepath, "ab") as f:
            arr.tofile(f)

    def run(self):
        print("[Leap] Thread started.")
        with self.connection.open():
            self.connection.set_tracking_mode(leap.TrackingMode.HMD) # HMD = Head Mounted Device, also ScreenTop or Desktop
            while self.running:
                time.sleep(0.01)
        print("[Leap] Thread stopped.")

    def get_latest_frame_id(self):
        return self.frame_id

    def get_latest_hands(self) -> list[dict]:
        """Return the most recent hand data captured by the listener."""
        return self.latest_hands_data

    def get_latest_hands_flattened(self) -> list[float]:
        floats: list[float] = [float(len(self.latest_hands_data))]
        for hand in self.latest_hands_data:
            floats.append(0.0 if hand["type"] == "left" else 1.0)
            for xyz in hand["keypoints_list"]:
                floats.extend(xyz)
        time.sleep(0.011) # Limit at ~90 Hz
        return floats

    def stop(self):
        self.running = False