import numpy as np
import threading
import queue
import time

# --- Grasp Inference Thread ---
class GraspInferenceThread(threading.Thread):
    """
    Rule-based grasp detector. A grasp is flagged only when all three
    signals agree for the current frame:
      1. Force  - thumb fingertip force exceeds its own threshold AND
                  (index + middle) fingertip force, summed, exceeds a
                  pinch threshold.
      2. Proximity - thumb/index/middle fingertips are each close to the
                     cube AND close to one another (tripod pinch geometry).
      3. Flex   - thumb/index/middle flex sensors read as "bent".
    """

    # ff payload column layout: ["Thumb","Index","Middle","Ring","Pinky"]
    FORCE_THUMB, FORCE_INDEX, FORCE_MIDDLE = 0, 1, 2
    FLEX_THUMB, FLEX_INDEX, FLEX_MIDDLE = 5, 6, 7

    # keypoints_list indices, per LANDMARK_NAMES in LeapConnector.py
    IDX_THUMB_TIP, IDX_INDEX_TIP, IDX_MIDDLE_TIP = 4, 8, 12

    def __init__(self, data_manager, thresholds):
        super().__init__()
        self.dm = data_manager
        self.thresholds = thresholds
        self.running = True
        self.daemon = True

        # Caches - ff / hp / cube arrive independently and at different
        # rates, so we hold the latest of each rather than requiring all
        # three to land in the same tick.
        self.latest_force = None   # 10 values: [f_thumb..f_pinky, flex_thumb..flex_pinky]
        self.latest_hands = None   # list of hand dicts (LeapConnector.get_latest_hands())
        self.latest_cube = None    # [tx, ty, tz, rx, ry, rz]

        self.latest_prediction = False
        self.frame_id = 0

    def run(self):
        while self.running:
            updated = False

            try:
                while True:
                    _, *ff = self.dm.subscribers['inf_ff'].get_nowait()
                    self.latest_force = ff
                    updated = True
            except queue.Empty:
                pass

            try:
                while True:
                    _, hands = self.dm.subscribers['inf_hp'].get_nowait()
                    self.latest_hands = hands
                    updated = True
            except queue.Empty:
                pass

            try:
                while True:
                    _, cube = self.dm.subscribers['inf_cube'].get_nowait()  # (timestamp, [tx,ty,tz,rx,ry,rz])
                    self.latest_cube = cube
                    updated = True
            except queue.Empty:
                pass

            if not updated:
                time.sleep(0.005)
                continue

            prev = self.latest_prediction
            self.latest_prediction = self._evaluate_grasp()
            self.frame_id += 1

            if self.latest_prediction != prev:
                self.dm.broadcast_grasp('GRASPED' if self.latest_prediction else 'RELEASED')
                self.dm.log_event(f"Grasp {'GRASPED' if self.latest_prediction else 'RELEASED'}")

    def _evaluate_grasp(self) -> bool:
        if self.latest_force is None or self.latest_hands is None or self.latest_cube is None:
            return False

        thumb_tip = index_tip = middle_tip = None
        for hand in self.latest_hands:
            kp = hand.get("keypoints_list")
            if not kp or len(kp) <= self.IDX_MIDDLE_TIP:
                continue
            raw_thumb = np.array(kp[self.IDX_THUMB_TIP])
            raw_index = np.array(kp[self.IDX_INDEX_TIP])
            raw_middle = np.array(kp[self.IDX_MIDDLE_TIP])

            break  # only using the first tracked hand for now

        if thumb_tip is None or index_tip is None or middle_tip is None:
            return False

        # 1. Force
        thumb_force = self.latest_force[self.FORCE_THUMB]
        pinch_force = self.latest_force[self.FORCE_INDEX] + self.latest_force[self.FORCE_MIDDLE]
        force_ok = (thumb_force > self.thresholds.force_thumb and pinch_force > self.thresholds.force_pinch)

        # 2. Proximity
        cube_pos = np.array(self.latest_cube[:3])
        cube_dist_ok = all(
            np.linalg.norm(tip - cube_pos) < self.thresholds.cube_proximity
            for tip in (thumb_tip, index_tip, middle_tip)
        )
        finger_dist_ok = (
            np.linalg.norm(thumb_tip - index_tip) < self.thresholds.finger_proximity and
            np.linalg.norm(thumb_tip - middle_tip) < self.thresholds.finger_proximity and
            np.linalg.norm(index_tip - middle_tip) < self.thresholds.finger_proximity
        )
        proximity_ok = cube_dist_ok and finger_dist_ok

        # 3. Flex
        flex_ok = (
            self.latest_force[self.FLEX_THUMB] > self.thresholds.flex_bent and
            self.latest_force[self.FLEX_INDEX] > self.thresholds.flex_bent and
            self.latest_force[self.FLEX_MIDDLE] > self.thresholds.flex_bent
        )

        return bool(force_ok and proximity_ok and flex_ok)

    def stop(self):
        self.running = False