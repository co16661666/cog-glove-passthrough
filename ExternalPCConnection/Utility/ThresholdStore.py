# --- Tunable thresholds (raw sensor units unless noted) ----------------
FORCE_THUMB_THRESHOLD       = 50
FORCE_PINCH_THRESHOLD       = 80     # index + middle summed
FLEX_BENT_THRESHOLD         = 500
CUBE_PROXIMITY_THRESHOLD    = 0.05   # meters
FINGER_PROXIMITY_THRESHOLD  = 0.05

class ThresholdStore:
    """Shared, GIL-safe container for live-tunable thresholds."""
    def __init__(self):
        self.force_thumb = FORCE_THUMB_THRESHOLD
        self.force_pinch = FORCE_PINCH_THRESHOLD
        self.flex_bent = FLEX_BENT_THRESHOLD
        self.cube_proximity = CUBE_PROXIMITY_THRESHOLD
        self.finger_proximity = FINGER_PROXIMITY_THRESHOLD