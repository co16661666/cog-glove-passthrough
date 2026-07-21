from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSpinBox, QFrame
)
from PyQt6.QtCore import Qt

class MotorTab(QWidget):
    """Per-finger vibration motor controls: on/off toggle, intensity field,
    and a live activity indicator that reflects whether each motor is
    currently vibrating."""

    FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    MIN_INTENSITY = 1
    MAX_INTENSITY = 123

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager

        # Per-finger widget references, keyed by finger name.
        self._toggles: dict[str, QCheckBox] = {}
        self._intensity_fields: dict[str, QSpinBox] = {}
        self._activity_labels: dict[str, QLabel] = {}

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Vibration Motor Control"))

        for finger in self.FINGERS:
            row = QFrame()
            h_layout = QHBoxLayout(row)
            h_layout.setContentsMargins(0, 0, 0, 0)

            # Finger name
            name_label = QLabel(finger)
            name_label.setFixedWidth(60)
            h_layout.addWidget(name_label)

            # On/off toggle
            toggle = QCheckBox("Enabled")
            toggle.stateChanged.connect(lambda state, f=finger: self._on_toggle_changed(f, state))
            self._toggles[finger] = toggle
            h_layout.addWidget(toggle)

            # Intensity number field (1-123)
            h_layout.addWidget(QLabel("Intensity:"))
            spin = QSpinBox()
            spin.setRange(self.MIN_INTENSITY, self.MAX_INTENSITY)
            spin.setValue(self.MIN_INTENSITY)
            spin.valueChanged.connect(lambda val, f=finger: self._on_intensity_changed(f, val))
            self._intensity_fields[finger] = spin
            h_layout.addWidget(spin)

            # Activity indicator
            activity_label = QLabel("Inactive")
            activity_label.setStyleSheet("color: gray;")
            activity_label.setFixedWidth(70)
            self._activity_labels[finger] = activity_label
            h_layout.addWidget(activity_label)

            h_layout.addStretch()
            layout.addWidget(row)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Internal signal handlers
    # ------------------------------------------------------------------ #
    def _on_toggle_changed(self, finger: str, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.dm.log_event(f"Motor {finger} toggled -> {'ON' if enabled else 'OFF'}")
        # This toggle is only a master switch; it does not drive the
        # activity indicator. Whatever process is actually firing the
        # motors should call update_activity() to reflect real state.

    def _on_intensity_changed(self, finger: str, value: int):
        self.dm.log_event(f"Motor {finger} intensity -> {value}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update_activity(self, active_motors: dict[str, bool]):
        """Update the widget to reflect which motors are currently vibrating.

        Args:
            active_motors: mapping of finger name -> bool, where True means
                the motor is currently vibrating (active) and False means
                it is idle. Fingers not present in the dict are left
                unchanged.
        """
        for finger, is_active in active_motors.items():
            label = self._activity_labels.get(finger)
            if label is None:
                continue  # unknown finger name, ignore
            if is_active:
                label.setText("Active")
                label.setStyleSheet("color: green; font-weight: bold;")
            else:
                label.setText("Inactive")
                label.setStyleSheet("color: gray;")

    def get_intensities(self) -> dict[str, int]:
        """Return the current intensity value (1-123) for each finger."""
        return {finger: spin.value() for finger, spin in self._intensity_fields.items()}

    def get_intensity(self, finger: str) -> int:
        """Return the current intensity value (1-123) for a single finger."""
        return self._intensity_fields[finger].value()

    def get_enabled_motors(self) -> dict[str, bool]:
        """Return the current on/off toggle state for each finger."""
        return {finger: toggle.isChecked() for finger, toggle in self._toggles.items()}