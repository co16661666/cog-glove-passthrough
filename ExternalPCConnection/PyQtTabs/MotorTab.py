from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSpinBox, QFrame, QPushButton
)
from PyQt6.QtCore import Qt, QTimer

class MotorTab(QWidget):
    """Per-finger vibration motor controls: on/off toggle, pattern field,
    and a live activity indicator that reflects whether each motor is
    currently vibrating."""

    MIN_INTENSITY = 1
    MAX_INTENSITY = 123

    def __init__(self, data_manager, test_callback=None, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.test_callback = test_callback

        # Per-finger widget references, keyed by finger name.
        self._toggles: dict[str, QCheckBox] = {}
        self._pattern_fields: dict[str, QSpinBox] = {}
        self._activity_labels: dict[str, QLabel] = {}

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Vibration Motor Control"))
        btn_test_all = QPushButton("Test All Enabled (1s)")
        btn_test_all.clicked.connect(self._on_test_all_clicked)
        header_layout.addWidget(btn_test_all)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        for finger in self.dm.Finger:
            row = QFrame()
            h_layout = QHBoxLayout(row)
            h_layout.setContentsMargins(0, 0, 0, 0)

            # Finger name label
            name_label = QLabel(finger.name.capitalize())
            name_label.setFixedWidth(60)
            h_layout.addWidget(name_label)

            # On/off toggle
            toggle = QCheckBox("Enabled")
            toggle.setChecked(True)
            toggle.stateChanged.connect(lambda state, f=finger: self._on_toggle_changed(f, state))
            self._toggles[finger] = toggle
            h_layout.addWidget(toggle)

            # Intensity field
            h_layout.addWidget(QLabel("Intensity:"))
            spin = QSpinBox()
            spin.setRange(self.MIN_INTENSITY, self.MAX_INTENSITY)
            spin.setValue(self.MIN_INTENSITY)
            spin.valueChanged.connect(lambda val, f=finger: self._on_pattern_changed(f, val))
            self._pattern_fields[finger] = spin
            h_layout.addWidget(spin)

            # Test button
            btn_test = QPushButton("Test")
            btn_test.clicked.connect(lambda _, f=finger: self._on_test_single_clicked(f))
            h_layout.addWidget(btn_test)

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
    # Motor test handlers
    # ------------------------------------------------------------------ #
    def _on_test_single_clicked(self, finger):
        if self.test_callback:
            self._trigger_test({f: (f == finger) for f in self.dm.Finger}, 500)

    def _on_test_all_clicked(self):
        if self.test_callback:
            self._trigger_test(self.get_enabled_motors(), 1000)

    def _trigger_test(self, targets: dict[str, bool], duration_ms: int):
        self.update_activity(targets)
        if self.test_callback is not None:
            self.test_callback(targets, self.get_patterns(), duration_ms)
            QTimer.singleShot(duration_ms, lambda: self.update_activity({f: False for f, active in targets.items() if active}))

    # ------------------------------------------------------------------ #
    # Internal signal handlers
    # ------------------------------------------------------------------ #
    def _on_toggle_changed(self, finger, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.dm.log_event(f"Motor {finger.name} toggled -> {'ON' if enabled else 'OFF'}")

    def _on_pattern_changed(self, finger, value: int):
        self.dm.log_event(f"Motor {finger.name} pattern -> {value}")

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

    def get_patterns(self) -> dict[str, int]:
        """Return the current pattern value (1-123) for each finger."""
        return {finger: spin.value() for finger, spin in self._pattern_fields.items()}

    def get_pattern(self, finger: str) -> int:
        """Return the current pattern value (1-123) for a single finger."""
        return self._pattern_fields[finger].value()

    def get_enabled_motors(self) -> dict[str, bool]:
        """Return the current on/off toggle state for each finger."""
        return {finger: toggle.isChecked() for finger, toggle in self._toggles.items()}