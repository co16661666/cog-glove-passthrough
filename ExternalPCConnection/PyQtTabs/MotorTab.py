from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt


class MotorTab(QWidget):
    """Placeholder controls for per-finger vibration motor intensity."""

    FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Vibration Motor Control (Placeholder)"))
        for finger in self.FINGERS:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(finger))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 255)
            slider.valueChanged.connect(lambda val, f=finger: self.dm.log_event(f"Motor {f} -> {val}"))
            h_layout.addWidget(slider)
            layout.addLayout(h_layout)