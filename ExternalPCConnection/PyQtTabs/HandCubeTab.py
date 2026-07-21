from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QCheckBox, QSlider, QLabel, 
                             QHBoxLayout,QGridLayout, QGroupBox)
import pyqtgraph.opengl as gl          # for 3D plots
from PyQt6.QtCore import Qt
import numpy as np

# --- 3D Plotting ---
class HandCubeTab(QWidget):
    """3D viewer for hand landmarks and the tracked cube, with visibility toggles."""

    # Standard 21-point hand landmark indices (per LANDMARK_NAMES in LeapConnector.py)
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    CUBE_HALF_EXTENT = 0.02  # meters

    def __init__(self, data_manager, thresholds, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.thresholds = thresholds
        self.show_fingertips_only = False
        self.show_cube = True
        self.show_hand = True
        self.latest_tripod = None # finger tips
        self.latest_cube_pos = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.chk_show_hand = QCheckBox("Show Hand")
        self.chk_show_hand.setChecked(True)
        self.chk_show_hand.stateChanged.connect(self._on_hand_toggle)
        controls.addWidget(self.chk_show_hand)

        self.chk_fingertips_only = QCheckBox("Fingertips Only")
        self.chk_fingertips_only.stateChanged.connect(self._on_fingertips_toggle)
        controls.addWidget(self.chk_fingertips_only)

        self.chk_show_cube = QCheckBox("Show Cube")
        self.chk_show_cube.setChecked(True)
        self.chk_show_cube.stateChanged.connect(self._on_cube_toggle)
        controls.addWidget(self.chk_show_cube)

        controls.addStretch()
        layout.addLayout(controls)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=0.5)
        grid = gl.GLGridItem()
        grid.setSize(1, 1)
        grid.setSpacing(0.05, 0.05)
        self.view.addItem(grid)
        layout.addWidget(self.view, stretch=1)

        self.hand_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0.2, 0.8, 1.0, 1.0), size=8)
        self.view.addItem(self.hand_scatter)

        # 12 edges of the cube, drawn as a wireframe via GLLinePlotItem(mode='lines')
        self.cube_lines = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0.4, 0.1, 1.0), width=2, mode='lines')
        self.view.addItem(self.cube_lines)

        thresh_group = QGroupBox("Proximity Thresholds")
        thresh_layout = QGridLayout()

        thresh_layout.addWidget(QLabel("Cube Proximity Threshold:"), 0, 0)
        self.sld_cube_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sld_cube_thresh.setRange(1, 500)  # millimeters
        self.sld_cube_thresh.setValue(int(self.thresholds.cube_proximity * 1000))
        self.sld_cube_thresh.valueChanged.connect(self._on_cube_thresh_slider)
        self.sld_cube_thresh.setMaximumHeight(16)
        thresh_layout.addWidget(self.sld_cube_thresh, 0, 1)
        self.lbl_cube_thresh_val = QLabel(f"{self.thresholds.cube_proximity:.3f} m")
        thresh_layout.addWidget(self.lbl_cube_thresh_val, 0, 2)
        thresh_layout.addWidget(QLabel("Min Tip-to-Cube Dist:"), 0, 3)
        self.lbl_cube_dist = QLabel("--")
        thresh_layout.addWidget(self.lbl_cube_dist, 0, 4)

        thresh_layout.addWidget(QLabel("Finger Proximity Threshold:"), 1, 0)
        self.sld_finger_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sld_finger_thresh.setRange(1, 500)  # millimeters
        self.sld_finger_thresh.setValue(int(self.thresholds.finger_proximity * 1000))
        self.sld_finger_thresh.valueChanged.connect(self._on_finger_thresh_slider)
        self.sld_finger_thresh.setMaximumHeight(16)
        thresh_layout.addWidget(self.sld_finger_thresh, 1, 1)
        self.lbl_finger_thresh_val = QLabel(f"{self.thresholds.finger_proximity:.3f} m")
        thresh_layout.addWidget(self.lbl_finger_thresh_val, 1, 2)
        thresh_layout.addWidget(QLabel("Max Finger-to-Finger Dist:"), 1, 3)
        self.lbl_finger_dist = QLabel("--")
        thresh_layout.addWidget(self.lbl_finger_dist, 1, 4)

        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group, stretch=0)

    def _on_cube_thresh_slider(self, value_mm):
        self.thresholds.cube_proximity = value_mm / 1000.0
        self.lbl_cube_thresh_val.setText(f"{self.thresholds.cube_proximity:.3f} m")

    def _on_finger_thresh_slider(self, value_mm):
        self.thresholds.finger_proximity = value_mm / 1000.0
        self.lbl_finger_thresh_val.setText(f"{self.thresholds.finger_proximity:.3f} m")

    def _on_hand_toggle(self, state):
        self.show_hand = bool(state)
        self.hand_scatter.setVisible(self.show_hand)

    def _on_fingertips_toggle(self, state):
        self.show_fingertips_only = bool(state)

    def _on_cube_toggle(self, state):
        self.show_cube = bool(state)
        self.cube_lines.setVisible(self.show_cube)

    def _update_threshold_readout(self):
        if self.latest_tripod is None:
            return

        thumb = np.array(self.latest_tripod['thumb'])
        index = np.array(self.latest_tripod['index'])
        middle = np.array(self.latest_tripod['middle'])

        finger_dist = max(
            np.linalg.norm(thumb - index),
            np.linalg.norm(thumb - middle),
            np.linalg.norm(index - middle),
        )
        finger_ok = finger_dist < self.thresholds.finger_proximity
        self.lbl_finger_dist.setText(f"{finger_dist:.3f} m")
        self.lbl_finger_dist.setStyleSheet(f"color: {'#2ecc71' if finger_ok else '#e74c3c'}; font-weight: bold;")

        if self.latest_cube_pos is not None:
            cube_dist = max(
                np.linalg.norm(thumb - self.latest_cube_pos),
                np.linalg.norm(index - self.latest_cube_pos),
                np.linalg.norm(middle - self.latest_cube_pos),
            )
            cube_ok = cube_dist < self.thresholds.cube_proximity
            self.lbl_cube_dist.setText(f"{cube_dist:.3f} m")
            self.lbl_cube_dist.setStyleSheet(f"color: {'#2ecc71' if cube_ok else '#e74c3c'}; font-weight: bold;")

    @staticmethod
    def _cube_corners(tx, ty, tz, rx, ry, rz, half_extent):
        v = np.array([rx, ry, rz])
        angle = np.linalg.norm(v)
        if angle < 1e-8:
            R = np.eye(3)
        else:
            axis = v / angle
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        h = half_extent
        local_corners = np.array([
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ])
        return (R @ local_corners.T).T + np.array([tx, ty, tz])

    @staticmethod
    def _cube_edges(corners):
        edge_idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        pts = []
        for i, j in edge_idx:
            pts.append(corners[i])
            pts.append(corners[j])
        return np.array(pts)
    
    @staticmethod
    def _unity_to_plot(x, y, z):
        """
        Remap Unity's Y-up, left-handed world coords into pyqtgraph's
        GLViewWidget convention, where GLGridItem lies flat in the XY
        plane and Z is up.

        Unity:      X = right,  Y = up,      Z = forward (depth)
        pyqtgraph:  X = right,  Y = forward,  Z = up
        """
        return (x, z, y)
    
    @staticmethod
    def _unity_to_plot_axis(x, y, z):
        """Same remap as _unity_to_plot, but for axis-angle rotation vectors,
        which are pseudovectors and need a sign correction under this
        axis swap (determinant of the swap is -1)."""
        return (x, z, -y)

    def update_view(self, latest_hands, latest_cube):
        # --- Render Hand ---
        if latest_hands is not None and self.show_hand:
            pts = []
            for hand in latest_hands:
                kp = hand.get("keypoints_list")
                if not kp:
                    continue

                if len(kp) > 12:
                    self.latest_tripod = {
                        'thumb': self._unity_to_plot(*kp[4]),
                        'index': self._unity_to_plot(*kp[8]),
                        'middle': self._unity_to_plot(*kp[12])
                    }
                
                indices = self.FINGERTIP_INDICES if self.show_fingertips_only else range(len(kp))
                for idx in indices:
                    if idx < len(kp):
                        pts.append(self._unity_to_plot(*kp[idx]))
            if pts:
                self.hand_scatter.setData(pos=np.array(pts))

        # --- Render Cube ---
        if latest_cube is not None:
            tx, ty, tz, rx, ry, rz = latest_cube
            plot_pos = self._unity_to_plot(tx, ty, tz)
            self.latest_cube_pos = np.array(plot_pos)
            if self.show_cube:
                plot_rot = self._unity_to_plot_axis(rx, ry, rz)
                corners = self._cube_corners(*plot_pos, *plot_rot, self.CUBE_HALF_EXTENT)
                self.cube_lines.setData(pos=self._cube_edges(corners))

        self._update_threshold_readout()