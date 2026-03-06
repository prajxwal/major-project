"""
gaze/calibration.py — 9-point calibration system for eye gaze tracking.

Displays dots at known screen positions, records iris ratios at each point,
and computes an affine transformation matrix to map gaze → screen coordinates.
"""

import json
import os
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QFont


CALIBRATION_DIR = os.path.join(os.path.expanduser("~"), ".gazespeak")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")


class CalibrationScreen(QWidget):
    """
    Full-screen overlay that guides the user through 9-point calibration.
    
    Shows a pulsing dot at each calibration point. The gaze tracker records the
    iris ratios at each point. After all points are collected, it computes an
    affine transformation matrix.
    
    Signals:
        calibration_complete(np.ndarray): emitted with the 2x3 calibration matrix
        calibration_cancelled(): emitted if user presses Escape
    """
    
    calibration_complete = pyqtSignal(object)  # np.ndarray
    calibration_cancelled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("background-color: #0a0a0f;")
        
        # Calibration points as fractions of screen size
        # 9 points: corners, edge midpoints, and center
        self._cal_positions = [
            (0.1, 0.1),   # top-left
            (0.5, 0.1),   # top-center
            (0.9, 0.1),   # top-right
            (0.1, 0.5),   # middle-left
            (0.5, 0.5),   # center
            (0.9, 0.5),   # middle-right
            (0.1, 0.9),   # bottom-left
            (0.5, 0.9),   # bottom-center
            (0.9, 0.9),   # bottom-right
        ]
        
        self._current_point = 0
        self._samples_per_point = 30  # frames to average per calibration point
        self._current_samples = []
        self._collected_data = []  # list of (screen_x, screen_y, avg_gaze_x, avg_gaze_y)
        
        # Animation
        self._dot_radius = 20
        self._pulse_phase = 0.0
        self._collecting = False
        self._countdown = 0
        
        # Timer for animation
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.setInterval(33)  # ~30fps
        
        # Instruction label
        self._instruction_label = QLabel(self)
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Light))
        self._instruction_label.setStyleSheet("color: #aaaacc; background: transparent;")
        
        # Status label
        self._status_label = QLabel(self)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFont(QFont("Segoe UI", 14))
        self._status_label.setStyleSheet("color: #666688; background: transparent;")
        
    def start_calibration(self):
        """Begin the calibration sequence."""
        self.showFullScreen()
        self._current_point = 0
        self._collected_data = []
        self._current_samples = []
        self._collecting = False
        
        self._instruction_label.setText("Look at each glowing dot until it turns green\nPress SPACE to start")
        self._status_label.setText("Point 1 of 9")
        self._reposition_labels()
        
        self._anim_timer.start()
        self._waiting_for_start = True
    
    def _reposition_labels(self):
        """Position labels on screen."""
        w, h = self.width(), self.height()
        self._instruction_label.setGeometry(0, int(h * 0.35), w, 80)
        self._status_label.setGeometry(0, int(h * 0.35) + 90, w, 30)
        
    def receive_gaze_sample(self, gaze_x, gaze_y, confidence):
        """Called by the gaze tracker with each frame's iris ratios during calibration."""
        if not self._collecting:
            return
            
        if confidence > 0.3:  # only accept decent readings
            self._current_samples.append((gaze_x, gaze_y))
        
        if len(self._current_samples) >= self._samples_per_point:
            self._finish_current_point()
    
    def _finish_current_point(self):
        """Average samples for current point and move to next."""
        if self._current_samples:
            avg_x = np.mean([s[0] for s in self._current_samples])
            avg_y = np.mean([s[1] for s in self._current_samples])
            
            sx, sy = self._cal_positions[self._current_point]
            self._collected_data.append((sx, sy, avg_x, avg_y))
        
        self._current_samples = []
        self._collecting = False
        self._current_point += 1
        
        if self._current_point >= len(self._cal_positions):
            self._compute_calibration()
        else:
            self._status_label.setText(f"Point {self._current_point + 1} of 9")
            self._instruction_label.setText("Look at the dot...")
            # Brief pause before next point
            QTimer.singleShot(500, self._start_collecting)
    
    def _start_collecting(self):
        """Start collecting samples for the current point."""
        self._collecting = True
        self._current_samples = []
        self._instruction_label.setText("Hold your gaze steady...")
    
    def _compute_calibration(self):
        """Compute affine transformation from collected calibration data."""
        if len(self._collected_data) < 3:
            self._instruction_label.setText("Calibration failed — not enough data\nPress SPACE to retry")
            self._waiting_for_start = True
            return
        
        # Build matrices for least-squares affine fit
        # We want: [screen_x, screen_y] = M @ [gaze_x, gaze_y, 1]
        src_points = np.array([[d[2], d[3]] for d in self._collected_data])  # gaze ratios
        dst_points = np.array([[d[0], d[1]] for d in self._collected_data])  # screen fractions
        
        # Add homogeneous coordinate
        n = len(src_points)
        A = np.column_stack([src_points, np.ones(n)])  # Nx3
        
        # Solve for each output dimension separately
        # dst_x = a1*src_x + a2*src_y + a3
        # dst_y = b1*src_x + b2*src_y + b3
        mx, _, _, _ = np.linalg.lstsq(A, dst_points[:, 0], rcond=None)
        my, _, _, _ = np.linalg.lstsq(A, dst_points[:, 1], rcond=None)
        
        # Build 2x3 transformation matrix (we'll pad to 3x3 for convenience)
        calibration_matrix = np.array([
            [mx[0], mx[1], mx[2]],
            [my[0], my[1], my[2]],
            [0,     0,     1    ]
        ])
        
        # Save to disk
        self._save_calibration(calibration_matrix)
        
        self._instruction_label.setText("✓ Calibration complete!")
        self._status_label.setText("Starting in 2 seconds...")
        self._anim_timer.stop()
        self.update()
        
        QTimer.singleShot(2000, lambda: self._finish(calibration_matrix))
    
    def _finish(self, matrix):
        """Close calibration screen and emit result."""
        self.hide()
        self.calibration_complete.emit(matrix)
    
    def _save_calibration(self, matrix):
        """Save calibration matrix to disk."""
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        data = {"matrix": matrix.tolist()}
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_calibration():
        """Load saved calibration matrix from disk, or return None."""
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, "r") as f:
                    data = json.load(f)
                return np.array(data["matrix"])
            except (json.JSONDecodeError, KeyError, ValueError):
                return None
        return None
    
    def _animate(self):
        """Animation tick — pulse the calibration dot."""
        self._pulse_phase += 0.1
        self.update()
    
    def paintEvent(self, event):
        """Draw the calibration dot and progress."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Draw background
        painter.fillRect(0, 0, w, h, QColor(10, 10, 15))
        
        if hasattr(self, '_waiting_for_start') and self._waiting_for_start:
            return
        
        if self._current_point < len(self._cal_positions):
            # Draw the target dot
            sx, sy = self._cal_positions[self._current_point]
            cx, cy = int(sx * w), int(sy * h)
            
            # Pulsing glow
            pulse = 0.5 + 0.5 * np.sin(self._pulse_phase)
            glow_radius = int(self._dot_radius + 15 * pulse)
            
            # Outer glow
            gradient = QRadialGradient(cx, cy, glow_radius)
            if self._collecting:
                # Blue while collecting
                gradient.setColorAt(0, QColor(80, 140, 255, 200))
                gradient.setColorAt(0.5, QColor(40, 80, 200, 100))
                gradient.setColorAt(1, QColor(20, 40, 100, 0))
            else:
                # White/cyan idle
                gradient.setColorAt(0, QColor(200, 220, 255, 200))
                gradient.setColorAt(0.5, QColor(100, 140, 200, 100))
                gradient.setColorAt(1, QColor(50, 70, 100, 0))
            
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), glow_radius, glow_radius)
            
            # Inner dot
            core_color = QColor(120, 180, 255) if self._collecting else QColor(220, 230, 255)
            painter.setBrush(core_color)
            painter.drawEllipse(QPoint(cx, cy), self._dot_radius // 2, self._dot_radius // 2)
            
            # Progress ring (how many samples collected)
            if self._collecting and self._current_samples:
                progress = len(self._current_samples) / self._samples_per_point
                span_angle = int(progress * 360 * 16)
                painter.setPen(QColor(80, 200, 120, 200))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                rect_size = self._dot_radius * 2 + 10
                painter.drawArc(
                    cx - rect_size // 2, cy - rect_size // 2,
                    rect_size, rect_size,
                    90 * 16, -span_angle
                )
        
        painter.end()
    
    def keyPressEvent(self, event):
        """Handle key presses during calibration."""
        if event.key() == Qt.Key.Key_Escape:
            self._anim_timer.stop()
            self.hide()
            self.calibration_cancelled.emit()
        elif event.key() == Qt.Key.Key_Space:
            if hasattr(self, '_waiting_for_start') and self._waiting_for_start:
                self._waiting_for_start = False
                self._instruction_label.setText("Look at the dot...")
                self._reposition_labels()
                QTimer.singleShot(1000, self._start_collecting)
    
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self._reposition_labels()
