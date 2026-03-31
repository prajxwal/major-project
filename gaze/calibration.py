"""
gaze/calibration.py — 3-point horizontal calibration for eye gaze tracking.

The user looks at 3 dots: LEFT, CENTER, RIGHT.
- LEFT and RIGHT establish the gaze extremes for horizontal mapping.
- CENTER establishes the "neutral" zone where dwell selection happens.

This runs on every startup to ensure fresh, accurate calibration.
"""

import json
import os
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QFont, QPen


CALIBRATION_DIR = os.path.join(os.path.expanduser("~"), ".gazespeak")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")


class CalibrationScreen(QWidget):
    """
    Full-screen overlay for 3-point horizontal calibration.
    
    Phase 1: "Look LEFT"   → records gaze_x when looking left
    Phase 2: "Look CENTER"  → records gaze_x when looking straight ahead
    Phase 3: "Look RIGHT"  → records gaze_x when looking right
    
    Result: dict { "left_x", "center_x", "right_x" }
    
    Signals:
        calibration_complete(object): emitted with calibration data dict
        calibration_cancelled(): emitted if user presses Escape
    """
    
    calibration_complete = pyqtSignal(object)
    calibration_cancelled = pyqtSignal()
    
    # Calibration phases
    PHASE_INTRO = 0
    PHASE_LEFT = 1
    PHASE_CENTER = 2
    PHASE_RIGHT = 3
    PHASE_DONE = 4
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("background-color: #0a0a0f;")
        
        # Calibration state
        self._phase = self.PHASE_INTRO
        self._samples_per_point = 45  # ~1.5 sec at 30fps
        self._current_samples = []
        self._left_gaze_x = 0.0
        self._center_gaze_x = 0.0
        self._right_gaze_x = 0.0
        
        # Animation
        self._dot_radius = 24
        self._pulse_phase = 0.0
        self._collecting = False
        self._progress = 0.0
        
        # Timer for animation
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.setInterval(33)
        
        # Instruction label
        self._instruction_label = QLabel(self)
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Light))
        self._instruction_label.setStyleSheet("color: #b0b4d0; background: transparent;")
        
        # Sub-instruction label
        self._sub_label = QLabel(self)
        self._sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_label.setFont(QFont("Segoe UI", 14))
        self._sub_label.setStyleSheet("color: #5a5e78; background: transparent;")
        
        # Progress label
        self._progress_label = QLabel(self)
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setFont(QFont("Segoe UI", 13))
        self._progress_label.setStyleSheet("color: #50c878; background: transparent;")
    
    def start_calibration(self):
        """Begin the calibration sequence."""
        self.showFullScreen()
        self._phase = self.PHASE_INTRO
        self._current_samples = []
        self._collecting = False
        self._progress = 0.0
        
        self._instruction_label.setText("Gaze Calibration")
        self._sub_label.setText(
            "You will look at 3 dots: LEFT, CENTER, and RIGHT\n"
            "This maps your eye movement to the keyboard\n\n"
            "Press SPACE to start"
        )
        self._progress_label.setText("")
        self._reposition_labels()
        self._anim_timer.start()
    
    def _reposition_labels(self):
        w, h = self.width(), self.height()
        self._instruction_label.setGeometry(0, int(h * 0.25), w, 50)
        self._sub_label.setGeometry(0, int(h * 0.25) + 60, w, 120)
        self._progress_label.setGeometry(0, int(h * 0.25) + 190, w, 30)
    
    def receive_gaze_sample(self, gaze_x, gaze_y, confidence):
        """Called by the gaze tracker with each frame's iris ratios."""
        if not self._collecting:
            return
        
        if confidence > 0.25:
            self._current_samples.append(gaze_x)
            self._progress = len(self._current_samples) / self._samples_per_point
        
        if len(self._current_samples) >= self._samples_per_point:
            self._finish_current_phase()
    
    def _start_phase(self, phase):
        """Begin a calibration phase."""
        self._phase = phase
        self._current_samples = []
        self._progress = 0.0
        
        if phase == self.PHASE_LEFT:
            self._instruction_label.setText("👈  Look at the LEFT dot")
            self._sub_label.setText("Keep your gaze steady on the dot...")
            self._progress_label.setText("Step 1 of 3")
            QTimer.singleShot(800, self._begin_collecting)
            
        elif phase == self.PHASE_CENTER:
            self._instruction_label.setText("👁  Look at the CENTER dot")
            self._sub_label.setText("Look straight ahead at the center dot...")
            self._progress_label.setText("Step 2 of 3")
            QTimer.singleShot(800, self._begin_collecting)
            
        elif phase == self.PHASE_RIGHT:
            self._instruction_label.setText("Look at the RIGHT dot  👉")
            self._sub_label.setText("Keep your gaze steady on the dot...")
            self._progress_label.setText("Step 3 of 3")
            QTimer.singleShot(800, self._begin_collecting)
    
    def _begin_collecting(self):
        self._collecting = True
        self._current_samples = []
    
    def _finish_current_phase(self):
        """Process samples and advance to next phase."""
        self._collecting = False
        
        if not self._current_samples:
            self._instruction_label.setText("Failed — no data. Press SPACE to retry")
            self._phase = self.PHASE_INTRO
            return
        
        # Trimmed mean (remove outliers)
        sorted_samples = sorted(self._current_samples)
        trim = max(1, len(sorted_samples) // 5)
        trimmed = sorted_samples[trim:-trim] if trim < len(sorted_samples) // 2 else sorted_samples
        avg_x = np.mean(trimmed)
        
        if self._phase == self.PHASE_LEFT:
            self._left_gaze_x = avg_x
            self._progress_label.setText(f"✓ Left captured ({avg_x:.3f})")
            QTimer.singleShot(1000, lambda: self._start_phase(self.PHASE_CENTER))
            
        elif self._phase == self.PHASE_CENTER:
            self._center_gaze_x = avg_x
            self._progress_label.setText(f"✓ Center captured ({avg_x:.3f})")
            QTimer.singleShot(1000, lambda: self._start_phase(self.PHASE_RIGHT))
            
        elif self._phase == self.PHASE_RIGHT:
            self._right_gaze_x = avg_x
            self._progress_label.setText(f"✓ Right captured ({avg_x:.3f})")
            QTimer.singleShot(1000, self._compute_calibration)
    
    def _compute_calibration(self):
        """Compute and validate the 3-point calibration."""
        spread = abs(self._right_gaze_x - self._left_gaze_x)
        
        if spread < 0.05:
            self._instruction_label.setText("Calibration failed — not enough range")
            self._sub_label.setText(
                f"Left: {self._left_gaze_x:.3f}, Right: {self._right_gaze_x:.3f}\n"
                "Try looking further left/right.\n\nPress SPACE to retry"
            )
            self._phase = self.PHASE_INTRO
            return
        
        calibration_data = {
            "left_x": float(self._left_gaze_x),
            "center_x": float(self._center_gaze_x),
            "right_x": float(self._right_gaze_x),
            "type": "horizontal_3point",
        }
        
        self._save_calibration(calibration_data)
        
        self._phase = self.PHASE_DONE
        self._instruction_label.setText("✓ Calibration Complete!")
        self._sub_label.setText(
            f"Left: {self._left_gaze_x:.3f}  |  "
            f"Center: {self._center_gaze_x:.3f}  |  "
            f"Right: {self._right_gaze_x:.3f}"
        )
        self._progress_label.setText("Starting in 2 seconds...")
        self._anim_timer.stop()
        self.update()
        
        QTimer.singleShot(2000, lambda: self._finish(calibration_data))
    
    def _finish(self, cal_data):
        self.hide()
        self.calibration_complete.emit(cal_data)
    
    def _save_calibration(self, cal_data):
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(cal_data, f, indent=2)
    
    @staticmethod
    def load_calibration():
        """Load saved calibration — returns None to force fresh calibration."""
        # We always recalibrate on startup, so return None
        return None
    
    def _animate(self):
        self._pulse_phase += 0.08
        self.update()
    
    def paintEvent(self, event):
        """Draw the 3 calibration dots."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(10, 10, 15))
        
        if self._phase in (self.PHASE_INTRO, self.PHASE_DONE):
            painter.end()
            return
        
        pulse = 0.5 + 0.5 * np.sin(self._pulse_phase)
        
        # 3 dot positions
        left_x, dot_y = int(w * 0.08), int(h * 0.5)
        center_x = int(w * 0.50)
        right_x = int(w * 0.92)
        
        # Determine which dots are done
        left_done = self._phase in (self.PHASE_CENTER, self.PHASE_RIGHT, self.PHASE_DONE)
        center_done = self._phase in (self.PHASE_RIGHT, self.PHASE_DONE)
        
        # Draw dots
        self._draw_dot(painter, left_x, dot_y, pulse,
                       active=(self._phase == self.PHASE_LEFT), done=left_done)
        self._draw_dot(painter, center_x, dot_y, pulse,
                       active=(self._phase == self.PHASE_CENTER), done=center_done)
        self._draw_dot(painter, right_x, dot_y, pulse,
                       active=(self._phase == self.PHASE_RIGHT), done=False)
        
        # Connecting line
        painter.setPen(QPen(QColor(40, 44, 65), 2, Qt.PenStyle.DashLine))
        painter.drawLine(left_x + 35, dot_y, right_x - 35, dot_y)
        
        # Progress arc on active dot
        if self._collecting and self._progress > 0:
            if self._phase == self.PHASE_LEFT:
                ax = left_x
            elif self._phase == self.PHASE_CENTER:
                ax = center_x
            else:
                ax = right_x
            self._draw_progress_arc(painter, ax, dot_y, self._progress)
        
        painter.end()
    
    def _draw_dot(self, painter, cx, cy, pulse, active=False, done=False):
        glow_radius = int(self._dot_radius + 12 * pulse) if active else self._dot_radius
        
        gradient = QRadialGradient(cx, cy, glow_radius + 10)
        if done:
            gradient.setColorAt(0, QColor(80, 200, 120, 180))
            gradient.setColorAt(0.5, QColor(40, 150, 80, 80))
            gradient.setColorAt(1, QColor(20, 80, 40, 0))
        elif active:
            gradient.setColorAt(0, QColor(80, 140, 255, 220))
            gradient.setColorAt(0.5, QColor(40, 80, 200, 100))
            gradient.setColorAt(1, QColor(20, 40, 100, 0))
        else:
            gradient.setColorAt(0, QColor(120, 130, 160, 120))
            gradient.setColorAt(0.5, QColor(60, 70, 100, 50))
            gradient.setColorAt(1, QColor(30, 35, 50, 0))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(cx, cy), glow_radius + 10, glow_radius + 10)
        
        if done:
            core = QColor(80, 220, 120)
        elif active:
            core = QColor(120, 180, 255)
        else:
            core = QColor(80, 85, 110)
        
        painter.setBrush(core)
        painter.drawEllipse(QPoint(cx, cy), self._dot_radius // 2, self._dot_radius // 2)
    
    def _draw_progress_arc(self, painter, cx, cy, progress):
        radius = self._dot_radius + 8
        span = int(progress * 360 * 16)
        
        pen = QPen(QColor(80, 220, 140, 220), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        from PyQt6.QtCore import QRectF
        arc_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        painter.drawArc(arc_rect, 90 * 16, -span)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._anim_timer.stop()
            self._collecting = False
            self.hide()
            self.calibration_cancelled.emit()
        elif event.key() == Qt.Key.Key_Space:
            if self._phase == self.PHASE_INTRO:
                self._start_phase(self.PHASE_LEFT)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_labels()
