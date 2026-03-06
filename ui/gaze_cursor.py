"""
ui/gaze_cursor.py — Visual gaze cursor overlay.

A translucent circular cursor that follows the estimated gaze position.
Changes color and opacity based on tracking confidence.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient


class GazeCursor(QWidget):
    """
    Transparent overlay widget that displays the gaze cursor.
    Should be placed as a top-level overlay on the main window.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._confidence = 0.0
        self._visible = True
        self._radius = 22
        self._trail = []  # recent positions for trail effect
        self._max_trail = 5
    
    def update_position(self, x, y, confidence):
        """Update cursor position and confidence."""
        self._gaze_x = x
        self._gaze_y = y
        self._confidence = confidence
        
        # Maintain trail
        self._trail.append((x, y))
        if len(self._trail) > self._max_trail:
            self._trail.pop(0)
        
        self.update()
    
    def set_visible(self, visible):
        self._visible = visible
        self.update()
    
    def paintEvent(self, event):
        if not self._visible or self._confidence < 0.1:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw trail
        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(30 * (i + 1) / len(self._trail))
            trail_color = QColor(100, 160, 255, alpha)
            trail_r = int(self._radius * 0.4 * (i + 1) / len(self._trail))
            painter.setBrush(trail_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(tx, ty), trail_r, trail_r)
        
        # Main cursor - color based on confidence
        alpha = int(120 * min(1.0, self._confidence))
        
        if self._confidence > 0.6:
            base_color = QColor(80, 180, 255, alpha)  # blue = good tracking
            inner_color = QColor(150, 210, 255, alpha + 60)
        elif self._confidence > 0.3:
            base_color = QColor(255, 200, 60, alpha)   # yellow = medium
            inner_color = QColor(255, 230, 120, alpha + 60)
        else:
            base_color = QColor(255, 80, 80, alpha)    # red = poor tracking
            inner_color = QColor(255, 140, 140, alpha + 60)
        
        # Outer glow
        gradient = QRadialGradient(self._gaze_x, self._gaze_y, self._radius * 1.5)
        gradient.setColorAt(0, base_color)
        gradient.setColorAt(0.6, QColor(base_color.red(), base_color.green(),
                                         base_color.blue(), alpha // 3))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(self._gaze_x, self._gaze_y),
                           self._radius * 1.5, self._radius * 1.5)
        
        # Inner dot
        painter.setBrush(inner_color)
        painter.drawEllipse(QPointF(self._gaze_x, self._gaze_y),
                           self._radius * 0.35, self._radius * 0.35)
        
        # Crosshair ring
        ring_pen_color = QColor(base_color.red(), base_color.green(),
                                base_color.blue(), alpha + 40)
        painter.setPen(ring_pen_color)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(self._gaze_x, self._gaze_y),
                           self._radius, self._radius)
        
        painter.end()
