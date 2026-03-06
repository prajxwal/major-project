"""
ui/prediction_bar.py — Word suggestion strip.

Displays word predictions as gaze-selectable buttons above the keyboard.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont, QLinearGradient, QPainterPath
import math


class PredictionBar(QWidget):
    """
    Horizontal strip of word suggestion buttons.
    Supports both mouse click and gaze dwell selection.
    
    Signals:
        word_selected(str): emitted when a prediction is chosen
    """
    
    word_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._predictions = []
        self._button_rects = []  # list of (word, QRectF)
        self._hover_index = -1
        self._dwell_progress = 0.0
        self._dwell_time_ms = 800
        self._selected_index = -1
        
        # Gaze position
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        
        # Timers
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setInterval(16)
        self._dwell_timer.timeout.connect(self._update_dwell)
        self._dwell_timer.start()
        
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start()
        
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)
        
        self.setMinimumHeight(70)
        self.setMaximumHeight(80)
    
    def set_predictions(self, words):
        """Update the displayed word predictions."""
        self._predictions = words[:5]  # max 5
        self._compute_rects()
        self.update()
    
    def update_gaze_position(self, x, y):
        """Update gaze position in widget-local coordinates."""
        self._gaze_x = x
        self._gaze_y = y
    
    def set_dwell_time(self, ms):
        self._dwell_time_ms = max(300, min(3000, ms))
    
    def _compute_rects(self):
        """Compute button rectangles."""
        self._button_rects = []
        if not self._predictions:
            return
            
        w, h = self.width(), self.height()
        padding = 8
        num = len(self._predictions)
        btn_width = (w - padding * (num + 1)) / num
        
        for i, word in enumerate(self._predictions):
            x = padding + i * (btn_width + padding)
            rect = QRectF(x, padding, btn_width, h - 2 * padding)
            self._button_rects.append((word, rect))
    
    def _get_hovered_index(self):
        """Find which prediction button the gaze is on."""
        for i, (word, rect) in enumerate(self._button_rects):
            if rect.contains(QPointF(self._gaze_x, self._gaze_y)):
                return i
        return -1
    
    def _update_dwell(self):
        idx = self._get_hovered_index()
        
        if idx == -1:
            self._hover_index = -1
            self._dwell_progress = 0.0
            return
        
        if idx != self._hover_index:
            self._hover_index = idx
            self._dwell_progress = 0.0
            return
        
        increment = 16.0 / self._dwell_time_ms
        self._dwell_progress = min(1.0, self._dwell_progress + increment)
        
        if self._dwell_progress >= 1.0:
            self._select_word(idx)
            self._dwell_progress = 0.0
            self._hover_index = -1
    
    def _select_word(self, idx):
        if 0 <= idx < len(self._predictions):
            self._selected_index = idx
            self._flash_timer.start(300)
            self.word_selected.emit(self._predictions[idx])
    
    def _clear_flash(self):
        self._selected_index = -1
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_rects()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(14, 14, 22))
        
        if not self._button_rects:
            # Draw placeholder text
            painter.setPen(QColor(80, 80, 110))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                           "Start typing to see suggestions...")
            painter.end()
            return
        
        for i, (word, rect) in enumerate(self._button_rects):
            is_hovered = (i == self._hover_index)
            is_selected = (i == self._selected_index)
            
            # Background
            if is_selected:
                bg = QColor(60, 200, 120)
            elif is_hovered:
                bg = QColor(45, 50, 75)
            else:
                bg = QColor(30, 33, 50)
            
            path = QPainterPath()
            path.addRoundedRect(rect, 10, 10)
            
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, bg.lighter(110))
            gradient.setColorAt(1, bg)
            painter.fillPath(path, gradient)
            
            # Border
            border = QColor(80, 160, 255, 150) if is_hovered else QColor(50, 55, 75)
            painter.setPen(border)
            painter.drawPath(path)
            
            # Word text
            text_color = QColor(10, 10, 10) if is_selected else QColor(200, 210, 230)
            painter.setPen(text_color)
            painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Medium))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, word)
            
            # Dwell ring
            if is_hovered and self._dwell_progress > 0:
                center = rect.center()
                radius = min(rect.width(), rect.height()) / 2 - 4
                pen_color = QColor(80, 160, 255)
                painter.setPen(pen_color)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                arc_rect = QRectF(center.x() - radius, center.y() - radius,
                                  radius * 2, radius * 2)
                span = int(self._dwell_progress * 360 * 16)
                painter.drawArc(arc_rect, 90 * 16, -span)
        
        painter.end()
    
    def mousePressEvent(self, event):
        """Mouse click fallback for caregiver."""
        for i, (word, rect) in enumerate(self._button_rects):
            if rect.contains(event.position()):
                self._select_word(i)
                break
