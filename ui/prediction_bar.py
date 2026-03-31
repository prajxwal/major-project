"""
ui/prediction_bar.py — Word suggestion strip with step-based navigation.

Displays word predictions as selectable buttons above the keyboard.
Supports step-based highlight (like arrow keys) controlled by main.py,
plus mouse click fallback for caregivers.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont, QLinearGradient, QPainterPath, QPen
import math


class PredictionBar(QWidget):
    """
    Horizontal strip of word suggestion buttons.
    Supports step-based highlight navigation and dwell selection.
    
    Signals:
        word_selected(str): emitted when a prediction is chosen
    """
    
    word_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._predictions = []
        self._button_rects = []  # list of (word, QRectF)
        self._highlight_index = -1  # currently highlighted prediction
        self._dwell_progress = 0.0
        self._dwell_time_ms = 800
        self._selected_index = -1
        self._is_long_text = False
        self._navigating = False  # True when user is stepping
        
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
    
    # ─── Data ───────────────────────────────────────────────────
    
    def set_predictions(self, words):
        """Update the displayed word predictions."""
        self._predictions = words[:5]  # max 5
        self._is_long_text = any(len(w) > 15 for w in self._predictions)
        self._highlight_index = -1
        self._dwell_progress = 0.0
        self._compute_rects()
        self.update()
    
    def get_count(self):
        """Return the number of current predictions."""
        return len(self._predictions)
    
    # ─── Step Navigation ────────────────────────────────────────
    
    def set_highlight(self, index):
        """Set the highlighted prediction index. Use -1 to clear."""
        if self._highlight_index != index:
            self._dwell_progress = 0.0
        self._highlight_index = index
        self.update()
    
    def get_highlight(self):
        """Get the current highlight index."""
        return self._highlight_index
    
    def get_highlighted_rect(self):
        """Get the QRectF of the highlighted prediction (widget coords), or None."""
        idx = self._highlight_index
        if 0 <= idx < len(self._button_rects):
            return self._button_rects[idx][1]
        return None
    
    def set_navigating(self, is_navigating):
        """Set whether user is currently stepping (suppresses dwell)."""
        self._navigating = is_navigating
        if is_navigating:
            self._dwell_progress = 0.0
    
    # ─── Dwell ──────────────────────────────────────────────────
    
    def set_dwell_time(self, ms):
        self._dwell_time_ms = max(300, min(3000, ms))
    
    def _update_dwell(self):
        """Advance dwell on the highlighted prediction (only when not navigating)."""
        if self._navigating:
            self._dwell_progress = 0.0
            return
        
        idx = self._highlight_index
        if idx < 0 or idx >= len(self._predictions):
            self._dwell_progress = 0.0
            return
        
        increment = 16.0 / self._dwell_time_ms
        self._dwell_progress = min(1.0, self._dwell_progress + increment)
        
        if self._dwell_progress >= 1.0:
            self._select_word(idx)
            self._dwell_progress = 0.0
            self._highlight_index = -1
    
    def _select_word(self, idx):
        if 0 <= idx < len(self._predictions):
            self._selected_index = idx
            self._flash_timer.start(300)
            self.word_selected.emit(self._predictions[idx])
    
    def _clear_flash(self):
        self._selected_index = -1
    
    # ─── Geometry ───────────────────────────────────────────────
    
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
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_rects()
    
    # ─── Rendering ──────────────────────────────────────────────
    
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
            is_highlighted = (i == self._highlight_index)
            is_selected = (i == self._selected_index)
            
            # Background
            if is_selected:
                bg = QColor(60, 200, 120)
            elif is_highlighted:
                bg = QColor(45, 50, 75)
            else:
                bg = QColor(30, 33, 50)
            
            path = QPainterPath()
            path.addRoundedRect(rect, 10, 10)
            
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, bg.lighter(110))
            gradient.setColorAt(1, bg)
            painter.fillPath(path, gradient)
            
            # Border — bright for highlighted
            if is_highlighted:
                border = QColor(80, 160, 255, 200)
                border_width = 2.5
            else:
                border = QColor(50, 55, 75)
                border_width = 0.5
            painter.setPen(QPen(border, border_width))
            painter.drawPath(path)
            
            # Word text
            text_color = (QColor(10, 10, 10) if is_selected
                          else QColor(255, 255, 255) if is_highlighted
                          else QColor(200, 210, 230))
            font_size = 12 if self._is_long_text else 16
            weight = QFont.Weight.Bold if is_highlighted else QFont.Weight.Medium
            painter.setFont(QFont("Segoe UI", font_size, weight))
            painter.setPen(text_color)
            
            # Elide text if too long for the button
            metrics = painter.fontMetrics()
            elided = metrics.elidedText(word, Qt.TextElideMode.ElideRight, int(rect.width()) - 16)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, elided)
            
            # Dwell ring
            if is_highlighted and self._dwell_progress > 0 and not self._navigating:
                center = rect.center()
                radius = min(rect.width(), rect.height()) / 2 - 4
                pen = QPen(QColor(80, 160, 255), 3)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                arc_rect = QRectF(center.x() - radius, center.y() - radius,
                                  radius * 2, radius * 2)
                span = int(self._dwell_progress * 360 * 16)
                painter.drawArc(arc_rect, 90 * 16, -span)
        
        painter.end()
    
    # ─── Mouse fallback (caregiver mode) ────────────────────────
    
    def mousePressEvent(self, event):
        """Mouse click fallback for caregiver."""
        for i, (word, rect) in enumerate(self._button_rects):
            if rect.contains(event.position()):
                self._select_word(i)
                break
