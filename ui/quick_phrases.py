"""
ui/quick_phrases.py — Pre-built phrase bank for common ALS patient needs.

Categories: Pain/Comfort, Medical, Social, Yes/No, Daily.
One-tap selection triggers immediate TTS output.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont, QLinearGradient, QPainterPath


PHRASE_CATEGORIES = {
    "🔴 Urgent": [
        "I need help",
        "I'm in pain",
        "Call the nurse",
        "Call the doctor",
        "I can't breathe",
        "Emergency",
    ],
    "💊 Medical": [
        "I need medicine",
        "I need water",
        "I need to use the bathroom",
        "Adjust my position",
        "I need suction",
        "Change my position",
    ],
    "😊 Social": [
        "Thank you",
        "I love you",
        "How are you",
        "I'm doing well",
        "Tell me about your day",
        "I miss you",
    ],
    "👍 Quick Response": [
        "Yes",
        "No",
        "Maybe",
        "I don't know",
        "Please repeat",
        "I understand",
    ],
    "🏠 Comfort": [
        "I'm cold",
        "I'm hot",
        "I'm tired",
        "I want to rest",
        "Turn on the light",
        "Turn off the light",
        "I'm hungry",
        "I'm thirsty",
        "Open the window",
        "Close the window",
        "Too loud",
        "More pillow",
    ],
}


class QuickPhrasesPanel(QWidget):
    """
    Panel displaying categorized quick phrases for one-gaze selection.
    
    Signals:
        phrase_selected(str): emitted when a phrase is chosen
        back_requested(): emitted when user wants to go back to keyboard
    """
    
    phrase_selected = pyqtSignal(str)
    back_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._phrase_rects = []  # list of (phrase, QRectF)
        self._hover_index = -1
        self._dwell_progress = 0.0
        self._dwell_time_ms = 800
        self._selected_index = -1
        self._scroll_offset = 0
        
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        
        # Back button rect
        self._back_rect = QRectF()
        self._back_hovered = False
        self._back_dwell = 0.0
        
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
    
    def update_gaze_position(self, x, y):
        self._gaze_x = x
        self._gaze_y = y
    
    def set_dwell_time(self, ms):
        self._dwell_time_ms = max(300, min(3000, ms))
    
    def _compute_layout(self):
        """Compute all phrase button rectangles."""
        self._phrase_rects = []
        w, h = self.width(), self.height()
        padding = 8
        
        # Back button at top-left
        self._back_rect = QRectF(padding, padding, 100, 40)
        
        y_offset = 60 - self._scroll_offset
        cols = 3
        col_width = (w - padding * (cols + 1)) / cols
        row_height = 55
        
        for category, phrases in PHRASE_CATEGORIES.items():
            # Category header
            y_offset += 45
            
            for i, phrase in enumerate(phrases):
                col = i % cols
                x = padding + col * (col_width + padding)
                y = y_offset + (i // cols) * (row_height + padding)
                rect = QRectF(x, y, col_width, row_height)
                self._phrase_rects.append((phrase, rect, category))
            
            rows = (len(phrases) + cols - 1) // cols
            y_offset += rows * (row_height + padding) + padding
    
    def _get_hovered_index(self):
        for i, (phrase, rect, _) in enumerate(self._phrase_rects):
            if rect.contains(QPointF(self._gaze_x, self._gaze_y)):
                return i
        return -1
    
    def _update_dwell(self):
        # Check back button
        if self._back_rect.contains(QPointF(self._gaze_x, self._gaze_y)):
            self._back_hovered = True
            self._hover_index = -1
            self._back_dwell = min(1.0, self._back_dwell + 16.0 / self._dwell_time_ms)
            if self._back_dwell >= 1.0:
                self._back_dwell = 0.0
                self.back_requested.emit()
            return
        else:
            self._back_hovered = False
            self._back_dwell = 0.0
        
        idx = self._get_hovered_index()
        if idx == -1:
            self._hover_index = -1
            self._dwell_progress = 0.0
            return
        
        if idx != self._hover_index:
            self._hover_index = idx
            self._dwell_progress = 0.0
            return
        
        self._dwell_progress = min(1.0, self._dwell_progress + 16.0 / self._dwell_time_ms)
        
        if self._dwell_progress >= 1.0:
            self._selected_index = idx
            self._flash_timer.start(400)
            phrase = self._phrase_rects[idx][0]
            self.phrase_selected.emit(phrase)
            self._dwell_progress = 0.0
            self._hover_index = -1
    
    def _clear_flash(self):
        self._selected_index = -1
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_layout()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(12, 12, 20))
        
        if not self._phrase_rects:
            self._compute_layout()
        
        # Draw back button
        back_bg = QColor(60, 50, 80) if self._back_hovered else QColor(40, 35, 55)
        path = QPainterPath()
        path.addRoundedRect(self._back_rect, 8, 8)
        painter.fillPath(path, back_bg)
        painter.setPen(QColor(180, 170, 210))
        painter.setFont(QFont("Segoe UI", 14))
        painter.drawText(self._back_rect, Qt.AlignmentFlag.AlignCenter, "← Back")
        
        # Draw categories and phrases
        current_category = None
        for i, (phrase, rect, category) in enumerate(self._phrase_rects):
            # Category header
            if category != current_category:
                current_category = category
                header_rect = QRectF(rect.x(), rect.y() - 38, w, 30)
                painter.setPen(QColor(160, 165, 190))
                painter.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
                painter.drawText(header_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, category)
            
            is_hovered = (i == self._hover_index)
            is_selected = (i == self._selected_index)
            is_urgent = "Urgent" in category
            
            # Phrase button
            if is_selected:
                bg = QColor(60, 200, 120)
            elif is_hovered:
                bg = QColor(55, 45, 75) if not is_urgent else QColor(80, 40, 40)
            elif is_urgent:
                bg = QColor(55, 25, 30)
            else:
                bg = QColor(30, 30, 48)
            
            btn_path = QPainterPath()
            btn_path.addRoundedRect(rect, 10, 10)
            painter.fillPath(btn_path, bg)
            
            border = QColor(80, 160, 255, 150) if is_hovered else QColor(50, 50, 70)
            if is_urgent and not is_hovered:
                border = QColor(180, 60, 60, 100)
            painter.setPen(border)
            painter.drawPath(btn_path)
            
            text_color = QColor(10, 10, 10) if is_selected else QColor(210, 215, 230)
            painter.setPen(text_color)
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, phrase)
            
            # Dwell ring
            if is_hovered and self._dwell_progress > 0:
                center = rect.center()
                radius = min(rect.width(), rect.height()) / 2 - 3
                painter.setPen(QColor(80, 160, 255))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                arc_rect = QRectF(center.x() - radius, center.y() - radius,
                                  radius * 2, radius * 2)
                span = int(self._dwell_progress * 360 * 16)
                painter.drawArc(arc_rect, 90 * 16, -span)
        
        painter.end()
    
    def mousePressEvent(self, event):
        """Mouse click fallback."""
        if self._back_rect.contains(event.position()):
            self.back_requested.emit()
            return
        
        for i, (phrase, rect, _) in enumerate(self._phrase_rects):
            if rect.contains(event.position()):
                self._selected_index = i
                self._flash_timer.start(300)
                self.phrase_selected.emit(phrase)
                break
