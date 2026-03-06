"""
ui/keyboard_widget.py — On-screen gaze keyboard with dwell selection.

Renders a QWERTY keyboard grid with large, high-contrast keys. Each key shows
a dwell progress ring that fills while gaze stays on it. When dwell completes,
the key is selected with visual feedback.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import (QPainter, QColor, QFont, QPen, QLinearGradient,
                          QRadialGradient, QPainterPath)
import math


# Keyboard layouts
QWERTY_ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]

SPECIAL_KEYS = [
    ("⌫", "BACKSPACE"),
    ("␣", "SPACE"),
    ("🔊", "SPEAK"),
    ("⌧", "CLEAR"),
    ("💬", "PHRASES"),
    ("⚙", "SETTINGS"),
]


class KeyboardWidget(QWidget):
    """
    On-screen keyboard with gaze-driven dwell selection.
    
    Signals:
        key_pressed(str): emitted when a letter/action key is selected via dwell
    """
    
    key_pressed = pyqtSignal(str)
    special_key_pressed = pyqtSignal(str)  # BACKSPACE, SPACE, SPEAK, CLEAR, etc.
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Dwell configuration
        self._dwell_time_ms = 800  # milliseconds to dwell before selection
        self._current_hover_key = None
        self._dwell_start_time = 0
        self._dwell_progress = 0.0  # 0.0 to 1.0
        
        # Gaze position
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        
        # Key geometry cache
        self._key_rects = {}  # key_label -> QRectF
        
        # Visual state
        self._selected_key = None
        self._selected_flash_timer = QTimer(self)
        self._selected_flash_timer.setSingleShot(True)
        self._selected_flash_timer.timeout.connect(self._clear_selection_flash)
        
        # Dwell timer
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setInterval(16)  # ~60fps updates
        self._dwell_timer.timeout.connect(self._update_dwell)
        self._dwell_timer.start()
        
        # Animation timer for smooth rendering
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start()
        
        self.setMinimumHeight(300)
        
        # Colors
        self._bg_color = QColor(18, 18, 28)
        self._key_color = QColor(35, 38, 55)
        self._key_hover_color = QColor(50, 55, 80)
        self._key_text_color = QColor(220, 225, 240)
        self._dwell_ring_color = QColor(80, 160, 255)
        self._selected_color = QColor(60, 200, 120)
        self._special_key_color = QColor(45, 35, 60)
    
    def set_dwell_time(self, ms):
        """Set dwell time in milliseconds."""
        self._dwell_time_ms = max(300, min(3000, ms))
    
    def update_gaze_position(self, screen_x, screen_y):
        """Update the current gaze position (in widget coordinates)."""
        self._gaze_x = screen_x
        self._gaze_y = screen_y
    
    def _compute_key_rects(self):
        """Compute key rectangles based on current widget size."""
        self._key_rects.clear()
        w, h = self.width(), self.height()
        
        padding = 6
        total_rows = len(QWERTY_ROWS) + 1  # +1 for special keys row
        
        row_height = (h - padding * (total_rows + 1)) / total_rows
        
        for row_idx, row in enumerate(QWERTY_ROWS):
            num_keys = len(row)
            key_width = (w - padding * (num_keys + 1)) / num_keys
            
            # Center shorter rows
            row_offset = (w - (key_width * num_keys + padding * (num_keys - 1))) / 2
            
            for col_idx, key in enumerate(row):
                x = row_offset + col_idx * (key_width + padding)
                y = padding + row_idx * (row_height + padding)
                self._key_rects[key] = QRectF(x, y, key_width, row_height)
        
        # Special keys row
        special_row_y = padding + len(QWERTY_ROWS) * (row_height + padding)
        num_special = len(SPECIAL_KEYS)
        special_key_width = (w - padding * (num_special + 1)) / num_special
        
        for i, (label, action) in enumerate(SPECIAL_KEYS):
            x = padding + i * (special_key_width + padding)
            self._key_rects[label] = QRectF(x, special_row_y, special_key_width, row_height)
    
    def _get_key_at_position(self, x, y):
        """Find which key the given position falls on."""
        for key, rect in self._key_rects.items():
            if rect.contains(QPointF(x, y)):
                return key
        return None
    
    def _update_dwell(self):
        """Update dwell progress based on gaze position."""
        # Map gaze to local widget coordinates
        local_pos = self.mapFromGlobal(
            self.window().mapToGlobal(
                self.mapToParent(self.rect().topLeft())
            )
        )
        
        current_key = self._get_key_at_position(self._gaze_x, self._gaze_y)
        
        if current_key is None:
            self._current_hover_key = None
            self._dwell_progress = 0.0
            return
        
        if current_key != self._current_hover_key:
            # Gaze moved to a different key — reset dwell
            self._current_hover_key = current_key
            self._dwell_progress = 0.0
            return
        
        # Same key — advance dwell
        increment = 16.0 / self._dwell_time_ms  # fraction per tick
        self._dwell_progress = min(1.0, self._dwell_progress + increment)
        
        if self._dwell_progress >= 1.0:
            self._trigger_key(current_key)
            self._dwell_progress = 0.0
            self._current_hover_key = None
    
    def _trigger_key(self, key):
        """Handle key selection after dwell completes."""
        self._selected_key = key
        self._selected_flash_timer.start(300)
        
        # Check if it's a special key
        for label, action in SPECIAL_KEYS:
            if key == label:
                self.special_key_pressed.emit(action)
                return
        
        # Regular letter key
        self.key_pressed.emit(key)
    
    def _clear_selection_flash(self):
        """Clear the green flash after selection."""
        self._selected_key = None
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_key_rects()
    
    def paintEvent(self, event):
        """Render the keyboard."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, self._bg_color)
        
        if not self._key_rects:
            self._compute_key_rects()
        
        for key, rect in self._key_rects.items():
            is_hovered = (key == self._current_hover_key)
            is_selected = (key == self._selected_key)
            is_special = key in [label for label, _ in SPECIAL_KEYS]
            
            # Key background
            if is_selected:
                bg = self._selected_color
            elif is_hovered:
                bg = self._key_hover_color
            elif is_special:
                bg = self._special_key_color
            else:
                bg = self._key_color
            
            # Draw rounded rectangle
            path = QPainterPath()
            path.addRoundedRect(rect, 12, 12)
            
            # Subtle gradient
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, bg.lighter(115))
            gradient.setColorAt(1, bg)
            painter.fillPath(path, gradient)
            
            # Key border
            border_color = self._dwell_ring_color if is_hovered else QColor(60, 65, 85)
            painter.setPen(QPen(border_color, 1.5 if is_hovered else 0.5))
            painter.drawPath(path)
            
            # Key label
            painter.setPen(self._key_text_color if not is_selected else QColor(10, 10, 10))
            font_size = min(int(rect.height() * 0.35), 28)
            if is_special:
                font_size = min(int(rect.height() * 0.4), 32)
            painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.Medium))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, key)
            
            # Dwell progress ring
            if is_hovered and self._dwell_progress > 0:
                self._draw_dwell_ring(painter, rect, self._dwell_progress)
        
        painter.end()
    
    def _draw_dwell_ring(self, painter, rect, progress):
        """Draw circular dwell progress indicator around a key."""
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 4
        
        pen = QPen(self._dwell_ring_color, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        span_angle = int(progress * 360 * 16)
        arc_rect = QRectF(
            center.x() - radius, center.y() - radius,
            radius * 2, radius * 2
        )
        painter.drawArc(arc_rect, 90 * 16, -span_angle)
        
        # Glow effect at progress tip
        if progress > 0.05:
            angle_rad = math.radians(90 - progress * 360)
            tip_x = center.x() + radius * math.cos(angle_rad)
            tip_y = center.y() - radius * math.sin(angle_rad)
            
            glow = QRadialGradient(tip_x, tip_y, 8)
            glow.setColorAt(0, QColor(120, 180, 255, 200))
            glow.setColorAt(1, QColor(80, 140, 255, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(tip_x, tip_y), 8, 8)
    
    def handle_mouse_click(self, key):
        """Allow caregiver to click keys with mouse (fallback)."""
        self._trigger_key(key)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for caregiver mode."""
        key = self._get_key_at_position(event.position().x(), event.position().y())
        if key:
            self._trigger_key(key)
