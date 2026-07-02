"""
ui/keyboard_widget.py — On-screen gaze keyboard with step-based navigation.

Renders a QWERTY keyboard grid with large, high-contrast keys. Navigation is
DISCRETE: looking left/right steps the highlight one key at a time (like arrow
keys). The highlight wraps across rows. When the user stops navigating (center
gaze), the dwell timer fills on the highlighted key to select it.
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
    On-screen keyboard with step-based highlight navigation.
    
    The highlight moves one key at a time (like arrow keys) controlled by
    the gaze zone. Dwell selects the highlighted key when the user looks
    at center (stops navigating).
    
    Signals:
        key_pressed(str): emitted when a letter key is selected via dwell
        special_key_pressed(str): emitted for action keys (BACKSPACE, etc.)
    """
    
    key_pressed = pyqtSignal(str)
    special_key_pressed = pyqtSignal(str)  # BACKSPACE, SPACE, SPEAK, CLEAR, etc.
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Build the navigation grid (2D list of key labels)
        self._grid = [
            list("QWERTYUIOP"),                         # row 0: 10 keys
            list("ASDFGHJKL"),                          # row 1: 9 keys
            list("ZXCVBNM"),                            # row 2: 7 keys
            [label for label, _ in SPECIAL_KEYS],       # row 3: 6 keys
        ]
        
        # Current highlight position
        self._highlight_row = 0
        self._highlight_col = 0
        
        # Dwell configuration
        self._dwell_time_ms = 800  # milliseconds to dwell before selection
        self._dwell_progress = 0.0  # 0.0 to 1.0
        self._navigating = False    # True when user is stepping (suppresses dwell)
        self._dwell_armed = False   # True only after user has done a deliberate gesture
        
        # Key geometry cache
        self._key_rects = {}  # key_label -> QRectF
        self._grid_rects = []  # [row][col] -> QRectF (parallel to self._grid)
        
        # Visual state
        self._selected_key = None
        self._selected_flash_timer = QTimer(self)
        self._selected_flash_timer.setSingleShot(True)
        self._selected_flash_timer.timeout.connect(self._clear_selection_flash)
        
        # Dwell timer (only advances when armed and not navigating)
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
    
    # ─── Grid Navigation ────────────────────────────────────────
    
    def get_grid(self):
        """Return the 2D grid structure."""
        return self._grid
    
    def get_row_count(self):
        """Number of keyboard rows."""
        return len(self._grid)
    
    def set_highlight(self, row, col):
        """Set the highlighted key position. Use row=-1 to clear highlight."""
        if row < 0 or col < 0:
            self._highlight_row = -1
            self._highlight_col = -1
            self._dwell_progress = 0.0
        else:
            if self._highlight_row != row or self._highlight_col != col:
                self._dwell_progress = 0.0  # reset dwell on position change
            self._highlight_row = row
            self._highlight_col = col
        self.update()
    
    def get_highlight(self):
        """Get the current (row, col) highlight position."""
        return self._highlight_row, self._highlight_col
    
    def get_highlighted_key(self):
        """Get the label of the currently highlighted key, or None."""
        r, c = self._highlight_row, self._highlight_col
        if 0 <= r < len(self._grid) and 0 <= c < len(self._grid[r]):
            return self._grid[r][c]
        return None
    
    def get_highlighted_rect(self):
        """Get the QRectF of the highlighted key (in widget coords), or None."""
        r, c = self._highlight_row, self._highlight_col
        if 0 <= r < len(self._grid_rects) and 0 <= c < len(self._grid_rects[r]):
            return self._grid_rects[r][c]
        return None
    
    def set_navigating(self, is_navigating):
        """Set whether the user is currently stepping through keys.
        When navigating, dwell is suppressed and disarmed."""
        self._navigating = is_navigating
        if is_navigating:
            self._dwell_armed = False
            self._dwell_progress = 0.0
    
    def arm_dwell(self):
        """Arm the dwell timer — called when user returns to CENTER after a gesture."""
        if not self._navigating:
            self._dwell_armed = True
            self._dwell_progress = 0.0
    
    def disarm_dwell(self):
        """Disarm the dwell timer — called at startup or when navigating."""
        self._dwell_armed = False
        self._dwell_progress = 0.0
    
    # ─── Dwell & Selection ──────────────────────────────────────
    
    def set_dwell_time(self, ms):
        """Set dwell time in milliseconds."""
        self._dwell_time_ms = max(300, min(3000, ms))
    
    def _update_dwell(self):
        """Advance dwell progress on the highlighted key (only when armed and not navigating)."""
        if self._navigating or not self._dwell_armed:
            self._dwell_progress = 0.0
            return
        
        key = self.get_highlighted_key()
        if key is None:
            self._dwell_progress = 0.0
            return
        
        # Advance dwell
        increment = 16.0 / self._dwell_time_ms
        self._dwell_progress = min(1.0, self._dwell_progress + increment)
        
        if self._dwell_progress >= 1.0:
            self._trigger_key(key)
            self._dwell_progress = 0.0
    
    def _trigger_key(self, key):
        """Handle key selection after dwell completes."""
        self._selected_key = key
        self._selected_flash_timer.start(300)
        
        # Disarm dwell to prevent immediate double-fire; re-arm after brief pause
        self._dwell_armed = False
        self._dwell_progress = 0.0
        QTimer.singleShot(700, self._maybe_rearm_dwell)
        
        # Check if it's a special key
        for label, action in SPECIAL_KEYS:
            if key == label:
                self.special_key_pressed.emit(action)
                return
        
        # Regular letter key
        self.key_pressed.emit(key)
    
    def _maybe_rearm_dwell(self):
        """Re-arm dwell after the post-selection pause, if still not navigating."""
        if not self._navigating:
            self._dwell_armed = True
            self._dwell_progress = 0.0
    
    def _clear_selection_flash(self):
        """Clear the green flash after selection."""
        self._selected_key = None
    
    # ─── Geometry ───────────────────────────────────────────────
    
    def _compute_key_rects(self):
        """Compute key rectangles based on current widget size."""
        self._key_rects.clear()
        self._grid_rects = []
        w, h = self.width(), self.height()
        
        padding = 6
        total_rows = len(self._grid)
        row_height = (h - padding * (total_rows + 1)) / total_rows
        
        for row_idx, row in enumerate(self._grid):
            row_rects = []
            num_keys = len(row)
            key_width = (w - padding * (num_keys + 1)) / num_keys
            
            # Center shorter rows
            row_offset = (w - (key_width * num_keys + padding * (num_keys - 1))) / 2
            
            for col_idx, key in enumerate(row):
                x = row_offset + col_idx * (key_width + padding)
                y = padding + row_idx * (row_height + padding)
                rect = QRectF(x, y, key_width, row_height)
                self._key_rects[key] = rect
                row_rects.append(rect)
            
            self._grid_rects.append(row_rects)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_key_rects()
    
    # ─── Rendering ──────────────────────────────────────────────
    
    def paintEvent(self, event):
        """Render the keyboard with highlighted key."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, self._bg_color)
        
        if not self._key_rects:
            self._compute_key_rects()
        
        highlighted_key = self.get_highlighted_key()
        
        for row_idx, row in enumerate(self._grid):
            for col_idx, key in enumerate(row):
                rect = self._grid_rects[row_idx][col_idx]
                is_highlighted = (row_idx == self._highlight_row and
                                  col_idx == self._highlight_col)
                is_selected = (key == self._selected_key)
                is_special = key in [label for label, _ in SPECIAL_KEYS]
                
                # Key background
                if is_selected:
                    bg = self._selected_color
                elif is_highlighted:
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
                
                # Key border — bright for highlighted key
                if is_highlighted:
                    border_color = self._dwell_ring_color
                    border_width = 2.5
                else:
                    border_color = QColor(60, 65, 85)
                    border_width = 0.5
                painter.setPen(QPen(border_color, border_width))
                painter.drawPath(path)
                
                # Key label
                text_color = (QColor(10, 10, 10) if is_selected
                              else QColor(255, 255, 255) if is_highlighted
                              else self._key_text_color)
                font_size = min(int(rect.height() * 0.35), 28)
                if is_special:
                    font_size = min(int(rect.height() * 0.4), 32)
                weight = QFont.Weight.Bold if is_highlighted else QFont.Weight.Medium
                painter.setFont(QFont("Segoe UI", font_size, weight))
                painter.setPen(text_color)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, key)
                
                # Dwell progress ring on highlighted key
                if is_highlighted and self._dwell_progress > 0 and not self._navigating:
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
    
    # ─── Mouse fallback (caregiver mode) ────────────────────────
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for caregiver mode."""
        pos = event.position()
        for row_idx, row in enumerate(self._grid):
            for col_idx, key in enumerate(row):
                rect = self._grid_rects[row_idx][col_idx]
                if rect.contains(QPointF(pos.x(), pos.y())):
                    self._trigger_key(key)
                    return
