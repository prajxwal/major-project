"""
ui/conversation_panel.py — Binary-choice conversation UI for GazeSpeak.

Displays two large answer cards (LEFT / RIGHT) that the patient selects
by looking left or right. Features dwell progress rings, conversation
trail breadcrumbs, and action buttons for more options, typing, and back.
"""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QFont, QPen, QLinearGradient,
    QRadialGradient, QPainterPath,
)


class ConversationPanel(QWidget):
    """
    Binary-choice answer panel for LLM-powered conversations.

    The patient looks LEFT or RIGHT to select one of two answer cards.
    Dwell selection activates when gaze stays in a zone.

    Signals:
        answer_selected(str): emitted when the patient selects an answer
        more_options_requested(): emitted when "More Options" is chosen
        type_instead_requested(): emitted to switch to keyboard mode
        back_requested(): emitted to undo last selection
        speak_now_requested(): emitted to speak composed response early
    """

    answer_selected = pyqtSignal(str)
    more_options_requested = pyqtSignal()
    type_instead_requested = pyqtSignal()
    back_requested = pyqtSignal()
    speak_now_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current options
        self._left_text = ""
        self._right_text = ""
        self._question = ""
        self._selections: list[str] = []  # trail of prior selections
        self._is_loading = False
        self._is_final = False
        self._composed_response = ""

        # Gaze / dwell state
        self._current_zone = "CENTER"  # "LEFT", "CENTER", "RIGHT"
        self._dwell_progress = 0.0
        self._dwell_time_ms = 1200  # slightly longer than keyboard for safety
        self._selected_side = None  # "LEFT" or "RIGHT" flash
        self._navigating = False

        # Card geometry (computed in _compute_layout)
        self._left_card_rect = QRectF()
        self._right_card_rect = QRectF()
        self._vs_rect = QRectF()
        self._question_rect = QRectF()
        self._trail_rect = QRectF()
        self._action_rects: list[tuple[str, str, QRectF]] = []
        # (label, action_id, rect)

        # Timers
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setInterval(16)
        self._dwell_timer.timeout.connect(self._update_dwell)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start()

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

        # Pulse animation
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(33)
        self._pulse_timer.timeout.connect(self._advance_pulse)

        # Colors
        self._bg = QColor(10, 10, 18)
        self._card_left_color = QColor(30, 45, 80)
        self._card_right_color = QColor(55, 30, 65)
        self._card_hover_left = QColor(40, 65, 120)
        self._card_hover_right = QColor(80, 45, 100)
        self._card_selected = QColor(50, 200, 120)
        self._text_color = QColor(220, 225, 240)
        self._dim_text = QColor(100, 105, 130)
        self._accent_blue = QColor(80, 160, 255)
        self._accent_purple = QColor(160, 100, 255)

    # ─── Public API ─────────────────────────────────────────────

    def set_options(self, left: str, right: str, question: str,
                    selections: list[str], is_final: bool = False,
                    composed_response: str = ""):
        """Set the two answer options to display."""
        self._left_text = left
        self._right_text = right
        self._question = question
        self._selections = list(selections)
        self._is_final = is_final
        self._composed_response = composed_response
        self._is_loading = False
        self._dwell_progress = 0.0
        self._selected_side = None
        self._compute_layout()
        self._dwell_timer.start()
        self._pulse_timer.start()
        self.update()

    def set_loading(self, question: str, selections: list[str]):
        """Show a loading state while LLM generates options."""
        self._question = question
        self._selections = list(selections)
        self._is_loading = True
        self._left_text = ""
        self._right_text = ""
        self._is_final = False
        self._dwell_progress = 0.0
        self._dwell_timer.stop()
        self._compute_layout()
        self._pulse_timer.start()
        self.update()

    def set_final_response(self, response: str, question: str,
                           selections: list[str]):
        """Show the final composed response ready to speak."""
        self._composed_response = response
        self._question = question
        self._selections = list(selections)
        self._is_final = True
        self._is_loading = False
        self._left_text = ""
        self._right_text = ""
        self._dwell_timer.stop()
        self._compute_layout()
        self.update()

    def set_zone(self, zone: str):
        """Update the current gaze zone. Called by main.py from gaze tracker."""
        if zone != self._current_zone:
            self._current_zone = zone
            self._dwell_progress = 0.0

    def set_dwell_time(self, ms: int):
        """Set dwell time in milliseconds."""
        self._dwell_time_ms = max(500, min(3000, ms))

    def clear(self):
        """Reset the panel to empty state."""
        self._left_text = ""
        self._right_text = ""
        self._question = ""
        self._selections = []
        self._is_loading = False
        self._is_final = False
        self._composed_response = ""
        self._dwell_progress = 0.0
        self._dwell_timer.stop()
        self._pulse_timer.stop()
        self.update()

    # ─── Dwell logic ────────────────────────────────────────────

    def _update_dwell(self):
        """Advance dwell on the focused card."""
        if self._is_loading or self._is_final:
            return

        if self._current_zone == "CENTER":
            self._dwell_progress = max(0.0, self._dwell_progress - 0.02)
            return

        if self._current_zone in ("LEFT", "RIGHT"):
            increment = 16.0 / self._dwell_time_ms
            self._dwell_progress = min(1.0, self._dwell_progress + increment)

            if self._dwell_progress >= 1.0:
                self._trigger_selection()

    def _trigger_selection(self):
        """Trigger selection of the focused card."""
        if self._current_zone == "LEFT" and self._left_text:
            self._selected_side = "LEFT"
            self._flash_timer.start(400)
            self._dwell_progress = 0.0
            self._dwell_timer.stop()
            self.answer_selected.emit(self._left_text)
        elif self._current_zone == "RIGHT" and self._right_text:
            self._selected_side = "RIGHT"
            self._flash_timer.start(400)
            self._dwell_progress = 0.0
            self._dwell_timer.stop()
            self.answer_selected.emit(self._right_text)

    def _clear_flash(self):
        self._selected_side = None

    def _advance_pulse(self):
        self._pulse_phase += 0.06

    # ─── Layout ─────────────────────────────────────────────────

    def _compute_layout(self):
        """Compute all element rectangles."""
        w, h = self.width(), self.height()
        if w < 100 or h < 100:
            return

        pad = 16

        # Question area (top 15%)
        self._question_rect = QRectF(pad, pad, w - 2 * pad, h * 0.10)

        # Selection trail (below question, 8%)
        trail_y = self._question_rect.bottom() + 4
        self._trail_rect = QRectF(pad, trail_y, w - 2 * pad, h * 0.06)

        # Cards area (center 55%)
        cards_y = self._trail_rect.bottom() + pad
        cards_h = h * 0.52
        card_w = (w - 3 * pad - 80) / 2  # 80px for center "OR" divider

        self._left_card_rect = QRectF(pad, cards_y, card_w, cards_h)
        self._right_card_rect = QRectF(
            w - pad - card_w, cards_y, card_w, cards_h
        )
        self._vs_rect = QRectF(
            self._left_card_rect.right(),
            cards_y + cards_h * 0.3,
            w - 2 * pad - 2 * card_w,
            cards_h * 0.4,
        )

        # Action buttons (bottom 15%)
        action_y = cards_y + cards_h + pad
        action_h = 44
        actions = [
            ("🔄 More Options", "more"),
            ("⌨  Type Instead", "type"),
            ("↩  Back", "back"),
            ("🔊 Speak Now", "speak"),
        ]
        btn_w = (w - pad * (len(actions) + 1)) / len(actions)
        self._action_rects = []
        for i, (label, action_id) in enumerate(actions):
            x = pad + i * (btn_w + pad)
            rect = QRectF(x, action_y, btn_w, action_h)
            self._action_rects.append((label, action_id, rect))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_layout()

    # ─── Painting ───────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, self._bg)

        if not self._question:
            self._draw_empty_state(painter, w, h)
            painter.end()
            return

        # Draw question
        self._draw_question(painter)

        # Draw selection trail
        self._draw_trail(painter)

        if self._is_final:
            self._draw_final_response(painter, w, h)
        elif self._is_loading:
            self._draw_loading(painter, w, h)
        else:
            # Draw cards
            self._draw_card(painter, self._left_card_rect, self._left_text,
                            "LEFT", "👈", self._card_left_color,
                            self._card_hover_left, self._accent_blue)
            self._draw_card(painter, self._right_card_rect, self._right_text,
                            "RIGHT", "👉", self._card_right_color,
                            self._card_hover_right, self._accent_purple)

            # Draw "OR" divider
            self._draw_vs_divider(painter)

        # Draw action buttons
        self._draw_actions(painter)

        painter.end()

    def _draw_empty_state(self, painter, w, h):
        """Draw placeholder when no conversation is active."""
        painter.setPen(QColor(70, 75, 100))
        painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Light))
        painter.drawText(
            QRectF(0, h * 0.3, w, 40),
            Qt.AlignmentFlag.AlignCenter,
            "Waiting for caretaker to ask a question...",
        )
        painter.setPen(QColor(50, 55, 75))
        painter.setFont(QFont("Segoe UI", 13))
        painter.drawText(
            QRectF(0, h * 0.3 + 50, w, 30),
            Qt.AlignmentFlag.AlignCenter,
            "The caretaker can type or speak a question below",
        )

    def _draw_question(self, painter):
        """Draw the caretaker's question."""
        rect = self._question_rect
        # Background pill
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        painter.fillPath(path, QColor(20, 25, 42))
        painter.setPen(QPen(QColor(50, 55, 80), 1))
        painter.drawPath(path)

        # Icon + text
        painter.setPen(QColor(140, 150, 200))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(
            QRectF(rect.x() + 16, rect.y(), 100, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            "🗣 Caretaker:",
        )

        painter.setPen(self._text_color)
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        text_rect = QRectF(
            rect.x() + 130, rect.y(), rect.width() - 146, rect.height()
        )
        metrics = painter.fontMetrics()
        elided = metrics.elidedText(
            self._question, Qt.TextElideMode.ElideRight, int(text_rect.width())
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, elided)

    def _draw_trail(self, painter):
        """Draw breadcrumbs of prior selections."""
        if not self._selections:
            return

        rect = self._trail_rect
        painter.setPen(QColor(90, 95, 120))
        painter.setFont(QFont("Segoe UI", 10))

        trail_str = "  →  ".join(self._selections)
        painter.drawText(
            QRectF(rect.x() + 12, rect.y(), rect.width() - 24, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            f"Choices: {trail_str}",
        )

    def _draw_card(self, painter, rect, text, side, arrow, base_color,
                   hover_color, accent):
        """Draw one answer card."""
        is_hovered = self._current_zone == side and not self._is_loading
        is_selected = self._selected_side == side

        # Card background
        if is_selected:
            bg = self._card_selected
        elif is_hovered:
            bg = hover_color
        else:
            bg = base_color

        path = QPainterPath()
        path.addRoundedRect(rect, 20, 20)

        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, bg.lighter(120))
        gradient.setColorAt(1, bg)
        painter.fillPath(path, gradient)

        # Glow border when hovered
        if is_hovered:
            glow_pen = QPen(accent, 3)
            painter.setPen(glow_pen)
            painter.drawPath(path)

            # Outer glow
            for i in range(3):
                glow_color = QColor(accent.red(), accent.green(),
                                    accent.blue(), 40 - i * 12)
                painter.setPen(QPen(glow_color, 1))
                glow_path = QPainterPath()
                glow_rect = rect.adjusted(-i * 2, -i * 2, i * 2, i * 2)
                glow_path.addRoundedRect(glow_rect, 22 + i, 22 + i)
                painter.drawPath(glow_path)
        else:
            painter.setPen(QPen(QColor(50, 55, 75), 1))
            painter.drawPath(path)

        # Direction arrow at top
        painter.setPen(self._dim_text if not is_hovered else self._text_color)
        painter.setFont(QFont("Segoe UI", 22))
        arrow_rect = QRectF(rect.x(), rect.y() + 16, rect.width(), 40)
        painter.drawText(arrow_rect, Qt.AlignmentFlag.AlignCenter, arrow)

        # Direction label
        label = "LOOK LEFT" if side == "LEFT" else "LOOK RIGHT"
        painter.setPen(
            accent.lighter(130) if is_hovered else QColor(80, 85, 110)
        )
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        label_rect = QRectF(rect.x(), rect.y() + 52, rect.width(), 20)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

        # Main answer text
        text_color = (
            QColor(10, 10, 10) if is_selected
            else QColor(255, 255, 255) if is_hovered
            else self._text_color
        )
        painter.setPen(text_color)
        font_size = max(14, min(24, int(rect.width() / max(len(text), 1) * 1.2)))
        font_size = min(font_size, 24)
        painter.setFont(
            QFont("Segoe UI", font_size, QFont.Weight.Bold)
        )
        text_area = QRectF(
            rect.x() + 20, rect.y() + rect.height() * 0.35,
            rect.width() - 40, rect.height() * 0.40,
        )
        painter.drawText(
            text_area,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

        # Dwell progress ring
        if is_hovered and self._dwell_progress > 0:
            self._draw_dwell_ring(painter, rect, self._dwell_progress, accent)

    def _draw_dwell_ring(self, painter, rect, progress, color):
        """Draw circular progress around the card."""
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 16

        pen = QPen(color, 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        span = int(progress * 360 * 16)
        arc_rect = QRectF(
            center.x() - radius, center.y() - radius,
            radius * 2, radius * 2,
        )
        painter.drawArc(arc_rect, 90 * 16, -span)

        # Glow tip
        if progress > 0.05:
            angle_rad = math.radians(90 - progress * 360)
            tip_x = center.x() + radius * math.cos(angle_rad)
            tip_y = center.y() - radius * math.sin(angle_rad)
            glow = QRadialGradient(tip_x, tip_y, 10)
            glow.setColorAt(0, QColor(color.red(), color.green(),
                                      color.blue(), 200))
            glow.setColorAt(1, QColor(color.red(), color.green(),
                                      color.blue(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(tip_x, tip_y), 10, 10)

    def _draw_vs_divider(self, painter):
        """Draw the OR divider between cards."""
        rect = self._vs_rect
        cx = rect.center().x()

        # Vertical line
        painter.setPen(QPen(QColor(40, 45, 65), 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            int(cx), int(self._left_card_rect.top() + 20),
            int(cx), int(self._left_card_rect.bottom() - 20),
        )

        # "OR" badge
        badge_rect = QRectF(cx - 22, rect.center().y() - 16, 44, 32)
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 16, 16)
        painter.fillPath(badge_path, QColor(30, 30, 50))
        painter.setPen(QPen(QColor(60, 65, 90), 1))
        painter.drawPath(badge_path)

        painter.setPen(QColor(120, 130, 170))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "OR")

    def _draw_loading(self, painter, w, h):
        """Draw loading animation while LLM generates options."""
        center_y = (self._left_card_rect.top() + self._left_card_rect.bottom()) / 2
        painter.setPen(QColor(120, 130, 170))
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Light))
        painter.drawText(
            QRectF(0, center_y - 30, w, 30),
            Qt.AlignmentFlag.AlignCenter,
            "Thinking...",
        )

        # Animated dots
        pulse = 0.5 + 0.5 * math.sin(self._pulse_phase)
        dots_y = center_y + 10
        for i in range(3):
            dot_pulse = 0.5 + 0.5 * math.sin(self._pulse_phase + i * 0.8)
            alpha = int(80 + 120 * dot_pulse)
            dot_color = QColor(80, 160, 255, alpha)
            dot_r = 4 + 3 * dot_pulse
            painter.setBrush(dot_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                QPointF(w / 2 - 30 + i * 30, dots_y),
                dot_r, dot_r,
            )

    def _draw_final_response(self, painter, w, h):
        """Draw the final composed response ready to speak."""
        card_rect = QRectF(
            40,
            self._trail_rect.bottom() + 16,
            w - 80,
            h * 0.45,
        )

        # Card background
        path = QPainterPath()
        path.addRoundedRect(card_rect, 20, 20)
        gradient = QLinearGradient(card_rect.topLeft(), card_rect.bottomRight())
        gradient.setColorAt(0, QColor(25, 55, 45))
        gradient.setColorAt(1, QColor(15, 40, 30))
        painter.fillPath(path, gradient)
        painter.setPen(QPen(QColor(80, 200, 120, 150), 2))
        painter.drawPath(path)

        # "Ready to speak" label
        painter.setPen(QColor(80, 200, 120))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(
            QRectF(card_rect.x(), card_rect.y() + 16, card_rect.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            "✓ Response Ready",
        )

        # Response text
        painter.setPen(QColor(240, 245, 255))
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.DemiBold))
        text_rect = QRectF(
            card_rect.x() + 30, card_rect.y() + 50,
            card_rect.width() - 60, card_rect.height() - 80,
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            f'"{self._composed_response}"',
        )

        # Hint
        painter.setPen(QColor(80, 200, 120, 150))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(
            QRectF(card_rect.x(), card_rect.bottom() - 30,
                   card_rect.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            "Click 🔊 Speak Now to say this aloud",
        )

    def _draw_actions(self, painter):
        """Draw the bottom action buttons."""
        for label, action_id, rect in self._action_rects:
            # Background
            path = QPainterPath()
            path.addRoundedRect(rect, 10, 10)

            if action_id == "speak":
                bg = QColor(25, 55, 40)
                border = QColor(60, 150, 90, 120)
            elif action_id == "type":
                bg = QColor(30, 35, 55)
                border = QColor(60, 70, 110, 120)
            else:
                bg = QColor(25, 25, 40)
                border = QColor(50, 55, 75, 100)

            painter.fillPath(path, bg)
            painter.setPen(QPen(border, 1))
            painter.drawPath(path)

            # Label
            painter.setPen(QColor(180, 185, 210))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # ─── Mouse fallback (caregiver mode) ────────────────────────

    def mousePressEvent(self, event):
        """Mouse click fallback for caregiver."""
        pos = event.position()

        # Check answer cards
        if not self._is_loading and not self._is_final:
            if self._left_card_rect.contains(QPointF(pos.x(), pos.y())):
                if self._left_text:
                    self._selected_side = "LEFT"
                    self._flash_timer.start(400)
                    self.answer_selected.emit(self._left_text)
                    return
            if self._right_card_rect.contains(QPointF(pos.x(), pos.y())):
                if self._right_text:
                    self._selected_side = "RIGHT"
                    self._flash_timer.start(400)
                    self.answer_selected.emit(self._right_text)
                    return

        # Check action buttons
        for label, action_id, rect in self._action_rects:
            if rect.contains(QPointF(pos.x(), pos.y())):
                if action_id == "more":
                    self.more_options_requested.emit()
                elif action_id == "type":
                    self.type_instead_requested.emit()
                elif action_id == "back":
                    self.back_requested.emit()
                elif action_id == "speak":
                    self.speak_now_requested.emit()
                return
