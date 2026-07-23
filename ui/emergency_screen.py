"""
ui/emergency_screen.py — Full-Screen Emergency Overlay

Displayed when the patient triggers the rapid-blink SOS signal.
Covers the entire application window with a bright-red alert screen that:
  - Shows "EMERGENCY DETECTED" in large white text
  - Animates a pulsing red/dark border effect
  - Displays live SMS delivery status
  - Provides a large STOP button to dismiss the screen & silence the siren
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen


class EmergencyScreen(QWidget):
    """
    Full-window red overlay shown during a blink-triggered emergency.

    Signals
    -------
    stop_requested : emitted when the patient/caregiver presses STOP.
    """

    stop_requested = pyqtSignal()

    # SMS status display strings
    _STATUS_TEXT = {
        "sending": "Sending emergency SMS to caregiver...",
        "sent":    "Alert sent to caregiver successfully!",
        "failed":  "SMS could not be sent (check Twilio config)",
    }

    # Colors for each SMS status
    _STATUS_COLOR = {
        "sending": "#ffe08a",   # warm amber
        "sent":    "#a0ffb0",   # soft green
        "failed":  "#ff9090",   # soft red
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        # Make this widget cover its parent entirely and stay on top
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("EmergencyScreen")

        # Hidden by default
        self.hide()

        # Pulse animation state (for the border flash effect)
        self._pulse_bright = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)          # toggle every 500 ms
        self._pulse_timer.timeout.connect(self._on_pulse)

        self._build_ui()

    # ─────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the overlay layout."""

        # Root red background
        self.setStyleSheet("""
            QWidget#EmergencyScreen {
                background-color: #cc0000;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Pulsing border frame ──────────────────────────────────
        self._border_frame = QFrame()
        self._border_frame.setObjectName("BorderFrame")
        self._border_frame.setStyleSheet("""
            QFrame#BorderFrame {
                border: 14px solid #ff3333;
                background-color: transparent;
            }
        """)

        inner = QVBoxLayout(self._border_frame)
        inner.setContentsMargins(40, 40, 40, 40)
        inner.setSpacing(24)

        # ── Alert icon / Header ──────────────────────────────────
        icon_label = QLabel("ALERT")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setFont(QFont("Roboto Mono", 36, QFont.Weight.Bold))
        icon_label.setStyleSheet("color: white; background: transparent;")
        inner.addWidget(icon_label)

        # ── Main heading ──────────────────────────────────────────
        heading = QLabel("EMERGENCY DETECTED")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading_font = QFont("Roboto Mono", 52, QFont.Weight.Black)
        heading.setFont(heading_font)
        heading.setStyleSheet("""
            color: white;
            background: transparent;
            letter-spacing: 4px;
        """)
        inner.addWidget(heading)

        # ── Sub-heading ───────────────────────────────────────────
        sub = QLabel("Rapid blink signal detected — emergency protocol activated")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setFont(QFont("Roboto Mono", 18))
        sub.setStyleSheet("color: rgba(255,255,255,0.85); background: transparent;")
        inner.addWidget(sub)

        inner.addSpacing(20)

        # ── SMS status label ──────────────────────────────────────
        self._status_label = QLabel("Sending emergency SMS to caregiver...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFont(QFont("Roboto Mono", 16))
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("""
            color: #ffe08a;
            background: rgba(0, 0, 0, 0.30);
            border-radius: 14px;
            padding: 14px 30px;
        """)
        inner.addWidget(self._status_label)

        inner.addSpacing(30)

        # ── STOP button ───────────────────────────────────────────
        stop_btn = QPushButton("STOP")
        stop_btn.setObjectName("StopButton")
        stop_btn.setFixedHeight(100)
        stop_btn.setMinimumWidth(340)
        stop_btn.setFont(QFont("Roboto Mono", 30, QFont.Weight.Bold))
        stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        stop_btn.setStyleSheet("""
            QPushButton#StopButton {
                background-color: white;
                color: #cc0000;
                border: none;
                border-radius: 50px;
                padding: 0 60px;
                letter-spacing: 2px;
            }
            QPushButton#StopButton:hover {
                background-color: #ffe0e0;
            }
            QPushButton#StopButton:pressed {
                background-color: #ffb0b0;
            }
        """)
        stop_btn.clicked.connect(self._on_stop_clicked)

        # Centre the button horizontally
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(stop_btn)
        btn_row.addStretch()
        inner.addLayout(btn_row)

        inner.addStretch()

        root.addWidget(self._border_frame)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def activate(self):
        """Show the emergency screen and start the pulse animation."""
        self._status_label.setText(self._STATUS_TEXT["sending"])
        self._status_label.setStyleSheet(f"""
            color: {self._STATUS_COLOR['sending']};
            background: rgba(0, 0, 0, 0.30);
            border-radius: 14px;
            padding: 14px 30px;
        """)
        self._pulse_bright = False
        self._pulse_timer.start()
        self.raise_()
        self.show()

    def deactivate(self):
        """Hide the emergency screen and stop the pulse animation."""
        self._pulse_timer.stop()
        self.hide()

    def update_sms_status(self, status: str):
        """
        Update the SMS delivery status label.

        Parameters
        ----------
        status : str
            One of 'sending', 'sent', or 'failed'.
        """
        text = self._STATUS_TEXT.get(status, "")
        color = self._STATUS_COLOR.get(status, "white")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"""
            color: {color};
            background: rgba(0, 0, 0, 0.30);
            border-radius: 14px;
            padding: 14px 30px;
        """)

    # ─────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────

    def _on_stop_clicked(self):
        """User pressed STOP — deactivate and emit signal."""
        self.deactivate()
        self.stop_requested.emit()

    def _on_pulse(self):
        """Alternate the border colour for a strobing red effect."""
        self._pulse_bright = not self._pulse_bright
        if self._pulse_bright:
            border_color = "#ff6666"
            bg_color = "#dd0000"
        else:
            border_color = "#990000"
            bg_color = "#cc0000"

        self.setStyleSheet(f"""
            QWidget#EmergencyScreen {{
                background-color: {bg_color};
            }}
        """)
        self._border_frame.setStyleSheet(f"""
            QFrame#BorderFrame {{
                border: 14px solid {border_color};
                background-color: transparent;
            }}
        """)
