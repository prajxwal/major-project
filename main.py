"""
GazeSpeak — Assistive Eye-Gaze Typing for ALS Patients

Main entry point. Launches the PyQt6 application with:
- Webcam-based eye gaze tracking (MediaPipe Face Mesh)
- 2-point horizontal calibration (look left, look right)
- On-screen QWERTY keyboard with dwell-time selection
- Word prediction with frequency-ranked dictionary
- Context-aware abbreviation expansion
- Quick phrases for common needs
- Text-to-speech output
- Rapid-blink emergency alert (Twilio SMS)
- Caregiver mode (mouse/keyboard fallback + STT context)

Usage:
    python main.py
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QStackedWidget, QLabel, QFrame)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QFont, QColor, QScreen, QImage, QPixmap

from gaze.tracker import GazeTracker
from gaze.calibration import CalibrationScreen
from ui.keyboard_widget import KeyboardWidget
from ui.gaze_cursor import GazeCursor
from ui.sentence_bar import SentenceBar
from ui.prediction_bar import PredictionBar
from ui.quick_phrases import QuickPhrasesPanel
from ui.settings_panel import SettingsPanel
from ui.context_bar import ContextBar
from prediction.predictor import WordPredictor
from prediction.abbreviation_expander import AbbreviationExpander
from alerts.blink_alert import BlinkAlertManager


class WebcamWidget(QLabel):
    """Small webcam feed preview widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 150)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #2a2a4a;
                border-radius: 10px;
                background-color: #0a0a14;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("📷 Webcam")
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(self.styleSheet() + "color: #666;")
    
    def update_frame(self, frame):
        """Update with a new webcam frame (numpy array BGR)."""
        import cv2
        h, w, ch = frame.shape
        # Resize to fit widget
        frame_resized = cv2.resize(frame, (self.width(), self.height()))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       rgb.strides[0], QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))


class GazeSpeakApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GazeSpeak — Eye Gaze Communication")
        self.setMinimumSize(1200, 800)
        
        # Get screen geometry for gaze mapping
        screen = QApplication.primaryScreen()
        self._screen_geo = screen.geometry()
        
        # Core components
        self._tracker = GazeTracker()
        self._predictor = WordPredictor()
        self._abbreviation_expander = AbbreviationExpander()
        self._calibration = CalibrationScreen()
        self._blink_alert = BlinkAlertManager()
        
        # Abbreviation mode tracking
        self._abbrev_mode = False
        
        # ─── Step-based navigation state ───
        self._nav_area = "keyboard"   # "keyboard" or "predictions"
        self._key_row = 0
        self._key_col = 0
        self._pred_index = 0
        self._current_zone = "CENTER"  # "LEFT", "CENTER", "RIGHT"
        self._step_count = 0
        
        # Gaze zone thresholds (calibrated 0.0–1.0)
        self._zone_left = 0.30
        self._zone_right = 0.70
        
        # Step timing
        self._step_timer = QTimer(self)
        self._step_timer.timeout.connect(self._do_gaze_step)
        self._step_initial_delay = 450   # ms before first repeat
        self._step_repeat_delay = 300    # ms between subsequent steps
        
        # Build UI
        self._setup_ui()
        self._connect_signals()
        
        # Start gaze tracking (calibration will be triggered after window shows)
        self._tracker.start()
        
        # Apply dark theme
        self._apply_theme()
        
        # Set up alert notification UI callback
        self._blink_alert.set_alert_callback(self._show_alert_notification)
        
        # Initialize keyboard highlight
        self._keyboard.set_highlight(0, 0)
    
    def _setup_ui(self):
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Top area: Webcam + Sentence Bar ---
        top_area = QWidget()
        top_area.setStyleSheet("background-color: #0e0e1a;")
        top_layout = QHBoxLayout(top_area)
        top_layout.setContentsMargins(12, 8, 12, 8)
        
        # Webcam preview (small, left corner)
        self._webcam_widget = WebcamWidget()
        top_layout.addWidget(self._webcam_widget)
        
        # Sentence bar (takes remaining space)
        self._sentence_bar = SentenceBar()
        top_layout.addWidget(self._sentence_bar, stretch=1)
        
        # Tracking status indicator
        self._status_indicator = QLabel("● Tracking")
        self._status_indicator.setFont(QFont("Segoe UI", 11))
        self._status_indicator.setStyleSheet("color: #50c878; padding: 8px;")
        self._status_indicator.setFixedWidth(120)
        self._status_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self._status_indicator)
        
        main_layout.addWidget(top_area)
        
        # --- Prediction bar ---
        self._prediction_bar = PredictionBar()
        main_layout.addWidget(self._prediction_bar)
        
        # --- Stacked area: Keyboard / Quick Phrases / Settings ---
        self._stacked = QStackedWidget()
        
        self._keyboard = KeyboardWidget()
        self._quick_phrases = QuickPhrasesPanel()
        self._settings = SettingsPanel()
        
        self._stacked.addWidget(self._keyboard)       # index 0
        self._stacked.addWidget(self._quick_phrases)   # index 1
        self._stacked.addWidget(self._settings)        # index 2
        
        main_layout.addWidget(self._stacked, stretch=1)
        
        # --- Caretaker context bar (bottom) ---
        self._context_bar = ContextBar()
        main_layout.addWidget(self._context_bar)
        
        # --- Gaze cursor overlay ---
        self._gaze_cursor = GazeCursor(central)
        self._gaze_cursor.setGeometry(0, 0, self.width(), self.height())
        self._gaze_cursor.raise_()  # ensure it's on top
    
    def _connect_signals(self):
        """Wire up all signals between components."""
        
        # Gaze tracker → UI updates
        self._tracker.gaze_updated.connect(self._on_gaze_updated)
        self._tracker.frame_ready.connect(self._webcam_widget.update_frame)
        self._tracker.tracking_lost.connect(self._on_tracking_lost)
        
        # Blink detection → alert system
        self._tracker.blink_detected.connect(self._blink_alert.register_blink)
        
        # Calibration
        self._calibration.calibration_complete.connect(self._on_calibration_complete)
        
        # Keyboard → sentence bar
        self._keyboard.key_pressed.connect(self._on_key_pressed)
        self._keyboard.special_key_pressed.connect(self._on_special_key)
        
        # Prediction bar → sentence bar
        self._prediction_bar.word_selected.connect(self._on_word_selected)
        
        # Sentence bar → prediction updates
        self._sentence_bar.text_changed.connect(self._on_text_changed)
        
        # Quick phrases
        self._quick_phrases.phrase_selected.connect(self._on_phrase_selected)
        self._quick_phrases.back_requested.connect(lambda: self._stacked.setCurrentIndex(0))
        
        # Settings
        self._settings.dwell_time_changed.connect(self._on_dwell_time_changed)
        self._settings.smoothing_changed.connect(self._tracker.set_smoothing)
        self._settings.tts_rate_changed.connect(self._sentence_bar.set_tts_rate)
        self._settings.recalibrate_requested.connect(self._start_calibration)
        self._settings.close_requested.connect(lambda: self._stacked.setCurrentIndex(0))
        
        # Caretaker context bar
        self._context_bar.context_submitted.connect(self._on_caretaker_context)
    
    def _on_gaze_updated(self, gaze_x, gaze_y, confidence):
        """Handle gaze updates — zone-based step navigation."""
        # Determine horizontal zone
        if gaze_x < self._zone_left:
            new_zone = "LEFT"
        elif gaze_x > self._zone_right:
            new_zone = "RIGHT"
        else:
            new_zone = "CENTER"
        
        if new_zone != self._current_zone:
            self._current_zone = new_zone
            self._step_count = 0
            
            if new_zone in ("LEFT", "RIGHT"):
                # Entering a navigation zone — step once immediately
                self._keyboard.set_navigating(True)
                self._prediction_bar.set_navigating(True)
                self._do_gaze_step()
                # Start repeat timer
                self._step_timer.start(self._step_initial_delay)
            else:
                # Returned to CENTER — stop stepping, enable dwell
                self._step_timer.stop()
                self._keyboard.set_navigating(False)
                self._prediction_bar.set_navigating(False)
        
        # Snap the gaze cursor to the highlighted item's center
        self._snap_cursor_to_highlight(confidence)
        
        # Feed raw data to calibration screen if active
        if self._calibration.isVisible():
            self._calibration.receive_gaze_sample(gaze_x, gaze_y, confidence)
        
        # Update tracking status
        if confidence > 0.5:
            self._status_indicator.setText("● Tracking")
            self._status_indicator.setStyleSheet("color: #50c878; padding: 8px;")
        elif confidence > 0.3:
            self._status_indicator.setText("● Weak")
            self._status_indicator.setStyleSheet("color: #f0c040; padding: 8px;")
        else:
            self._status_indicator.setText("● Low")
            self._status_indicator.setStyleSheet("color: #f06040; padding: 8px;")
    
    def _do_gaze_step(self):
        """Execute one navigation step in the current direction."""
        self._step_count += 1
        direction = 1 if self._current_zone == "RIGHT" else -1
        
        if self._nav_area == "predictions":
            self._step_prediction(direction)
        else:
            self._step_keyboard(direction)
        
        # After first step, switch to faster repeat rate
        if self._step_count == 1:
            self._step_timer.start(self._step_repeat_delay)
    
    def _step_keyboard(self, direction):
        """Step the keyboard highlight left (-1) or right (+1) with row wrapping."""
        grid = self._keyboard.get_grid()
        row, col = self._key_row, self._key_col
        
        col += direction
        
        if col >= len(grid[row]):
            # Past end of row → next row
            row += 1
            col = 0
            if row >= len(grid):
                # Past last keyboard row → wrap to predictions (if any)
                if self._prediction_bar.get_count() > 0:
                    self._nav_area = "predictions"
                    self._pred_index = 0
                    self._keyboard.set_highlight(-1, -1)
                    self._prediction_bar.set_highlight(0)
                    return
                else:
                    row = 0  # wrap to first keyboard row
        
        elif col < 0:
            # Past start of row → previous row
            row -= 1
            if row < 0:
                # Past first keyboard row → wrap to predictions (if any)
                if self._prediction_bar.get_count() > 0:
                    self._nav_area = "predictions"
                    self._pred_index = self._prediction_bar.get_count() - 1
                    self._keyboard.set_highlight(-1, -1)
                    self._prediction_bar.set_highlight(self._pred_index)
                    return
                else:
                    row = len(grid) - 1  # wrap to last keyboard row
            col = len(grid[row]) - 1
        
        self._key_row = row
        self._key_col = col
        self._keyboard.set_highlight(row, col)
        self._prediction_bar.set_highlight(-1)
    
    def _step_prediction(self, direction):
        """Step the prediction bar highlight left (-1) or right (+1)."""
        count = self._prediction_bar.get_count()
        if count == 0:
            # No predictions — jump to keyboard
            self._nav_area = "keyboard"
            self._keyboard.set_highlight(self._key_row, self._key_col)
            return
        
        idx = self._pred_index + direction
        
        if idx >= count:
            # Past last prediction → jump to keyboard row 0, col 0
            self._nav_area = "keyboard"
            self._key_row = 0
            self._key_col = 0
            self._keyboard.set_highlight(0, 0)
            self._prediction_bar.set_highlight(-1)
            return
        
        if idx < 0:
            # Past first prediction → jump to keyboard last row, last col
            self._nav_area = "keyboard"
            grid = self._keyboard.get_grid()
            self._key_row = len(grid) - 1
            self._key_col = len(grid[self._key_row]) - 1
            self._keyboard.set_highlight(self._key_row, self._key_col)
            self._prediction_bar.set_highlight(-1)
            return
        
        self._pred_index = idx
        self._prediction_bar.set_highlight(idx)
        self._keyboard.set_highlight(-1, -1)
    
    def _snap_cursor_to_highlight(self, confidence):
        """Snap the visual gaze cursor to the center of the highlighted item."""
        rect = None
        widget = None
        
        if self._nav_area == "predictions":
            rect = self._prediction_bar.get_highlighted_rect()
            widget = self._prediction_bar
        else:
            rect = self._keyboard.get_highlighted_rect()
            widget = self._keyboard
        
        if rect is not None and widget is not None:
            # Convert rect center to main window coordinates
            center = rect.center()
            global_pos = widget.mapTo(self.centralWidget(), QPoint(int(center.x()), int(center.y())))
            self._gaze_cursor.update_position(
                float(global_pos.x()), float(global_pos.y()), confidence
            )
        else:
            # Fallback — keep cursor at center
            self._gaze_cursor.update_position(
                self.width() / 2, self.height() / 2, confidence * 0.3
            )
    
    def _on_tracking_lost(self):
        """Handle loss of face/eye tracking."""
        self._status_indicator.setText("○ No face")
        self._status_indicator.setStyleSheet("color: #ff4040; padding: 8px;")
    
    def _on_key_pressed(self, key):
        """Handle letter key selection."""
        self._sentence_bar.add_character(key)
    
    def _on_special_key(self, action):
        """Handle special key actions."""
        if action == "BACKSPACE":
            self._sentence_bar.backspace()
        elif action == "SPACE":
            self._sentence_bar.add_space()
        elif action == "SPEAK":
            self._sentence_bar.speak()
        elif action == "CLEAR":
            self._sentence_bar.clear()
        elif action == "PHRASES":
            self._stacked.setCurrentIndex(1)  # show quick phrases
        elif action == "SETTINGS":
            self._stacked.setCurrentIndex(2)  # show settings
    
    def _on_word_selected(self, word):
        """Handle word/expansion prediction selection."""
        if self._abbrev_mode:
            # In abbreviation mode: replace ALL text with the expanded sentence
            self._sentence_bar.set_text(word)
            self._abbreviation_expander.add_patient_message(word)
            self._abbrev_mode = False
        else:
            self._sentence_bar.add_word(word)
    
    def _on_text_changed(self, text):
        """Update word predictions when text changes — with abbreviation detection."""
        current_word = self._sentence_bar.get_current_word()
        
        if current_word:
            # Check if this looks like an abbreviation
            if self._abbreviation_expander.is_likely_abbreviation(current_word):
                self._abbrev_mode = True
                # Show local predictions immediately as fallback
                local_predictions = self._predictor.predict(current_word)
                self._prediction_bar.set_predictions(local_predictions)
                
                # Fire async abbreviation expansion
                self._abbreviation_expander.expand(
                    current_word,
                    callback=self._on_abbreviation_expanded
                )
            else:
                self._abbrev_mode = False
                # Normal word prediction
                local_predictions = self._predictor.predict(current_word)
                self._prediction_bar.set_predictions(local_predictions)
                
                # Async LLM predictions (replace local when ready)
                self._predictor.predict_with_llm(
                    text, current_word,
                    callback=self._on_llm_predictions
                )
        elif text.endswith(" "):
            self._abbrev_mode = False
            # Just finished a word — predict next word
            self._predictor.predict_next_word(
                text,
                callback=self._on_llm_predictions
            )
        else:
            self._abbrev_mode = False
            self._prediction_bar.set_predictions([])
    
    def _on_llm_predictions(self, words):
        """Handle async LLM prediction results (called from background thread)."""
        if not self._abbrev_mode:  # don't overwrite abbreviation expansions
            QTimer.singleShot(0, lambda: self._prediction_bar.set_predictions(words))
    
    def _on_abbreviation_expanded(self, expansions):
        """Handle async abbreviation expansion results."""
        QTimer.singleShot(0, lambda: self._prediction_bar.set_predictions(expansions))
    
    def _on_caretaker_context(self, question):
        """Handle caretaker question submission — sets context for abbreviation expander."""
        self._abbreviation_expander.set_caretaker_context(question)
        print(f"[GazeSpeak] Caretaker context set: '{question}'")
    
    def _on_phrase_selected(self, phrase):
        """Handle quick phrase selection — speak immediately."""
        self._sentence_bar.speak_text(phrase)
    
    def _on_dwell_time_changed(self, ms):
        """Update dwell time across all components."""
        self._keyboard.set_dwell_time(ms)
        self._prediction_bar.set_dwell_time(ms)
        self._quick_phrases.set_dwell_time(ms)
    
    def _start_calibration(self):
        """Launch the calibration screen."""
        self._calibration.start_calibration()
    
    def _on_calibration_complete(self, cal_data):
        """Apply calibration and compute zone thresholds from center_x."""
        self._tracker.set_calibration(cal_data)
        
        # Use the calibrated center point to define zone boundaries
        # center_x is in calibrated space (0.0-1.0 after mapping)
        if isinstance(cal_data, dict) and 'center_x' in cal_data:
            left_x = cal_data['left_x']
            center_x = cal_data['center_x']
            right_x = cal_data['right_x']
            span = right_x - left_x
            
            if abs(span) > 0.01:
                # Map center to 0.0-1.0 range
                center_norm = (center_x - left_x) / span
                center_norm = max(0.2, min(0.8, center_norm))
                
                # Build dead zone around center (±15% of range)
                margin = 0.15
                self._zone_left = center_norm - margin
                self._zone_right = center_norm + margin
                
                print(f"[GazeSpeak] Zone thresholds: "
                      f"LEFT < {self._zone_left:.2f} | "
                      f"CENTER {self._zone_left:.2f}-{self._zone_right:.2f} | "
                      f"RIGHT > {self._zone_right:.2f}")
    
    def _apply_theme(self):
        """Apply the dark theme to the entire application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a14;
            }
        """)
    
    def resizeEvent(self, event):
        """Resize the gaze cursor overlay to match window."""
        super().resizeEvent(event)
        if hasattr(self, '_gaze_cursor'):
            self._gaze_cursor.setGeometry(0, 0, self.width(), self.height())
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts (caregiver mode)."""
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        elif event.key() == Qt.Key.Key_F5:
            self._start_calibration()
    
    def _show_alert_notification(self, message):
        """Show emergency alert notification on screen."""
        # Use QTimer to safely update UI from background thread
        QTimer.singleShot(0, lambda: self._display_alert_banner(message))
    
    def _display_alert_banner(self, message):
        """Display a red alert banner at the top of the screen."""
        alert_banner = QLabel(message)
        alert_banner.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        alert_banner.setStyleSheet("""
            QLabel {
                background-color: #ff2040;
                color: white;
                padding: 16px;
                border-radius: 0;
                font-size: 18px;
            }
        """)
        alert_banner.setFixedHeight(60)
        
        # Insert at top of main layout
        main_layout = self.centralWidget().layout()
        main_layout.insertWidget(0, alert_banner)
        
        # Auto-remove after 10 seconds
        QTimer.singleShot(10000, lambda: self._remove_alert_banner(alert_banner))
    
    def _remove_alert_banner(self, banner):
        """Remove the alert banner from the layout."""
        try:
            banner.setParent(None)
            banner.deleteLater()
        except RuntimeError:
            pass  # widget already deleted
    
    def closeEvent(self, event):
        """Clean up on close."""
        self._blink_alert.stop_alarm()
        self._tracker.stop()
        self._tracker.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    app.setFont(QFont("Segoe UI", 12))
    
    # Global dark palette
    app.setStyle("Fusion")
    
    window = GazeSpeakApp()
    window.show()
    
    # Always run calibration on startup
    QTimer.singleShot(1000, window._start_calibration)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
