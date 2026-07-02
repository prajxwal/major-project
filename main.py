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
from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal
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
from ui.conversation_panel import ConversationPanel
from prediction.predictor import WordPredictor
from prediction.abbreviation_expander import AbbreviationExpander
from prediction.conversation_engine import ConversationEngine
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

    # Thread-safe signal: delivers LLM conversation results from the worker
    # thread to the GUI thread without relying on QTimer.singleShot.
    _conversation_ready = pyqtSignal(object)
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
        self._conversation_engine = ConversationEngine()
        self._calibration = CalibrationScreen()
        self._blink_alert = BlinkAlertManager()
        
        # Track whether this is the initial startup (calibration before app shows)
        self._initial_startup = True
        
        # Conversation mode tracking
        self._conversation_mode = False
        
        # Abbreviation mode tracking
        self._abbrev_mode = False
        
        # ─── Gesture-based navigation state ───
        self._nav_area = "keyboard"      # "keyboard" or "predictions"
        self._key_row = 0
        self._key_col = 0
        self._pred_index = 0
        self._current_zone = "CENTER"    # committed zone: "LEFT", "CENTER", "RIGHT"
        self._pending_zone = "CENTER"    # zone candidate accumulating hysteresis frames
        self._zone_frame_count = 0       # consecutive frames matching _pending_zone
        self._HYSTERESIS_FRAMES = 4      # frames required before committing a zone change
        self._user_has_gestured = False  # True after first deliberate LEFT/RIGHT gesture

        # Gaze zone thresholds (calibrated 0.0–1.0)
        self._zone_left = 0.30
        self._zone_right = 0.70
        
        # Build UI
        self._setup_ui()
        self._connect_signals()
        
        # Do NOT start tracker here — it starts just before calibration
        
        # Apply dark theme
        self._apply_theme()
        
        # Set up alert notification UI callback
        self._blink_alert.set_alert_callback(self._show_alert_notification)
    
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
        
        # --- Stacked area: Keyboard / Quick Phrases / Settings / Conversation ---
        self._stacked = QStackedWidget()
        
        self._keyboard = KeyboardWidget()
        self._quick_phrases = QuickPhrasesPanel()
        self._settings = SettingsPanel()
        self._conversation_panel = ConversationPanel()
        
        self._stacked.addWidget(self._keyboard)            # index 0
        self._stacked.addWidget(self._quick_phrases)        # index 1
        self._stacked.addWidget(self._settings)             # index 2
        self._stacked.addWidget(self._conversation_panel)   # index 3
        
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
        
        # Conversation result signal (worker thread → GUI thread)
        self._conversation_ready.connect(self._apply_conversation_options)

        # Gaze tracker → UI updates
        self._tracker.gaze_updated.connect(self._on_gaze_updated)
        self._tracker.frame_ready.connect(self._webcam_widget.update_frame)
        self._tracker.tracking_lost.connect(self._on_tracking_lost)
        
        # Blink detection → alert system
        self._tracker.blink_detected.connect(self._blink_alert.register_blink)
        
        # Calibration
        self._calibration.calibration_complete.connect(self._on_calibration_complete)
        self._calibration.calibration_cancelled.connect(self._on_calibration_cancelled)
        
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
        self._context_bar.end_conversation_requested.connect(self._on_conversation_type_instead)
        
        # Conversation panel
        self._conversation_panel.answer_selected.connect(self._on_conversation_answer)
        self._conversation_panel.more_options_requested.connect(self._on_conversation_more)
        self._conversation_panel.type_instead_requested.connect(self._on_conversation_type_instead)
        self._conversation_panel.back_requested.connect(self._on_conversation_back)
        self._conversation_panel.speak_now_requested.connect(self._on_conversation_speak_now)
    
    def _on_gaze_updated(self, gaze_x, gaze_y, confidence):
        """Handle gaze updates — gesture-based scan navigation with hysteresis."""
        # Feed raw data to calibration screen if active — skip navigation
        if self._calibration.isVisible():
            self._calibration.receive_gaze_sample(gaze_x, gaze_y, confidence)
            return

        # Skip navigation if the main window isn't visible yet
        if not self.isVisible():
            return

        # Determine raw horizontal zone from gaze position
        if gaze_x < self._zone_left:
            raw_zone = "LEFT"
        elif gaze_x > self._zone_right:
            raw_zone = "RIGHT"
        else:
            raw_zone = "CENTER"

        # ─── Conversation mode: use raw zone directly for responsiveness ───
        if self._conversation_mode and self._stacked.currentIndex() == 3:
            self._conversation_panel.set_zone(raw_zone)
            self._current_zone = raw_zone
            self._update_tracking_status(confidence)
            return

        # ─── Keyboard navigation: apply hysteresis before committing zone ───
        if raw_zone == self._current_zone:
            # Still in the committed zone — reset hysteresis counter
            self._pending_zone = raw_zone
            self._zone_frame_count = 0
        else:
            # Accumulate frames toward a potential zone change
            if raw_zone == self._pending_zone:
                self._zone_frame_count += 1
            else:
                # Direction changed before threshold — restart counter
                self._pending_zone = raw_zone
                self._zone_frame_count = 1

            if self._zone_frame_count >= self._HYSTERESIS_FRAMES:
                # Enough consecutive frames — commit the zone change
                prev_zone = self._current_zone
                self._current_zone = raw_zone
                self._zone_frame_count = 0
                self._pending_zone = raw_zone
                self._handle_zone_change(prev_zone, raw_zone)

        # Snap the gaze cursor to the highlighted item's center
        self._snap_cursor_to_highlight(confidence)

        # Update tracking status
        self._update_tracking_status(confidence)
    
    def _update_tracking_status(self, confidence):
        """Update the tracking status indicator."""
        if confidence > 0.5:
            self._status_indicator.setText("● Tracking")
            self._status_indicator.setStyleSheet("color: #50c878; padding: 8px;")
        elif confidence > 0.3:
            self._status_indicator.setText("● Weak")
            self._status_indicator.setStyleSheet("color: #f0c040; padding: 8px;")
        else:
            self._status_indicator.setText("● Low")
            self._status_indicator.setStyleSheet("color: #f06040; padding: 8px;")
    
    def _handle_zone_change(self, prev_zone, new_zone):
        """React to a committed zone transition.

        Gesture model:
          CENTER → LEFT or RIGHT : one deliberate step, disarm dwell
          LEFT or RIGHT → CENTER : stop, arm dwell (if user has gestured before)
        """
        if new_zone == "CENTER":
            # User returned to neutral — stop navigating and arm dwell
            self._keyboard.set_navigating(False)
            self._prediction_bar.set_navigating(False)
            if self._user_has_gestured:
                self._keyboard.arm_dwell()
        else:
            # Deliberate LEFT or RIGHT gesture — one step, dwell disarmed
            self._user_has_gestured = True
            self._keyboard.set_navigating(True)
            self._prediction_bar.set_navigating(True)
            direction = -1 if new_zone == "LEFT" else 1
            if self._nav_area == "predictions":
                self._step_prediction(direction)
            else:
                self._step_keyboard(direction)
    
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
        # Block layout-switching keys while conversation mode is active
        if self._conversation_mode and action in ("PHRASES", "SETTINGS"):
            return
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
        """Handle caretaker question — start conversation mode with LLM options."""
        self._abbreviation_expander.set_caretaker_context(question)
        print(f"[GazeSpeak] Caretaker context set: '{question}'")

        # Switch to conversation mode
        self._conversation_mode = True
        self._context_bar.set_conversation_active(True)
        self._conversation_panel.set_loading(question, [])
        self._stacked.setCurrentIndex(3)  # show conversation panel

        # Lock keyboard so its dwell timer cannot fire invisibly in the background
        self._keyboard.disarm_dwell()
        self._keyboard.set_navigating(True)

        # Generate answer options via LLM
        self._conversation_engine.start_conversation(
            question,
            callback=self._on_conversation_options_ready,
        )
    
    # ─── Conversation mode handlers ─────────────────────────────
    
    def _on_conversation_options_ready(self, result):
        """Called from LLM worker thread — emit signal to deliver to GUI thread."""
        print(f"[GazeSpeak] Conversation result ready: "
              f"left='{result.get('left')}' right='{result.get('right')}' "
              f"is_final={result.get('is_final')}")
        self._conversation_ready.emit(result)
    
    def _apply_conversation_options(self, result):
        """Apply conversation options on the UI thread (called via signal)."""
        print(f"[GazeSpeak] Applying conversation options on UI thread")
        if result.get('is_final'):
            # LLM decided the response is complete
            composed = result.get('composed_response', '')
            self._conversation_panel.set_final_response(
                composed,
                self._conversation_engine.current_question,
                self._conversation_engine.selections,
            )
            # Auto-speak the response
            self._sentence_bar.speak_text(composed)
            self._conversation_engine.record_final_response(composed)
            print(f"[GazeSpeak] Final response: '{composed}'")
        else:
            # Show the two options
            self._conversation_panel.set_options(
                left=result.get('left', 'Yes'),
                right=result.get('right', 'No'),
                question=self._conversation_engine.current_question,
                selections=self._conversation_engine.selections,
            )
    
    def _on_conversation_answer(self, chosen_text):
        """Handle patient selecting an answer card."""
        print(f"[GazeSpeak] Patient selected: '{chosen_text}'")
        
        # Show loading while generating follow-up
        self._conversation_panel.set_loading(
            self._conversation_engine.current_question,
            self._conversation_engine.selections + [chosen_text],
        )
        
        # Ask engine to generate follow-up options
        self._conversation_engine.select_option(
            chosen_text,
            callback=self._on_conversation_options_ready,
        )
    
    def _on_conversation_more(self):
        """Handle 'More Options' request."""
        self._conversation_panel.set_loading(
            self._conversation_engine.current_question,
            self._conversation_engine.selections,
        )
        self._conversation_engine.request_more_options(
            callback=self._on_conversation_options_ready,
        )
    
    def _on_conversation_type_instead(self):
        """Switch from conversation mode back to keyboard."""
        self._conversation_mode = False
        self._conversation_panel.clear()
        self._context_bar.set_conversation_active(False)
        self._stacked.setCurrentIndex(0)
        # Restore keyboard to a safe state — user must gesture before dwell re-arms
        self._keyboard.set_navigating(False)
        self._keyboard.disarm_dwell()
        self._user_has_gestured = False
        print("[GazeSpeak] Switched to keyboard mode")
    
    def _on_conversation_back(self):
        """Undo the last selection in the conversation."""
        removed = self._conversation_engine.undo_last_selection()
        if removed:
            print(f"[GazeSpeak] Undid selection: '{removed}'")
            # Regenerate options for the previous step
            self._conversation_panel.set_loading(
                self._conversation_engine.current_question,
                self._conversation_engine.selections,
            )
            self._conversation_engine.request_more_options(
                callback=self._on_conversation_options_ready,
            )
        else:
            # Nothing to undo — go back to keyboard
            self._on_conversation_type_instead()
    
    def _on_conversation_speak_now(self):
        """Compose and speak the response from selections so far."""
        selections = self._conversation_engine.selections
        if selections:
            self._conversation_panel.set_loading(
                self._conversation_engine.current_question,
                selections,
            )
            self._conversation_engine.compose_early(
                callback=self._on_conversation_options_ready,
            )
        else:
            print("[GazeSpeak] No selections yet to speak")
    
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
        # Pause blink alert so natural calibration blinks don’t trigger SOS
        self._blink_alert.pause()
        # Ensure tracker is running so calibration can receive gaze samples
        if not self._tracker.isRunning():
            self._tracker.start()
        self._calibration.start_calibration()
    
    def _on_calibration_complete(self, cal_data):
        """Apply calibration and compute zone thresholds from center_x."""
        self._tracker.set_calibration(cal_data)
        
        # Use the calibrated center point to define zone boundaries
        # After _apply_calibration maps raw→screen, center_norm tells us
        # where "straight ahead" sits in the 0.0–1.0 screen space.
        if isinstance(cal_data, dict) and 'center_x' in cal_data:
            left_x = cal_data['left_x']
            center_x = cal_data['center_x']
            right_x = cal_data['right_x']
            span = right_x - left_x

            if abs(span) > 0.01:
                # Map center to 0.0-1.0 range (same mapping the tracker uses)
                center_norm = (center_x - left_x) / span
                center_norm = max(0.2, min(0.8, center_norm))

                # Build dead zone around center (±20% of range — wider = easier to hit)
                margin = 0.20
                self._zone_left = center_norm - margin
                self._zone_right = center_norm + margin

                print(f"[GazeSpeak] Zone thresholds: "
                      f"LEFT < {self._zone_left:.2f} | "
                      f"CENTER {self._zone_left:.2f}-{self._zone_right:.2f} | "
                      f"RIGHT > {self._zone_right:.2f}")

        # Resume blink alert now that calibration is done
        self._blink_alert.resume()

        # Reset navigation state so no accidental dwell fires after calibration
        self._user_has_gestured = False
        self._current_zone = "CENTER"
        self._pending_zone = "CENTER"
        self._zone_frame_count = 0
        self._keyboard.disarm_dwell()
        self._keyboard.set_navigating(False)

        # On initial startup, now show the main window
        if self._initial_startup:
            self._initial_startup = False
            self._keyboard.set_highlight(0, 0)
            self.show()
            print("[GazeSpeak] ✓ Calibration complete — app is now active")
    
    def _on_calibration_cancelled(self):
        """Handle calibration cancellation — show app with default thresholds."""
        # Resume blink alert regardless of calibration outcome
        self._blink_alert.resume()
        print("[GazeSpeak] Calibration cancelled — using default zone thresholds")
        if self._initial_startup:
            self._initial_startup = False
            self._keyboard.set_highlight(0, 0)
            # Ensure tracker is running even without calibration
            if not self._tracker.isRunning():
                self._tracker.start()
            self.show()
    
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
    # Do NOT call window.show() here — the main window will be shown
    # after calibration completes (see _on_calibration_complete).
    
    # Start calibration immediately. This starts the tracker and shows
    # the calibration fullscreen overlay. The main app window stays hidden.
    QTimer.singleShot(500, window._start_calibration)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
