"""
GazeSpeak — Assistive Eye-Gaze Typing for ALS Patients

Main entry point. Launches the PyQt6 application with:
- Webcam-based eye gaze tracking (MediaPipe Face Mesh)
- 9-point calibration on first launch
- On-screen QWERTY keyboard with dwell-time selection
- Word prediction with frequency-ranked dictionary
- Quick phrases for common needs
- Text-to-speech output
- Caregiver mode (mouse/keyboard fallback)

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
from prediction.predictor import WordPredictor


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
        self._calibration = CalibrationScreen()
        
        # Build UI
        self._setup_ui()
        self._connect_signals()
        
        # Load saved calibration or start fresh
        saved_cal = CalibrationScreen.load_calibration()
        if saved_cal is not None:
            self._tracker.set_calibration(saved_cal)
        
        # Start gaze tracking
        self._tracker.start()
        
        # Apply dark theme
        self._apply_theme()
    
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
    
    def _on_gaze_updated(self, gaze_x, gaze_y, confidence):
        """Handle gaze position updates from the tracker."""
        # Map normalized gaze (0-1) to window coordinates
        wx = gaze_x * self.width()
        wy = gaze_y * self.height()
        
        # Update all gaze-aware widgets
        self._gaze_cursor.update_position(wx, wy, confidence)
        
        # Map to keyboard widget coordinates
        kb_pos = self._keyboard.mapFrom(self.centralWidget(), QPoint(int(wx), int(wy)))
        self._keyboard.update_gaze_position(kb_pos.x(), kb_pos.y())
        
        # Map to prediction bar
        pred_pos = self._prediction_bar.mapFrom(self.centralWidget(), QPoint(int(wx), int(wy)))
        self._prediction_bar.update_gaze_position(pred_pos.x(), pred_pos.y())
        
        # Map to quick phrases (if visible)
        if self._stacked.currentIndex() == 1:
            qp_pos = self._quick_phrases.mapFrom(self.centralWidget(), QPoint(int(wx), int(wy)))
            self._quick_phrases.update_gaze_position(qp_pos.x(), qp_pos.y())
        
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
        """Handle word prediction selection."""
        self._sentence_bar.add_word(word)
    
    def _on_text_changed(self, text):
        """Update word predictions when text changes — hybrid local + LLM."""
        current_word = self._sentence_bar.get_current_word()
        
        if current_word:
            # Instant local predictions
            local_predictions = self._predictor.predict(current_word)
            self._prediction_bar.set_predictions(local_predictions)
            
            # Async LLM predictions (replace local when ready)
            self._predictor.predict_with_llm(
                text, current_word,
                callback=self._on_llm_predictions
            )
        elif text.endswith(" "):
            # Just finished a word — predict next word
            self._predictor.predict_next_word(
                text,
                callback=self._on_llm_predictions
            )
        else:
            self._prediction_bar.set_predictions([])
    
    def _on_llm_predictions(self, words):
        """Handle async LLM prediction results (called from background thread)."""
        # Use QTimer to safely update UI from background thread
        QTimer.singleShot(0, lambda: self._prediction_bar.set_predictions(words))
    
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
    
    def _on_calibration_complete(self, matrix):
        """Apply the new calibration matrix."""
        self._tracker.set_calibration(matrix)
    
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
    
    def closeEvent(self, event):
        """Clean up on close."""
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
    
    # Show calibration on first run (if no saved calibration)
    if CalibrationScreen.load_calibration() is None:
        QTimer.singleShot(1000, window._start_calibration)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
