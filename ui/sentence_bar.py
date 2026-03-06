"""
ui/sentence_bar.py — Sentence display and TTS output.

Shows the composed text at the top of the screen with controls for
speaking, clearing, and backspacing.
"""

import pyttsx3
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor


class TTSWorker(QThread):
    """Background thread for text-to-speech to avoid blocking the UI."""
    finished = pyqtSignal()
    
    def __init__(self, text, rate=150, voice_id=None):
        super().__init__()
        self._text = text
        self._rate = rate
        self._voice_id = voice_id
    
    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            engine.say(self._text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass
        self.finished.emit()


class SentenceBar(QWidget):
    """
    Top bar displaying the composed sentence with blinking cursor.
    
    Supports:
    - Adding characters/words
    - Backspace (delete last character)
    - Clear all
    - Speak via pyttsx3
    """
    
    text_changed = pyqtSignal(str)  # emitted when text changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._tts_worker = None
        self._tts_rate = 150
        self._tts_voice_id = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        
        self._label = QLabel("")
        self._label.setFont(QFont("Segoe UI", 28, QFont.Weight.Normal))
        self._label.setStyleSheet("""
            QLabel {
                color: #e0e4f0;
                background-color: #1a1a2e;
                border: 2px solid #2a2a4a;
                border-radius: 16px;
                padding: 16px 24px;
                min-height: 60px;
            }
        """)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label.setWordWrap(True)
        
        layout.addWidget(self._label)
        
        self.setStyleSheet("background: transparent;")
    
    def add_character(self, char):
        """Add a character to the sentence."""
        self._text += char
        self._update_display()
        self.text_changed.emit(self._text)
    
    def add_word(self, word):
        """Add a complete word (from prediction), replacing partial input."""
        # Find the last incomplete word and replace it
        parts = self._text.rsplit(" ", 1)
        if len(parts) > 1:
            self._text = parts[0] + " " + word + " "
        else:
            self._text = word + " "
        self._update_display()
        self.text_changed.emit(self._text)
    
    def backspace(self):
        """Delete the last character."""
        if self._text:
            self._text = self._text[:-1]
            self._update_display()
            self.text_changed.emit(self._text)
    
    def add_space(self):
        """Add a space."""
        self._text += " "
        self._update_display()
        self.text_changed.emit(self._text)
    
    def clear(self):
        """Clear all text."""
        self._text = ""
        self._update_display()
        self.text_changed.emit(self._text)
    
    def speak(self):
        """Speak the current sentence using TTS."""
        if not self._text.strip():
            return
        
        if self._tts_worker and self._tts_worker.isRunning():
            return  # already speaking
        
        self._tts_worker = TTSWorker(self._text.strip(), self._tts_rate, self._tts_voice_id)
        self._tts_worker.start()
    
    def speak_text(self, text):
        """Speak arbitrary text (for quick phrases)."""
        if self._tts_worker and self._tts_worker.isRunning():
            return
        
        self._tts_worker = TTSWorker(text, self._tts_rate, self._tts_voice_id)
        self._tts_worker.start()
    
    def set_tts_rate(self, rate):
        self._tts_rate = rate
    
    def set_tts_voice(self, voice_id):
        self._tts_voice_id = voice_id
    
    def get_text(self):
        return self._text
    
    def get_current_word(self):
        """Get the word currently being typed (for prediction)."""
        if not self._text:
            return ""
        parts = self._text.split(" ")
        return parts[-1] if parts else ""
    
    def _update_display(self):
        """Update the label text with a blinking cursor."""
        display = self._text + "│"  # Unicode cursor character
        self._label.setText(display)
