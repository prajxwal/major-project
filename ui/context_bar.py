"""
ui/context_bar.py — Caretaker Context Input Bar with Speech-to-Text.

Provides two input modes for the caretaker:
1. Text input (typing) — always available as fallback
2. Microphone input (STT via AssemblyAI) — press mic button to speak

The transcribed or typed question is submitted as context for the
abbreviation expander.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from audio import SpeechToTextWorker


class ContextBar(QWidget):
    """
    Bar for caretaker to provide context via typing or voice.
    
    Signals:
        context_submitted(str): emitted when caretaker submits a question
        end_conversation_requested(): emitted to end conversation mode
    """
    
    context_submitted = pyqtSignal(str)
    end_conversation_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stt = SpeechToTextWorker()
        self._is_recording = False
        self._conversation_active = False
        self._setup_ui()
        self._connect_stt()
    
    def _setup_ui(self):
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QWidget {
                background-color: #0e0e1a;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)
        
        # Caretaker label
        label = QLabel("Caretaker")
        label.setFont(QFont("Roboto Mono", 10, QFont.Weight.Bold))
        label.setStyleSheet("color: #5a5e78; letter-spacing: 1px;")
        label.setFixedWidth(70)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        # Microphone button
        self._mic_btn = QPushButton("Speak")
        self._mic_btn.setFont(QFont("Roboto Mono", 11, QFont.Weight.DemiBold))
        self._mic_btn.setFixedSize(110, 40)
        self._mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mic_btn.setStyleSheet(self._mic_idle_style())
        self._mic_btn.clicked.connect(self._toggle_recording)
        
        # Disable mic button if STT not available
        if not self._stt.is_available:
            self._mic_btn.setEnabled(False)
            self._mic_btn.setToolTip("Install assemblyai & pyaudio packages")
            self._mic_btn.setStyleSheet(self._mic_disabled_style())
        
        layout.addWidget(self._mic_btn)
        
        # Text input (fallback + shows live transcript)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask a question (e.g. 'How are you feeling?') → patient picks answer...")
        self._input.setFont(QFont("Roboto Mono", 13))
        self._input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a2e;
                border: 1px solid #2a2a4a;
                border-radius: 10px;
                padding: 8px 16px;
                color: #c0c4d8;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #5090ff;
                background-color: #1e1e34;
            }
        """)
        self._input.returnPressed.connect(self._submit)
        layout.addWidget(self._input, stretch=1)
        
        # Submit button — "Ask" sends the question to conversation mode
        self._submit_btn = QPushButton("Ask  ->")
        self._submit_btn.setFont(QFont("Roboto Mono", 11, QFont.Weight.DemiBold))
        self._submit_btn.setFixedHeight(40)
        self._submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a4a8a;
                border: 1px solid #3a6acc;
                border-radius: 8px;
                padding: 6px 16px;
                color: #d0d8f0;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a5aaa;
                border-color: #5090ff;
            }
            QPushButton:pressed {
                background-color: #1a3a6a;
            }
        """)
        self._submit_btn.clicked.connect(self._submit)
        layout.addWidget(self._submit_btn)
        
        # End conversation button (hidden by default)
        self._end_btn = QPushButton("X  End")
        self._end_btn.setFont(QFont("Roboto Mono", 10, QFont.Weight.DemiBold))
        self._end_btn.setFixedSize(80, 40)
        self._end_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._end_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a1a2a;
                border: 1px solid #7a2a4a;
                border-radius: 8px;
                padding: 4px 10px;
                color: #ff8080;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #6a2a3a;
                border-color: #ff4040;
            }
        """)
        self._end_btn.clicked.connect(self._end_conversation)
        self._end_btn.hide()
        layout.addWidget(self._end_btn)
        
        # Current context indicator
        self._context_label = QLabel("")
        self._context_label.setFont(QFont("Roboto Mono", 9))
        self._context_label.setStyleSheet("color: #5a5e78; font-style: italic;")
        self._context_label.setFixedWidth(180)
        self._context_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._context_label.setWordWrap(True)
        layout.addWidget(self._context_label)
        
        # Recording pulse animation timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(600)
        self._pulse_timer.timeout.connect(self._pulse_mic)
        self._pulse_on = False
    
    def set_conversation_active(self, active):
        """Toggle conversation mode visual state."""
        self._conversation_active = active
        if active:
            self._end_btn.show()
            self._context_label.setText("Conversation active")
            self._context_label.setStyleSheet("color: #50c878; font-style: italic;")
        else:
            self._end_btn.hide()
            self._context_label.setText("")
            self._context_label.setStyleSheet("color: #5a5e78; font-style: italic;")
    
    def _end_conversation(self):
        """End the current conversation."""
        self.set_conversation_active(False)
        self.end_conversation_requested.emit()
    
    def _connect_stt(self):
        """Connect STT signals."""
        self._stt.partial_transcript.connect(self._on_partial_transcript)
        self._stt.final_transcript.connect(self._on_final_transcript)
        self._stt.error_occurred.connect(self._on_stt_error)
        self._stt.session_started.connect(self._on_session_started)
        self._stt.session_ended.connect(self._on_session_ended)
    
    def _toggle_recording(self):
        """Toggle microphone recording on/off."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start microphone recording."""
        self._is_recording = True
        self._input.clear()
        self._input.setPlaceholderText("[REC] Listening... speak your question")
        self._mic_btn.setText("[ ] Stop")
        self._mic_btn.setStyleSheet(self._mic_recording_style())
        self._pulse_timer.start()
        self._stt.start_recording()
    
    def _stop_recording(self):
        """Stop microphone recording and auto-submit."""
        self._is_recording = False
        self._pulse_timer.stop()
        self._mic_btn.setText("Speak")
        self._mic_btn.setStyleSheet(self._mic_idle_style())
        self._input.setPlaceholderText("Type or speak your question (e.g. 'Where does it hurt?')...")
        self._stt.stop_recording()
        
        # Auto-submit the accumulated text
        text = self._input.text().strip()
        if text:
            QTimer.singleShot(300, self._submit)
    
    def _on_partial_transcript(self, text):
        """Show live partial transcript in the input field."""
        self._input.setText(text)
        self._input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a2e;
                border: 1px solid #4a3a2a;
                border-radius: 10px;
                padding: 8px 16px;
                color: #f0c860;
                font-size: 13px;
            }
        """)
    
    def _on_final_transcript(self, text):
        """Show finalized transcript in the input field."""
        self._input.setText(text)
        self._input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a2e;
                border: 1px solid #2a4a2a;
                border-radius: 10px;
                padding: 8px 16px;
                color: #60d880;
                font-size: 13px;
            }
        """)
    
    def _on_stt_error(self, error):
        """Handle STT errors."""
        print(f"[ContextBar] STT error: {error}")
        self._stop_recording()
        self._context_label.setText("! STT Error")
        self._context_label.setStyleSheet("color: #ff6060; font-style: italic;")
    
    def _on_session_started(self):
        """Handle STT session start."""
        self._context_label.setText("[REC] Recording...")
        self._context_label.setStyleSheet("color: #ff4040; font-style: italic;")
    
    def _on_session_ended(self):
        """Handle STT session end."""
        self._is_recording = False
        self._pulse_timer.stop()
        self._mic_btn.setText("Speak")
        self._mic_btn.setStyleSheet(self._mic_idle_style())
    
    def _pulse_mic(self):
        """Animate the mic button while recording."""
        self._pulse_on = not self._pulse_on
        if self._pulse_on:
            self._mic_btn.setStyleSheet(self._mic_recording_pulse_style())
        else:
            self._mic_btn.setStyleSheet(self._mic_recording_style())
    
    def _submit(self):
        """Submit the context (typed or transcribed)."""
        text = self._input.text().strip()
        if text:
            self.context_submitted.emit(text)
            self._context_label.setText(f">> \"{text[:35]}{'...' if len(text) > 35 else ''}\"")
            self._context_label.setStyleSheet("color: #50c878; font-style: italic;")
            self._input.clear()
            # Reset input style
            self._input.setStyleSheet("""
                QLineEdit {
                    background-color: #1a1a2e;
                    border: 1px solid #2a2a4a;
                    border-radius: 10px;
                    padding: 8px 16px;
                    color: #c0c4d8;
                    font-size: 13px;
                }
                QLineEdit:focus {
                    border-color: #5090ff;
                    background-color: #1e1e34;
                }
            """)
    
    def get_current_text(self):
        return self._input.text()
    
    # --- Styles ---
    
    @staticmethod
    def _mic_idle_style():
        return """
            QPushButton {
                background-color: #1a2a4a;
                border: 1px solid #2a4a6a;
                border-radius: 8px;
                color: #80b0e0;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2a3a5a;
                border-color: #4a8aff;
            }
        """
    
    @staticmethod
    def _mic_recording_style():
        return """
            QPushButton {
                background-color: #5a1a1a;
                border: 2px solid #ff4040;
                border-radius: 8px;
                color: #ff8080;
                font-size: 11px;
            }
        """
    
    @staticmethod
    def _mic_recording_pulse_style():
        return """
            QPushButton {
                background-color: #7a2a2a;
                border: 2px solid #ff6060;
                border-radius: 8px;
                color: #ffa0a0;
                font-size: 11px;
            }
        """
    
    @staticmethod
    def _mic_disabled_style():
        return """
            QPushButton {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                color: #4a4a4a;
                font-size: 11px;
            }
        """
