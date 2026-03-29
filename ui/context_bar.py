"""
ui/context_bar.py — Caretaker Context Input Bar.

A text input at the bottom of the screen where the caretaker can type their
question. This provides context for the abbreviation expander so the patient's
short abbreviations can be intelligently expanded.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ContextBar(QWidget):
    """
    Bar for caretaker to type their question, providing context
    for the abbreviation expansion engine.
    
    Signals:
        context_submitted(str): emitted when caretaker submits a question
    """
    
    context_submitted = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setFixedHeight(56)
        self.setStyleSheet("""
            QWidget {
                background-color: #0e0e1a;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)
        
        # Caretaker icon/label
        icon_label = QLabel("🗣")
        icon_label.setFont(QFont("Segoe UI", 16))
        icon_label.setFixedWidth(30)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)
        
        # Text input
        self._input = QLineEdit()
        self._input.setPlaceholderText("Caretaker: Type your question here for context (e.g. 'Where does it hurt?')...")
        self._input.setFont(QFont("Segoe UI", 13))
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
        
        # Submit button
        submit_btn = QPushButton("Set Context")
        submit_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        submit_btn.setFixedHeight(36)
        submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        submit_btn.setStyleSheet("""
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
        submit_btn.clicked.connect(self._submit)
        layout.addWidget(submit_btn)
        
        # Current context indicator
        self._context_label = QLabel("")
        self._context_label.setFont(QFont("Segoe UI", 10))
        self._context_label.setStyleSheet("color: #5a5e78; font-style: italic;")
        self._context_label.setFixedWidth(200)
        self._context_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._context_label.setWordWrap(True)
        layout.addWidget(self._context_label)
    
    def _submit(self):
        text = self._input.text().strip()
        if text:
            self.context_submitted.emit(text)
            self._context_label.setText(f"📌 \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
            self._context_label.setStyleSheet("color: #50c878; font-style: italic;")
            self._input.clear()
    
    def get_current_text(self):
        return self._input.text()
