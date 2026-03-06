"""
ui/settings_panel.py — Settings and configuration panel.

Allows caregiver/patient to adjust dwell time, font size, TTS settings,
keyboard layout, and trigger recalibration.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QSlider, QPushButton, QComboBox, QGroupBox,
                              QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class SettingsPanel(QWidget):
    """
    Settings panel for configuring GazeSpeak.
    
    Signals:
        dwell_time_changed(int): ms
        font_size_changed(int): pt
        tts_rate_changed(int): words per minute
        theme_changed(str): 'dark' or 'light'
        recalibrate_requested()
        smoothing_changed(float): 0.1-1.0
        close_requested()
    """
    
    dwell_time_changed = pyqtSignal(int)
    font_size_changed = pyqtSignal(int)
    tts_rate_changed = pyqtSignal(int)
    theme_changed = pyqtSignal(str)
    recalibrate_requested = pyqtSignal()
    smoothing_changed = pyqtSignal(float)
    close_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #12121e;
                color: #d0d4e0;
                font-family: 'Segoe UI';
            }
            QGroupBox {
                border: 1px solid #2a2a4a;
                border-radius: 12px;
                margin-top: 16px;
                padding: 20px 16px 12px 16px;
                font-size: 15px;
                font-weight: bold;
                color: #a0a4c0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a4a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5090ff;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #3060c0;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #252540;
                border: 1px solid #3a3a5a;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                color: #d0d4e0;
            }
            QPushButton:hover {
                background-color: #353560;
                border-color: #5090ff;
            }
            QPushButton#calibrate_btn {
                background-color: #1a4a2a;
                border-color: #2a7a4a;
            }
            QPushButton#calibrate_btn:hover {
                background-color: #2a6a3a;
            }
            QPushButton#close_btn {
                background-color: #4a1a2a;
                border-color: #7a2a4a;
            }
            QPushButton#close_btn:hover {
                background-color: #6a2a3a;
            }
            QLabel {
                font-size: 13px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)
        
        # Title
        title = QLabel("⚙  Settings")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e4f0; padding-bottom: 8px;")
        layout.addWidget(title)
        
        # --- Gaze Control Group ---
        gaze_group = QGroupBox("Gaze Control")
        gaze_layout = QVBoxLayout()
        
        # Dwell time
        dwell_row = QHBoxLayout()
        dwell_label = QLabel("Dwell Time:")
        self._dwell_value = QLabel("800 ms")
        self._dwell_value.setStyleSheet("color: #5090ff; font-weight: bold;")
        self._dwell_slider = QSlider(Qt.Orientation.Horizontal)
        self._dwell_slider.setRange(300, 2500)
        self._dwell_slider.setValue(800)
        self._dwell_slider.valueChanged.connect(self._on_dwell_changed)
        dwell_row.addWidget(dwell_label)
        dwell_row.addWidget(self._dwell_slider)
        dwell_row.addWidget(self._dwell_value)
        gaze_layout.addLayout(dwell_row)
        
        # Smoothing
        smooth_row = QHBoxLayout()
        smooth_label = QLabel("Cursor Smoothing:")
        self._smooth_value = QLabel("0.30")
        self._smooth_value.setStyleSheet("color: #5090ff; font-weight: bold;")
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(5, 100)  # 0.05 to 1.0
        self._smooth_slider.setValue(30)
        self._smooth_slider.valueChanged.connect(self._on_smooth_changed)
        smooth_row.addWidget(smooth_label)
        smooth_row.addWidget(self._smooth_slider)
        smooth_row.addWidget(self._smooth_value)
        gaze_layout.addLayout(smooth_row)
        
        # Calibrate button
        calibrate_btn = QPushButton("🎯  Recalibrate Eye Tracking")
        calibrate_btn.setObjectName("calibrate_btn")
        calibrate_btn.clicked.connect(self.recalibrate_requested.emit)
        gaze_layout.addWidget(calibrate_btn)
        
        gaze_group.setLayout(gaze_layout)
        layout.addWidget(gaze_group)
        
        # --- Display Group ---
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        
        font_row = QHBoxLayout()
        font_label = QLabel("Sentence Font Size:")
        self._font_value = QLabel("28 pt")
        self._font_value.setStyleSheet("color: #5090ff; font-weight: bold;")
        self._font_slider = QSlider(Qt.Orientation.Horizontal)
        self._font_slider.setRange(16, 48)
        self._font_slider.setValue(28)
        self._font_slider.valueChanged.connect(self._on_font_changed)
        font_row.addWidget(font_label)
        font_row.addWidget(self._font_slider)
        font_row.addWidget(self._font_value)
        display_layout.addLayout(font_row)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # --- Speech Group ---
        speech_group = QGroupBox("Speech Output")
        speech_layout = QVBoxLayout()
        
        rate_row = QHBoxLayout()
        rate_label = QLabel("Speech Speed:")
        self._rate_value = QLabel("150 wpm")
        self._rate_value.setStyleSheet("color: #5090ff; font-weight: bold;")
        self._rate_slider = QSlider(Qt.Orientation.Horizontal)
        self._rate_slider.setRange(80, 250)
        self._rate_slider.setValue(150)
        self._rate_slider.valueChanged.connect(self._on_rate_changed)
        rate_row.addWidget(rate_label)
        rate_row.addWidget(self._rate_slider)
        rate_row.addWidget(self._rate_value)
        speech_layout.addLayout(rate_row)
        
        speech_group.setLayout(speech_layout)
        layout.addWidget(speech_group)
        
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("✕  Close Settings")
        close_btn.setObjectName("close_btn")
        close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(close_btn)
    
    def _on_dwell_changed(self, val):
        self._dwell_value.setText(f"{val} ms")
        self.dwell_time_changed.emit(val)
    
    def _on_smooth_changed(self, val):
        f = val / 100.0
        self._smooth_value.setText(f"{f:.2f}")
        self.smoothing_changed.emit(f)
    
    def _on_font_changed(self, val):
        self._font_value.setText(f"{val} pt")
        self.font_size_changed.emit(val)
    
    def _on_rate_changed(self, val):
        self._rate_value.setText(f"{val} wpm")
        self.tts_rate_changed.emit(val)
