"""
audio/speech_to_text.py — Real-time Speech-to-Text using AssemblyAI.

Streams microphone audio to AssemblyAI's real-time transcription API
and emits partial/final transcripts as Qt signals.

Used by the caretaker to speak their question instead of typing.
"""

import os
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from dotenv import load_dotenv

load_dotenv()


class SpeechToTextWorker(QObject):
    """
    Real-time speech-to-text using AssemblyAI streaming API.
    
    Runs in a background thread, captures microphone audio, and streams
    it to AssemblyAI for real-time transcription.
    
    Signals:
        partial_transcript(str): interim text as user speaks
        final_transcript(str): finalized sentence with punctuation
        error_occurred(str): error message
        session_started(): recording has begun
        session_ended(): recording has stopped
    """
    
    partial_transcript = pyqtSignal(str)
    final_transcript = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    session_started = pyqtSignal()
    session_ended = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
        self._transcriber = None
        self._mic_stream = None
        self._is_recording = False
        self._thread = None
        self._accumulated_text = ""  # accumulates final transcripts
        
        if not self._api_key:
            print("[STT] ✗ No ASSEMBLYAI_API_KEY found in .env")
        else:
            print("[STT] ✓ AssemblyAI API key loaded")
    
    @property
    def is_recording(self):
        return self._is_recording
    
    @property
    def is_available(self):
        """Check if STT is available (API key set + packages installed)."""
        if not self._api_key:
            return False
        try:
            import assemblyai  # noqa: F401
            return True
        except ImportError:
            return False
    
    def start_recording(self):
        """Start recording and streaming to AssemblyAI."""
        if self._is_recording:
            return
        
        if not self._api_key:
            self.error_occurred.emit("No AssemblyAI API key configured")
            return
        
        self._accumulated_text = ""
        self._is_recording = True
        self._thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._thread.start()
    
    def stop_recording(self):
        """Stop recording and close the streaming session."""
        if not self._is_recording:
            return
        
        self._is_recording = False
        
        try:
            if self._transcriber:
                self._transcriber.close()
                self._transcriber = None
        except Exception as e:
            print(f"[STT] Error closing transcriber: {e}")
        
        self.session_ended.emit()
    
    def _recording_loop(self):
        """Background thread: connect to AssemblyAI and stream mic audio."""
        try:
            import assemblyai as aai
            
            aai.settings.api_key = self._api_key
            
            def on_open(session_opened: aai.RealtimeSessionOpened):
                print(f"[STT] ✓ Session opened: {session_opened.session_id}")
                self.session_started.emit()
            
            def on_data(transcript: aai.RealtimeTranscript):
                if not transcript.text:
                    return
                
                if isinstance(transcript, aai.RealtimeFinalTranscript):
                    # Final transcript — accumulate
                    if self._accumulated_text:
                        self._accumulated_text += " " + transcript.text
                    else:
                        self._accumulated_text = transcript.text
                    self.final_transcript.emit(self._accumulated_text)
                else:
                    # Partial transcript — show current speech + accumulated
                    partial = self._accumulated_text
                    if partial:
                        partial += " " + transcript.text
                    else:
                        partial = transcript.text
                    self.partial_transcript.emit(partial)
            
            def on_error(error: aai.RealtimeError):
                print(f"[STT] ✗ Error: {error}")
                self.error_occurred.emit(str(error))
            
            def on_close():
                print("[STT] Session closed")
                self._is_recording = False
                self.session_ended.emit()
            
            self._transcriber = aai.RealtimeTranscriber(
                on_data=on_data,
                on_error=on_error,
                on_open=on_open,
                on_close=on_close,
                sample_rate=16_000,
            )
            
            self._transcriber.connect()
            
            # Stream from microphone (blocks until closed)
            self._mic_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
            self._transcriber.stream(self._mic_stream)
            
        except ImportError as e:
            error_msg = f"Missing package: {e}. Install with: pip install assemblyai pyaudio"
            print(f"[STT] ✗ {error_msg}")
            self.error_occurred.emit(error_msg)
        except Exception as e:
            print(f"[STT] ✗ Recording error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._is_recording = False
            self.session_ended.emit()
    
    def get_accumulated_text(self):
        """Return all accumulated final transcripts from this session."""
        return self._accumulated_text.strip()
