"""
alerts/blink_alert.py — Rapid Blink Detection & Emergency Alert System.

Monitors blink events from the gaze tracker. If the user blinks rapidly
(≥5 blinks within a 4-second window), the system:
  1. Shows a full-screen red emergency overlay
  2. Plays a continuous siren alarm
  3. Sends an SMS to the caregiver via Twilio API

Designed as an SOS mechanism for ALS patients using GazeSpeak.
"""

import os
import time
import threading
import winsound
from collections import deque

from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()


class BlinkAlertManager:
    """
    Tracks rapid blinks and triggers emergency alerts.

    Parameters
    ----------
    blink_threshold : int
        Minimum number of blinks within the time window to trigger alert.
    time_window : float
        Seconds over which blinks are counted (default 4s).
    cooldown : float
        Seconds to wait after an alert before allowing another (default 60s).
    caregiver_phone : str
        Phone number to send SMS to (E.164 format, e.g. '+919876543210').
    twilio_phone : str
        Your Twilio phone number (E.164 format).
    """

    def __init__(
        self,
        blink_threshold=5,
        time_window=4.0,
        cooldown=60.0,
        caregiver_phone=None,
        twilio_phone=None,
    ):
        self._blink_threshold = blink_threshold
        self._time_window = time_window
        self._cooldown = cooldown
        self._caregiver_phone = caregiver_phone or os.getenv("CAREGIVER_PHONE", "")
        self._twilio_phone = twilio_phone or os.getenv("TWILIO_PHONE", "")

        # Twilio client
        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self._twilio_client = None
        if sid and token:
            try:
                self._twilio_client = Client(sid, token)
                print("[BlinkAlert] ✓ Twilio client initialized")
            except Exception as e:
                print(f"[BlinkAlert] ✗ Twilio init failed: {e}")

        # Blink timestamps (deque for sliding window)
        self._blink_times: deque[float] = deque()
        self._last_alert_time = 0.0
        self._paused = False  # True during calibration — blinks are ignored
        self._resume_time = 0.0  # time when detection last resumed
        self._GRACE_PERIOD = 2.0  # seconds to ignore blinks after resume

        # Alarm control
        self._alarm_active = False
        self._alarm_thread: threading.Thread | None = None

        # Alert callback (for UI notifications)
        self._on_alert_callback = None

        # SMS status callback: called with 'sending', 'sent', or 'failed'
        self._on_sms_status_callback = None

    def set_alert_callback(self, callback):
        """Set a callback function to be called when alert triggers.
        Signature: callback(message: str)"""
        self._on_alert_callback = callback

    def set_sms_status_callback(self, callback):
        """Set a callback to receive SMS delivery status updates.
        Signature: callback(status: str)  where status is 'sending', 'sent', or 'failed'"""
        self._on_sms_status_callback = callback

    def pause(self):
        """Pause blink detection (e.g. during calibration).
        Clears accumulated blink history so calibration blinks never count."""
        self._paused = True
        self._blink_times.clear()
        print("[BlinkAlert] ⏸ Blink detection paused")

    def resume(self):
        """Resume blink detection after calibration completes or is cancelled.
        Clears history again so any blinks during calibration don't carry over.
        Starts a grace period so residual queued blinks don't fire immediately."""
        self._paused = False
        self._blink_times.clear()
        self._last_alert_time = 0.0   # reset cooldown so next alert isn't blocked
        self._resume_time = time.time()
        print("[BlinkAlert] ▶ Blink detection resumed (2s grace period active)")

    def register_blink(self):
        """
        Call this each time a blink is detected.
        Manages the sliding window and checks for rapid-blink trigger.
        No-op while paused or during the grace period after resume.
        """
        if self._paused:
            return

        now = time.time()

        # Ignore blinks during the grace period right after resume
        if (now - self._resume_time) < self._GRACE_PERIOD:
            return

        self._blink_times.append(now)

        # Purge old blinks outside the window
        while self._blink_times and (now - self._blink_times[0]) > self._time_window:
            self._blink_times.popleft()

        blink_count = len(self._blink_times)
        print(f"[BlinkAlert] 👁 Blink #{blink_count} / {self._blink_threshold} "
              f"(window={self._time_window:.1f}s)")

        # Check trigger conditions
        if blink_count >= self._blink_threshold:
            if (now - self._last_alert_time) > self._cooldown:
                self._last_alert_time = now
                self._blink_times.clear()
                self._trigger_alert()

    def _trigger_alert(self):
        """Fire the emergency alert — siren + SMS."""
        print(f"[BlinkAlert] 🚨 EMERGENCY ALERT TRIGGERED — rapid blinks detected!")

        # Notify UI first (shows emergency screen)
        if self._on_alert_callback:
            self._on_alert_callback("🚨 Emergency alert sent to caregiver!")

        # Report SMS as 'sending' immediately
        if self._on_sms_status_callback:
            self._on_sms_status_callback("sending")

        # Start alarm siren in background
        self._start_alarm()

        # Send SMS in background thread (don't block the tracker)
        sms_thread = threading.Thread(target=self._send_sms, daemon=True)
        sms_thread.start()

    def _start_alarm(self):
        """Play a continuous siren alarm (Windows winsound) until stop_alarm() is called."""
        if self._alarm_active:
            return

        self._alarm_active = True

        def _siren_loop():
            """Alternating high/low tones — loops forever until _alarm_active is False."""
            while self._alarm_active:
                try:
                    # Rising wail — sweep from 800 Hz → 1200 Hz
                    for freq in range(800, 1201, 40):
                        if not self._alarm_active:
                            return
                        winsound.Beep(freq, 30)
                    # Falling wail — sweep from 1200 Hz → 800 Hz
                    for freq in range(1200, 799, -40):
                        if not self._alarm_active:
                            return
                        winsound.Beep(freq, 30)
                except Exception:
                    break
            self._alarm_active = False

        self._alarm_thread = threading.Thread(target=_siren_loop, daemon=True)
        self._alarm_thread.start()

    def stop_alarm(self):
        """Stop the alarm beeping."""
        self._alarm_active = False

    def _send_sms(self):
        """Send emergency SMS to the caregiver via Twilio."""
        if not self._twilio_client:
            print("[BlinkAlert] ✗ Twilio not configured — skipping SMS")
            if self._on_sms_status_callback:
                self._on_sms_status_callback("failed")
            return

        if not self._caregiver_phone:
            print("[BlinkAlert] ✗ No caregiver phone number set — skipping SMS")
            if self._on_sms_status_callback:
                self._on_sms_status_callback("failed")
            return

        if not self._twilio_phone:
            print("[BlinkAlert] ✗ No Twilio phone number set — skipping SMS")
            if self._on_sms_status_callback:
                self._on_sms_status_callback("failed")
            return

        try:
            message = self._twilio_client.messages.create(
                body=(
                    "SOS ALERT: Patient has triggered emergency blink signal "
                    f"({self._blink_threshold} blinks). Please check immediately."
                ),
                from_=self._twilio_phone,
                to=self._caregiver_phone,
            )
            print(f"[BlinkAlert] ✓ SMS sent — SID: {message.sid}")
            if self._on_sms_status_callback:
                self._on_sms_status_callback("sent")
        except Exception as e:
            print(f"[BlinkAlert] ✗ SMS failed: {e}")
            if self._on_sms_status_callback:
                self._on_sms_status_callback("failed")

    @property
    def is_alarm_active(self):
        return self._alarm_active
