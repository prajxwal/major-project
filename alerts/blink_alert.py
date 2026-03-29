"""
alerts/blink_alert.py — Rapid Blink Detection & Emergency Alert System.

Monitors blink events from the gaze tracker. If the user blinks rapidly
(≥6 blinks within a 4-second window), the system:
  1. Plays a loud repeating beep alarm
  2. Sends an SMS to the caregiver via Twilio API

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
        blink_threshold=6,
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

        # Alarm control
        self._alarm_active = False
        self._alarm_thread: threading.Thread | None = None

        # Alert callback (for UI notifications)
        self._on_alert_callback = None

    def set_alert_callback(self, callback):
        """Set a callback function to be called when alert triggers.
        Signature: callback(message: str)"""
        self._on_alert_callback = callback

    def register_blink(self):
        """
        Call this each time a blink is detected.
        Manages the sliding window and checks for rapid-blink trigger.
        """
        now = time.time()
        self._blink_times.append(now)

        # Purge old blinks outside the window
        while self._blink_times and (now - self._blink_times[0]) > self._time_window:
            self._blink_times.popleft()

        blink_count = len(self._blink_times)

        # Check trigger conditions
        if blink_count >= self._blink_threshold:
            if (now - self._last_alert_time) > self._cooldown:
                self._last_alert_time = now
                self._blink_times.clear()
                self._trigger_alert()

    def _trigger_alert(self):
        """Fire the emergency alert — beep + SMS."""
        print(f"[BlinkAlert] 🚨 EMERGENCY ALERT TRIGGERED — rapid blinks detected!")

        # Start alarm beeping in background
        self._start_alarm()

        # Send SMS in background thread (don't block the tracker)
        sms_thread = threading.Thread(target=self._send_sms, daemon=True)
        sms_thread.start()

        # Notify UI
        if self._on_alert_callback:
            self._on_alert_callback("🚨 Emergency alert sent to caregiver!")

    def _start_alarm(self):
        """Play repeating beep alarm (Windows winsound)."""
        if self._alarm_active:
            return

        self._alarm_active = True

        def _beep_loop():
            # Beep for 15 seconds (30 short beeps)
            for _ in range(30):
                if not self._alarm_active:
                    break
                try:
                    winsound.Beep(1800, 250)  # 1800 Hz for 250ms
                    time.sleep(0.15)          # 150ms gap
                    winsound.Beep(2200, 250)  # 2200 Hz for 250ms
                    time.sleep(0.15)
                except Exception:
                    break
            self._alarm_active = False

        self._alarm_thread = threading.Thread(target=_beep_loop, daemon=True)
        self._alarm_thread.start()

    def stop_alarm(self):
        """Stop the alarm beeping."""
        self._alarm_active = False

    def _send_sms(self):
        """Send emergency SMS to the caregiver via Twilio."""
        if not self._twilio_client:
            print("[BlinkAlert] ✗ Twilio not configured — skipping SMS")
            return

        if not self._caregiver_phone:
            print("[BlinkAlert] ✗ No caregiver phone number set — skipping SMS")
            return

        if not self._twilio_phone:
            print("[BlinkAlert] ✗ No Twilio phone number set — skipping SMS")
            return

        try:
            message = self._twilio_client.messages.create(
                body=(
                    "🚨 GazeSpeak EMERGENCY ALERT\n\n"
                    "Your patient triggered an emergency alert by blinking rapidly.\n"
                    "Please check on them immediately.\n\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                ),
                from_=self._twilio_phone,
                to=self._caregiver_phone,
            )
            print(f"[BlinkAlert] ✓ SMS sent — SID: {message.sid}")
        except Exception as e:
            print(f"[BlinkAlert] ✗ SMS failed: {e}")

    @property
    def is_alarm_active(self):
        return self._alarm_active
