"""
gaze/tracker.py — Eye gaze tracking engine using MediaPipe Face Landmarker (Tasks API) + OpenCV.

Captures webcam frames in a background thread, runs MediaPipe Face Landmarker to detect
478 facial landmarks + iris landmarks, and computes screen gaze coordinates.

Uses the new mediapipe.tasks API (compatible with mediapipe >= 0.10.x).
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PyQt6.QtCore import QThread, pyqtSignal, QMutex


# MediaPipe Face Landmarker landmark indices
# Left eye corners
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Right eye corners
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Iris landmarks (enabled when output_face_blendshapes or refine_landmarks is used)
LEFT_IRIS = [468, 469, 470, 471, 472]   # center, right, top, left, bottom
RIGHT_IRIS = [473, 474, 475, 476, 477]  # center, right, top, left, bottom

# Path to the face landmarker model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


class GazeTracker(QThread):
    """
    Background thread that captures webcam frames, processes them through
    MediaPipe Face Landmarker, and emits gaze coordinates.
    
    Signals:
        gaze_updated(float, float, float): x, y screen position, confidence
        frame_ready(ndarray): processed frame with landmarks drawn
        tracking_lost(): emitted when face/eyes not detected
    """
    
    gaze_updated = pyqtSignal(float, float, float)  # x, y, confidence
    frame_ready = pyqtSignal(np.ndarray)
    tracking_lost = pyqtSignal()
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._running = False
        self._mutex = QMutex()
        
        # Calibration coefficients (set by calibration module)
        self._calibration_matrix = None  # 3x3 affine transform
        
        # Smoothing: exponential moving average
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._smooth_factor = 0.3  # lower = smoother but more latency
        
    def set_calibration(self, matrix):
        """Set the calibration transformation matrix."""
        self._mutex.lock()
        self._calibration_matrix = matrix
        self._mutex.unlock()
        
    def set_smoothing(self, factor):
        """Set smoothing factor (0.1 = very smooth, 1.0 = no smoothing)."""
        self._smooth_factor = max(0.05, min(1.0, factor))
    
    def stop(self):
        """Stop the tracking thread."""
        self._running = False
        
    def run(self):
        """Main tracking loop — runs in background thread."""
        self._running = True
        
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create Face Landmarker using the new Tasks API
        if not os.path.exists(MODEL_PATH):
            print(f"[GazeSpeak] ✗ Face landmarker model not found at: {MODEL_PATH}")
            print("[GazeSpeak] Please download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
            self._running = False
            return
        
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            result = landmarker.detect(mp_image)
            
            if result.face_landmarks and len(result.face_landmarks) > 0:
                landmarks = result.face_landmarks[0]  # first face
                
                # Check if iris landmarks are available (indices 468-477)
                if len(landmarks) > 477:
                    gaze_data = self._compute_gaze(landmarks, w, h)
                    
                    if gaze_data is not None:
                        raw_x, raw_y, confidence = gaze_data
                        
                        # Apply calibration if available
                        screen_x, screen_y = self._apply_calibration(raw_x, raw_y)
                        
                        # Apply exponential moving average smoothing
                        self._smooth_x = (self._smooth_factor * screen_x + 
                                          (1 - self._smooth_factor) * self._smooth_x)
                        self._smooth_y = (self._smooth_factor * screen_y + 
                                          (1 - self._smooth_factor) * self._smooth_y)
                        
                        self.gaze_updated.emit(self._smooth_x, self._smooth_y, confidence)
                        
                        # Draw landmarks on frame for visual feedback
                        self._draw_landmarks(frame, landmarks, w, h)
                    else:
                        self.tracking_lost.emit()
                else:
                    # Face detected but no iris landmarks
                    gaze_data = self._compute_gaze_from_eyes(landmarks, w, h)
                    if gaze_data:
                        raw_x, raw_y, confidence = gaze_data
                        screen_x, screen_y = self._apply_calibration(raw_x, raw_y)
                        self._smooth_x = (self._smooth_factor * screen_x + 
                                          (1 - self._smooth_factor) * self._smooth_x)
                        self._smooth_y = (self._smooth_factor * screen_y + 
                                          (1 - self._smooth_factor) * self._smooth_y)
                        self.gaze_updated.emit(self._smooth_x, self._smooth_y, confidence * 0.7)
                    else:
                        self.tracking_lost.emit()
            else:
                self.tracking_lost.emit()
            
            self.frame_ready.emit(frame)
        
        cap.release()
        landmarker.close()
    
    def _compute_gaze(self, landmarks, frame_w, frame_h):
        """
        Compute gaze direction from iris landmarks.
        
        Returns (gaze_ratio_x, gaze_ratio_y, confidence) where ratios are 0.0–1.0.
        """
        try:
            # Get iris centers (landmarks are NormalizedLandmark with x, y, z in [0,1])
            left_iris_center = np.array([
                landmarks[LEFT_IRIS[0]].x * frame_w,
                landmarks[LEFT_IRIS[0]].y * frame_h
            ])
            right_iris_center = np.array([
                landmarks[RIGHT_IRIS[0]].x * frame_w,
                landmarks[RIGHT_IRIS[0]].y * frame_h
            ])
            
            # Get eye corners for reference frame
            left_inner = np.array([landmarks[LEFT_EYE_INNER].x * frame_w,
                                   landmarks[LEFT_EYE_INNER].y * frame_h])
            left_outer = np.array([landmarks[LEFT_EYE_OUTER].x * frame_w,
                                   landmarks[LEFT_EYE_OUTER].y * frame_h])
            left_top = np.array([landmarks[LEFT_EYE_TOP].x * frame_w,
                                 landmarks[LEFT_EYE_TOP].y * frame_h])
            left_bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x * frame_w,
                                    landmarks[LEFT_EYE_BOTTOM].y * frame_h])
            
            right_inner = np.array([landmarks[RIGHT_EYE_INNER].x * frame_w,
                                    landmarks[RIGHT_EYE_INNER].y * frame_h])
            right_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * frame_w,
                                    landmarks[RIGHT_EYE_OUTER].y * frame_h])
            right_top = np.array([landmarks[RIGHT_EYE_TOP].x * frame_w,
                                  landmarks[RIGHT_EYE_TOP].y * frame_h])
            right_bottom = np.array([landmarks[RIGHT_EYE_BOTTOM].x * frame_w,
                                     landmarks[RIGHT_EYE_BOTTOM].y * frame_h])
            
            # Compute gaze ratio for each eye
            left_eye_width = np.linalg.norm(left_inner - left_outer)
            right_eye_width = np.linalg.norm(right_inner - right_outer)
            
            if left_eye_width < 1 or right_eye_width < 1:
                return None
            
            # Horizontal ratio
            left_ratio_x = (left_iris_center[0] - left_outer[0]) / left_eye_width
            right_ratio_x = (right_iris_center[0] - right_outer[0]) / right_eye_width
            
            # Vertical ratio
            left_eye_height = np.linalg.norm(left_top - left_bottom)
            right_eye_height = np.linalg.norm(right_top - right_bottom)
            
            if left_eye_height < 1 or right_eye_height < 1:
                return None
            
            left_ratio_y = (left_iris_center[1] - left_top[1]) / left_eye_height
            right_ratio_y = (right_iris_center[1] - right_top[1]) / right_eye_height
            
            # Average both eyes
            gaze_x = (left_ratio_x + right_ratio_x) / 2.0
            gaze_y = (left_ratio_y + right_ratio_y) / 2.0
            
            # Clamp to [0, 1]
            gaze_x = max(0.0, min(1.0, gaze_x))
            gaze_y = max(0.0, min(1.0, gaze_y))
            
            # Confidence based on eye agreement
            x_agreement = 1.0 - abs(left_ratio_x - right_ratio_x)
            y_agreement = 1.0 - abs(left_ratio_y - right_ratio_y)
            confidence = (x_agreement + y_agreement) / 2.0
            
            return (gaze_x, gaze_y, confidence)
            
        except (IndexError, ZeroDivisionError, AttributeError):
            return None
    
    def _compute_gaze_from_eyes(self, landmarks, frame_w, frame_h):
        """
        Fallback gaze computation using only eye corner landmarks (no iris).
        Less accurate but works when iris landmarks aren't available.
        """
        try:
            left_inner = np.array([landmarks[LEFT_EYE_INNER].x, landmarks[LEFT_EYE_INNER].y])
            left_outer = np.array([landmarks[LEFT_EYE_OUTER].x, landmarks[LEFT_EYE_OUTER].y])
            right_inner = np.array([landmarks[RIGHT_EYE_INNER].x, landmarks[RIGHT_EYE_INNER].y])
            right_outer = np.array([landmarks[RIGHT_EYE_OUTER].x, landmarks[RIGHT_EYE_OUTER].y])
            
            # Use midpoint of eyes as approximate gaze
            left_center = (left_inner + left_outer) / 2
            right_center = (right_inner + right_outer) / 2
            gaze = (left_center + right_center) / 2
            
            return (float(gaze[0]), float(gaze[1]), 0.4)
        except (IndexError, AttributeError):
            return None
    
    def _apply_calibration(self, raw_x, raw_y):
        """Apply calibration matrix to map raw gaze ratios to screen coordinates."""
        self._mutex.lock()
        matrix = self._calibration_matrix
        self._mutex.unlock()
        
        if matrix is not None:
            point = np.array([raw_x, raw_y, 1.0])
            result = matrix @ point
            return float(result[0]), float(result[1])
        else:
            return raw_x, raw_y
    
    def _draw_landmarks(self, frame, landmarks, w, h):
        """Draw iris and eye landmarks on the frame for visual feedback."""
        # Draw iris centers
        if len(landmarks) > 477:
            for idx in [LEFT_IRIS[0], RIGHT_IRIS[0]]:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw iris rings
            for iris_indices in [LEFT_IRIS, RIGHT_IRIS]:
                points = []
                for idx in iris_indices[1:]:  # skip center
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    points.append((x, y))
                if len(points) == 4:
                    for i in range(len(points)):
                        cv2.line(frame, points[i], points[(i + 1) % len(points)],
                                 (0, 255, 0), 1)
        
        # Draw eye corners
        for idx in [LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER]:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
