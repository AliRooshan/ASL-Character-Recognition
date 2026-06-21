import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import warnings
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QImage

warnings.filterwarnings("ignore", category=UserWarning)

class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage, str, bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.running = True
        
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.7
        self.draw_landmarks = True
        self.active_model_name = "svm"
        
        self.models = {}
        self.scaler = None
        self.label_map = None
        
        self.scaler_path = "data/landmarks/scaler.pkl"
        self.label_map_path = "data/landmarks/label_encoder.pkl"
        self.model_paths = {
            "svm": "models/svm.pkl",
            "mlp": "models/mlp.pkl",
            "rf": "models/random_forest.pkl"
        }

    def load_resources(self):
        try:
            if not os.path.exists(self.scaler_path) or not os.path.exists(self.label_map_path):
                raise FileNotFoundError("Scaler or Label encoder files are missing from data/landmarks/.")
            
            self.scaler = joblib.load(self.scaler_path)
            self.label_map = joblib.load(self.label_map_path)
            
            default_path = self.model_paths[self.active_model_name]
            if not os.path.exists(default_path):
                raise FileNotFoundError(f"Default model file missing: {default_path}")
            
            self.models[self.active_model_name] = joblib.load(default_path)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Asset Load Error: {str(e)}")
            return False

    def set_model(self, model_name):
        self.mutex.lock()
        try:
            if model_name in self.model_paths:
                self.active_model_name = model_name
                if model_name not in self.models:
                    path = self.model_paths[model_name]
                    if os.path.exists(path):
                        self.models[model_name] = joblib.load(path)
                    else:
                        raise FileNotFoundError(f"Model file {path} not found.")
        except Exception as e:
            self.error_occurred.emit(f"Error switching model: {str(e)}")
        finally:
            self.mutex.unlock()

    def set_min_detection_confidence(self, val):
        self.mutex.lock()
        self.min_detection_confidence = val
        self.mutex.unlock()

    def set_min_tracking_confidence(self, val):
        self.mutex.lock()
        self.min_tracking_confidence = val
        self.mutex.unlock()

    def set_draw_landmarks(self, enabled):
        self.mutex.lock()
        self.draw_landmarks = enabled
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()

    def run(self):
        if not self.load_resources():
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_occurred.emit("Cannot open webcam. Verify if another app is using camera index 0.")
            return

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        self.mutex.lock()
        curr_det_conf = self.min_detection_confidence
        curr_track_conf = self.min_tracking_confidence
        self.mutex.unlock()

        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=curr_det_conf,
            min_tracking_confidence=curr_track_conf
        )

        try:
            while True:
                self.mutex.lock()
                if not self.running:
                    self.mutex.unlock()
                    break
                
                if (curr_det_conf != self.min_detection_confidence or 
                    curr_track_conf != self.min_tracking_confidence):
                    curr_det_conf = self.min_detection_confidence
                    curr_track_conf = self.min_tracking_confidence
                    hands.close()
                    hands = mp_hands.Hands(
                        max_num_hands=1,
                        min_detection_confidence=curr_det_conf,
                        min_tracking_confidence=curr_track_conf
                    )
                
                draw_enabled = self.draw_landmarks
                active_model = self.models.get(self.active_model_name, None)
                self.mutex.unlock()

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                prediction = "Waiting..."
                hand_detected = False

                if results.multi_hand_landmarks:
                    hand_detected = True
                    hand = results.multi_hand_landmarks[0]

                    if draw_enabled:
                        mp_draw.draw_landmarks(
                            frame,
                            hand,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(241, 102, 99), thickness=2, circle_radius=4),
                            mp_draw.DrawingSpec(color=(129, 185, 16), thickness=2, circle_radius=2)
                        )

                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])

                    landmarks = np.array(landmarks)
                    wrist = landmarks[0]
                    landmarks = landmarks - wrist

                    scale = np.max(np.linalg.norm(landmarks, axis=1))
                    if scale > 0:
                        landmarks = landmarks / scale

                    features = landmarks.flatten().reshape(1, -1)

                    if self.scaler is not None and active_model is not None and self.label_map is not None:
                        try:
                            features = self.scaler.transform(features)
                            pred = active_model.predict(features)[0]
                            prediction = self.label_map[pred]
                        except Exception:
                            prediction = "Error"

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

                self.frame_ready.emit(q_img, prediction, hand_detected)
                
                time.sleep(0.03)
        finally:
            try:
                hands.close()
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass