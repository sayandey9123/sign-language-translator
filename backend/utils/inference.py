import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import cv2
import base64
import os

# ── BASE DIR ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── LOAD MODEL & ENCODER ──
print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models/sign_model.h5")
)

with open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

print("✅ Model loaded!")

# ── MEDIAPIPE SETUP ──
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def decode_frame(base64_frame):
    img_data = base64.b64decode(base64_frame.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        row = []
        for lm in landmarks.landmark:
            row += [lm.x, lm.y, lm.z]
        return np.array(row), result.multi_hand_landmarks[0]
    return None, None


def normalize_landmarks(landmarks):
    X = landmarks.reshape(1, -1)
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
    return X_normalized


def predict_sign(base64_frame):
    try:
        frame = decode_frame(base64_frame)
        landmarks, hand_landmarks = extract_landmarks(frame)

        if landmarks is None:
            return {
                "letter": None,
                "confidence": 0,
                "top3": [],
                "hand_detected": False
            }

        X = normalize_landmarks(landmarks)
        predictions = model.predict(X, verbose=0)[0]

        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3 = [
            {
                "letter": encoder.classes_[i],
                "confidence": float(predictions[i])
            }
            for i in top3_idx
        ]

        best_idx = top3_idx[0]
        best_letter = encoder.classes_[best_idx]
        best_confidence = float(predictions[best_idx])

        return {
            "letter": best_letter,
            "confidence": best_confidence,
            "top3": top3,
            "hand_detected": True
        }

    except Exception as e:
        return {
            "letter": None,
            "confidence": 0,
            "top3": [],
            "hand_detected": False,
            "error": str(e)
        }