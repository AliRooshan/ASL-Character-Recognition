import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import random
import joblib
from tqdm import tqdm

DATA_DIR = Path("data/raw")
SAVE_DIR = Path("data/landmarks")
IMAGES_PER_CLASS = 500
SEED = 42

random.seed(SEED)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

X = []
y = []
label_map = {}

classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

for label_idx, cls in enumerate(classes):

    label_map[label_idx] = cls
    print(f"\nProcessing {cls}")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(list((DATA_DIR / cls).glob(ext)))

    selected = random.sample(
        images,
        min(IMAGES_PER_CLASS, len(images))
    )

    for img_path in tqdm(selected):

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand = results.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Translation normalization
        wrist = landmarks[0]
        landmarks = landmarks - wrist

        # Scale normalization
        scale = np.max(np.linalg.norm(landmarks, axis=1))
        if scale > 0:
            landmarks = landmarks / scale

        features = landmarks.flatten()

        X.append(features)
        y.append(label_idx)

hands.close()

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

np.save(SAVE_DIR / "X.npy", X)
np.save(SAVE_DIR / "y.npy", y)
joblib.dump(label_map, SAVE_DIR / "label_encoder.pkl")

print("\nDone")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", len(label_map))