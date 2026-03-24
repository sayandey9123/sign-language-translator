import mediapipe as mp
import cv2
import numpy as np
import os
import csv
from pathlib import Path

print("Starting landmark extraction...")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# Paths
DATASET_PATH = "data/dataset/asl_alphabet_train/asl_alphabet_train"
OUTPUT_PATH = "data/processed"
OUTPUT_FILE = "data/processed/landmarks.csv"

# Create output folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get all class labels (A-Z, del, nothing, space)
labels = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(labels)} classes: {labels}")

# Create CSV file
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    # Header row — 63 features (21 landmarks x 3 coordinates)
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

    total = 0
    skipped = 0

    # Loop through each label folder
    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        images = os.listdir(label_path)
        print(f"Processing {label}: {len(images)} images...")

        count = 0
        for img_name in images:
            img_path = os.path.join(label_path, img_name)

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract landmarks
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]

                # Get all 21 landmark coordinates
                row = []
                for lm in landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]

                # Add label
                row.append(label)

                # Write to CSV
                writer.writerow(row)
                count += 1
                total += 1

            else:
                skipped += 1

        print(f"  ✓ {label}: saved {count} landmarks")

print(f"\n✅ Done! Total saved: {total} rows")
print(f"⚠️  Skipped (no hand detected): {skipped} images")
print(f"📄 Saved to: {OUTPUT_FILE}")

hands.close()