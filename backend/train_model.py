import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("Loading dataset...")

# ── LOAD DATA ──
df = pd.read_csv("data/processed/landmarks.csv")
print(f"Total samples: {len(df)}")
print(f"Classes: {df['label'].unique()}")

# ── SPLIT FEATURES AND LABELS ──
X = df.drop("label", axis=1).values
y = df["label"].values

# ── ENCODE LABELS ──
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {encoder.classes_}")

# ── NORMALIZE FEATURES ──
# Normalize each sample relative to itself
# This makes model work for any hand size or position
X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X_normalized = (X - X_min) / (X_max - X_min + 1e-8)

# ── TRAIN/VAL/TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── BUILD MODEL ──
print("\nBuilding model...")

model = Sequential([
    # Input layer
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden layer 1
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Hidden layer 2
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Output layer
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── CALLBACKS ──
os.makedirs("backend/models", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "backend/models/sign_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── TRAIN ──
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ── EVALUATE ──
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ── SAVE LABEL ENCODER ──
with open("backend/models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("✅ Label encoder saved!")

# ── PLOT ACCURACY ──
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("backend/models/training_history.png")
print("✅ Training plot saved!")

# ── CONFUSION MATRIX ──
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

plt.figure(figsize=(15, 12))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cmap='Blues'
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("backend/models/confusion_matrix.png")
print("✅ Confusion matrix saved!")

print("\n🎉 Training complete!")
print(f"📁 Model saved to: backend/models/sign_model.h5")
print(f"📁 Encoder saved to: backend/models/label_encoder.pkl")
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")