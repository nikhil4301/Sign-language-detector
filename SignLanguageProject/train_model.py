import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

DATA_PATH       = "data"
ACTIONS         = np.array(['hello', 'thanks', 'yes', 'no'])
NO_SEQUENCES    = 30
SEQUENCE_LENGTH = 30
LOG_DIR         = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
print("Loading data...")
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(ACTIONS)}

for action in ACTIONS:
    for seq in range(NO_SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            path = os.path.join(DATA_PATH, action, str(seq), f'{frame_num}.npy')
            if os.path.exists(path):
                window.append(np.load(path))
            else:
                window.append(np.zeros(63))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

# ── Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── Build LSTM model ───────────────────────────────────────
model = Sequential([
    LSTM(64,  return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64,  return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# ── Callbacks ──────────────────────────────────────────────
tb_callback = TensorBoard(log_dir=LOG_DIR)
cp_callback = ModelCheckpoint('action.h5', save_best_only=True, monitor='val_loss', verbose=1)

# ── Train ──────────────────────────────────────────────────
print("\nTraining model...")
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, cp_callback]
)

# ── Evaluate ───────────────────────────────────────────────
print("\nEvaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

print("\n✅ Model saved to action.h5")
print("👉 Now run: python main.py")
