import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2, numpy as np, mediapipe as mp, time, threading, queue
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from tensorflow.keras.models       import load_model

ACTIONS         = np.array(['hello', 'thanks', 'yes', 'no'])
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.50
MP_MODEL        = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ── Load model & detector ──────────────────────────────────
print("Loading model...")
model = load_model('action.h5')
print("✅ Model loaded")

print("Loading MediaPipe...")
detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL),
    running_mode=RunningMode.IMAGE, num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
))
print("✅ MediaPipe loaded")

# ── TTS in background thread — never blocks the camera loop ─
tts_queue = queue.Queue()

def tts_worker():
    import pyttsx3
    engine = pyttsx3.init()
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
print("✅ TTS thread started")

def speak(text):
    tts_queue.put(text)

# ── Open camera ────────────────────────────────────────────
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No camera found.")
        exit()

# ── Warm up ────────────────────────────────────────────────
print("Warming up camera (3 seconds)...")
start = time.time()
while time.time() - start < 3.0:
    cap.read()
    time.sleep(0.05)

ret, test_frame = cap.read()
if not ret or test_frame is None or test_frame.mean() < 5:
    print("❌ Camera returning black frames — close other apps using the camera.")
    cap.release()
    exit()
print(f"✅ Camera ready! Brightness: {test_frame.mean():.1f}")

cv2.namedWindow("Sign to Speech — press Q to quit", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign to Speech — press Q to quit", 640, 480)
print("✅ Starting detection... Press Q to quit.\n")

sequence, sentence = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    # ── Hand detection ─────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    keypoints   = np.zeros(63)
    hand_detect = False
    if result.hand_landmarks:
        hand_detect = True
        lms = result.hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
        h, w = frame.shape[:2]
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
        for a,b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0,200,255), 2)
        for x,y in pts:
            cv2.circle(frame, (x,y), 5, (0,255,0), -1)

    # ── Buffer & prediction ────────────────────────────────
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]

    pred, conf = "---", 0.0
    if len(sequence) == SEQUENCE_LENGTH:
        res  = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        idx  = int(np.argmax(res))
        conf = float(res[idx])
        pred = ACTIONS[idx]

        if conf > THRESHOLD:
            if not sentence or pred != sentence[-1]:
                sentence.append(pred)
                speak(pred)

    # ── Display overlay ────────────────────────────────────
    # Top bar — sentence output
    cv2.rectangle(frame, (0, 0), (640, 45), (0, 0, 0), -1)
    cv2.putText(frame, ' '.join(sentence[-5:]) or '(show a gesture...)',
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Bottom debug bar
    cv2.rectangle(frame, (0, 420), (640, 480), (30, 30, 30), -1)

    hand_color = (0, 255, 0) if hand_detect else (0, 0, 255)
    cv2.putText(frame, "Hand: YES" if hand_detect else "Hand: NO",
                (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    buf_color = (0, 255, 0) if len(sequence) == SEQUENCE_LENGTH else (0, 165, 255)
    cv2.putText(frame, f"Buf: {len(sequence)}/{SEQUENCE_LENGTH}",
                (160, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, buf_color, 2)

    conf_color = (0, 255, 0) if conf > THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"Pred: {pred} ({conf:.2f})",
                (290, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    cv2.putText(frame, f"Thr:{THRESHOLD}",
                (520, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Sign to Speech — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tts_queue.put(None)
cap.release()
cv2.destroyAllWindows()
print("Done.")
