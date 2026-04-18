import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2, numpy as np, mediapipe as mp, time
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

DATA_PATH       = "data"
ACTIONS         = ['hello', 'thanks', 'yes', 'no']
NO_SEQUENCES    = 30
SEQUENCE_LENGTH = 30
MP_MODEL        = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ── Create folders ─────────────────────────────────────────
for action in ACTIONS:
    for seq in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)
print(f"✅ Folders ready at: {os.path.abspath(DATA_PATH)}")

# ── Load MediaPipe detector ────────────────────────────────
detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL),
    running_mode=RunningMode.IMAGE, num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
))

def extract_keypoints(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    keypoints = np.zeros(63)
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
        h, w = frame.shape[:2]
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
        for a,b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0,200,255), 2)
        for x,y in pts:
            cv2.circle(frame, (x,y), 5, (0,255,0), -1)
    return keypoints

# ── Open camera ────────────────────────────────────────────
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No camera found. Check your webcam connection.")
        exit()

# ── Warm up camera ─────────────────────────────────────────
print("Warming up camera (3 seconds)...")
start = time.time()
while time.time() - start < 3.0:
    cap.read()
    time.sleep(0.05)

ret, test_frame = cap.read()
if not ret or test_frame is None or test_frame.mean() < 5:
    print("❌ Camera returning black frames — close all other apps using the camera.")
    cap.release()
    exit()
print(f"✅ Camera ready! Brightness: {test_frame.mean():.1f}")

cv2.namedWindow('Collecting Data', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Collecting Data', 640, 480)

# ── Collection loop ────────────────────────────────────────
total_saved = 0

for action in ACTIONS:
    for seq in range(NO_SEQUENCES):

        # Smooth countdown — keeps camera feed alive
        for countdown in range(3, 0, -1):
            deadline = time.time() + 1.0
            while time.time() < deadline:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0,0), (640,80), (0,0,0), -1)
                cv2.putText(frame, f'GET READY: [{action.upper()}]',
                            (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.putText(frame, f'Seq {seq+1}/{NO_SEQUENCES}  —  Starting in {countdown}',
                            (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow('Collecting Data', frame)
                cv2.waitKey(30)

        # Collect frames
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"  ⚠️  Frame drop at {action}/seq{seq}/frame{frame_num} — saving zeros")
                keypoints = np.zeros(63)
            else:
                frame = cv2.flip(frame, 1)
                keypoints = extract_keypoints(frame)
                cv2.rectangle(frame, (0,0), (640,60), (0,0,0), -1)
                cv2.putText(frame, f'[{action.upper()}]  Seq {seq+1}/{NO_SEQUENCES}',
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f'Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                            (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.imshow('Collecting Data', frame)
                cv2.waitKey(1)

            save_path = os.path.join(DATA_PATH, action, str(seq), str(frame_num))
            np.save(save_path, keypoints)
            total_saved += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\nStopped early. Saved {total_saved} frames so far.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f'  ✅ [{action}] seq {seq+1}/{NO_SEQUENCES} saved  (total frames: {total_saved})')

cap.release()
cv2.destroyAllWindows()

# ── Verify saved files ─────────────────────────────────────
print(f'\n🎉 Collection complete! Total frames saved: {total_saved}')
print('\nVerifying saved data...')
for action in ACTIONS:
    count = 0
    for seq in range(NO_SEQUENCES):
        for f in range(SEQUENCE_LENGTH):
            p = os.path.join(DATA_PATH, action, str(seq), f'{f}.npy')
            if os.path.exists(p):
                count += 1
    print(f'  [{action}]: {count}/{NO_SEQUENCES * SEQUENCE_LENGTH} frames saved')

print('\n👉 Now run: python train_model.py')
