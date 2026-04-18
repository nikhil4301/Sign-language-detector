# 🖐️ Sign Language Recognition System

A real-time sign language recognition system that detects hand gestures from a webcam and converts them into spoken words using deep learning and computer vision.

## 📽️ Demo

| Data Collection | Live Detection |
|---|---|
| Hand landmarks tracked in real-time | Gesture recognized with 1.00 confidence |

---

## 🧠 How It Works

```
Webcam → MediaPipe (21 hand landmarks) → Buffer (30 frames) → LSTM Model → Speech Output
```

1. **MediaPipe** detects 21 keypoints on the hand per frame (63 values: x, y, z each)
2. A **sliding buffer** collects the last 30 frames
3. The **LSTM model** classifies the gesture from the 30-frame sequence
4. **pyttsx3** speaks the recognized word aloud via a background thread

---

## ✋ Supported Gestures

| Gesture | Word |
|---|---|
| 👋 | Hello |
| 🙏 | Thanks |
| ✅ | Yes |
| ❌ | No |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Main language |
| OpenCV | Camera capture & display |
| MediaPipe | Hand landmark detection |
| TensorFlow / Keras | LSTM model training & inference |
| NumPy | Data storage and processing |
| pyttsx3 | Offline text-to-speech |
| Threading | Non-blocking speech output |

---

## 📁 Project Structure

```
SignLanguageProject/
│
├── collect_data.py       # Step 1: Collect gesture training data
├── train_model.py        # Step 2: Train the LSTM model
├── main.py               # Step 3: Run real-time detection
│
├── hand_landmarker.task  # MediaPipe model (download separately)
├── action.h5             # Trained model (generated after training)
│
├── data/                 # Collected gesture data (generated after collection)
│   ├── hello/
│   ├── thanks/
│   ├── yes/
│   └── no/
│
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/SignLanguageProject.git
cd SignLanguageProject
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe Hand Landmarker model
Download `hand_landmarker.task` from the official MediaPipe page and place it in the project root:

🔗 https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

---

## 🚀 How to Run

### Step 1 — Collect Training Data
```bash
python collect_data.py
```
- A window will open showing your webcam
- Perform each gesture (hello, thanks, yes, no) when prompted
- 30 sequences × 30 frames are recorded per gesture
- Press **Q** to quit early

### Step 2 — Train the Model
```bash
python train_model.py
```
- Loads collected data from the `data/` folder
- Trains the LSTM model for 100 epochs
- Saves the best model as `action.h5`

### Step 3 — Run Live Detection
```bash
python main.py
```
- Opens the webcam
- Shows real-time hand landmark overlay
- Displays recognized gesture and speaks it aloud
- Press **Q** to quit

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| Black camera window | Close all other apps using the camera (Teams, Zoom, browser) |
| Camera not opening | Make sure no other app has the camera. Check Windows Camera Privacy settings |
| Low prediction confidence | Lower `THRESHOLD` in `main.py` from 0.80 to 0.50 |
| TTS freezing the video | Already fixed — pyttsx3 runs in a background thread |
| `hand_landmarker.task` not found | Download from MediaPipe website and place in project root |

---

## 📊 Model Architecture

```
Input: (30, 63)
  ↓
LSTM(64,  return_sequences=True)
  ↓
LSTM(128, return_sequences=True)
  ↓
LSTM(64,  return_sequences=False)
  ↓
Dense(64)
  ↓
Dense(32)
  ↓
Dense(4, softmax)  ← one output per gesture
```

---

## 🔮 Future Improvements

- [ ] Expand to full ISL/ASL alphabet
- [ ] Two-hand gesture support
- [ ] Multi-user generalization via transfer learning
- [ ] Mobile deployment with TensorFlow Lite
- [ ] Bidirectional system (speech → sign animation)

---

## 👤 Author

**Nikhil**
B.E. Computer Science and Engineering
Chandigarh University

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
