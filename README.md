# Real-Time Crime Detection using Deep Learning

This project detects criminal or anomalous activities in videos and real-time camera feeds. It uses **EfficientNetB0** for spatial feature extraction on individual video frames and sequence modeling networks (**LSTM, BiLSTM, or GRU**) to capture the temporal evolution of these features over time.

## 🚀 Features
- **Frame Extraction:** Automatically extracts frames from video datasets.
- **Deep Feature Extraction:** Uses pre-trained `EfficientNetB0` to convert images into rich feature vectors.
- **Temporal Analysis:** Supports using **LSTM, BiLSTM, or GRU** sequence modeling for robust classification (`crime` vs `normal`).
- **Real-Time Prediction:** Integrates with your webcam to classify activities dynamically in real time.

---

## 🛠️ Project Structure
- `extract_all_frames.py` : Script to extract frames from standard video datasets (`dataset/crime/` and `dataset/normal/`).
- `feature_extraction.py` : Passes extracted frames through EfficientNetB0 to create sequence feature arrays (`features.npy`, `features_labels.npy`).
- `train_lstm.py`, `train_bilstm.py`, `train_gru.py` : Scripts to train either an LSTM, BiLSTM, or GRU network on the extracted features. Each saves its respective `.keras` file.
- `predict.py` : Uses a connected camera (or video feed) to predict crime vs. normal events in real time.
- `utils/video_utils.py` : Helper functions for operations like extracting frames.

---

## 💻 Setup Instructions

1. **Clone the repository and navigate to the project directory.**
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/MacOS
   venv\Scripts\activate     # On Windows
   ```
3. **Install the dependencies:**
   Make sure you have libraries like TensorFlow, OpenCV, NumPy, and Scikit-Learn installed.
   ```bash
   pip install tensorflow opencv-python numpy scikit-learn tqdm
   ```

---

## 🧠 How to Train the Model

1. **Prepare your dataset:**
   Place your raw video files in the `dataset` directory structured by class:
   ```text
   dataset/
   ├── crime/       # Put crime videos here
   └── normal/      # Put normal videos here
   ```

2. **Extract Frames:**
   Run the frame extraction script to convert videos into images:
   ```bash
   python extract_all_frames.py
   ```
   *This will populate the `frame_data/` folder.*

3. **Extract Features:**
   Process the images using EfficientNetB0 to yield numpy array files (`features.npy` and `features_labels.npy`):
   ```bash
   python feature_extraction.py
   ```

4. **Train the Sequence Model:**
   You can choose between an LSTM, BiLSTM, or GRU model based on your performance preference:
   ```bash
   python train_lstm.py     # Trains and saves crime_detection_model_lstm.keras
   python train_bilstm.py   # Trains and saves crime_detection_model_bilstm.keras
   python train_gru.py      # Trains and saves crime_detection_model_gru.keras
   ```

---

## 🎥 How to Use for Real-Time Prediction

Once the models are trained (or downloaded), you can run the live webcam prediction script.
It will continually read sequences of 30 frames, extract their features using EfficientNetB0, and evaluate them via the chosen temporal model (by default, configured to load the GRU model in `predict.py`).

1. Ensure your chosen trained model (e.g. `crime_detection_model_gru.keras`) is in the main directory and specified correctly in `predict.py`.
2. Run the real-time prediction script:
   ```bash
   python predict.py
   ```
3. A live camera window will open, showcasing real-time FPS and whether the activity is **"CRIME"** (Red) or **"NORMAL"** (Green).
4. Press `q` to terminate the camera feed.

---

## 🔗 Pre-trained Model Link

To avoid retraining everything from scratch, three variations of the fully trained models (LSTM, BiLSTM, and GRU) are available via Google Drive. You can download any of them and use them directly without retraining:

[**Download Pre-Trained Model (Google Drive)**](https://drive.google.com/drive/folders/163xiJJRDZWzpEdeBn7OLGQCisjkC0j97?usp=sharing)
