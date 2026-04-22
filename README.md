# Real-Time Crime Detection using Deep Learning

This project detects criminal or anomalous activities in videos and real-time camera feeds. It uses **EfficientNetB0** for spatial feature extraction on individual video frames and a **GRU (Gated Recurrent Unit)** network to capture the temporal evolution of these features over time (sequence modeling).

## 🚀 Features
- **Frame Extraction:** Automatically extracts frames from video datasets.
- **Deep Feature Extraction:** Uses pre-trained `EfficientNetB0` to convert images into rich feature vectors.
- **Temporal Analysis:** Uses a GRU model for robust sequence classification (`crime` vs `normal`).
- **Real-Time Prediction:** Integrates with your webcam to classify activities dynamically in real time.

---

## 🛠️ Project Structure
- `extract_all_frames.py` : Script to extract frames from standard video datasets (`dataset/crime/` and `dataset/normal/`).
- `feature_extraction.py` : Passes extracted frames through EfficientNetB0 to create sequence feature arrays (`features.npy`, `features_labels.npy`).
- `train_model.py` : Trains the GRU model on extracted sequence features and saves it to a `.keras` file.
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

4. **Train the GRU sequence model:**
   Train the recurrent network with the extracted feature representations:
   ```bash
   python train_model.py
   ```
   *Upon completion, a trained model file (`crime_detection_model.keras`) will be generated.*

---

## 🎥 How to Use for Real-Time Prediction

Once the model is trained (or downloaded), you can run the live webcam prediction script.
It will continually read sequences of 30 frames, extract their features using EfficientNetB0, and evaluate them via the GRU model.

1. Ensure the saved model (`crime_detection_model.keras`) is in the main directory.
2. Run the real-time prediction script:
   ```bash
   python predict.py
   ```
3. A live camera window will open, showcasing real-time FPS and whether the activity is **"CRIME"** (Red) or **"NORMAL"** (Green).
4. Press `q` to terminate the camera feed.

---

## 🔗 Pre-trained Model Link

To avoid retraining everything from scratch, you can download the fully trained Keras model files via Google Drive and place them in the root of the project directory.

[**Download Pre-Trained Model (Google Drive)**](https://drive.google.com/drive/folders/163xiJJRDZWzpEdeBn7OLGQCisjkC0j97?usp=sharing)
