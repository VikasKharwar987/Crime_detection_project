import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from collections import deque

model = load_model('crime_detection_model.keras')

base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    feature = feature_extractor.predict(x, verbose=0)[0]
    return feature

def detect_from_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FPS, 50)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")
    frame_buffer = deque(maxlen=30)
    fps_times = []
    last_pred_time = time.time()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        feature = extract_features_from_frame(frame)
        frame_buffer.append(feature)

        if len(frame_buffer) == 30:
            input_seq = np.expand_dims(np.array(frame_buffer), axis=0)
            prediction = model.predict(input_seq, verbose=0)[0][0]

            color = (0, 255, 0)
            label = f"Normal ({prediction:.2f})"

            if prediction > 0.5:
                color = (0, 0, 255)
                label = f"CRIME ({prediction:.2f})"

            cv2.rectangle(frame, (10, 10), (630, 470), color, 3)
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_times.append(fps)
        avg_fps = np.mean(fps_times[-30:]) 
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Crime Detection (Press 'q' to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    detect_from_camera()
