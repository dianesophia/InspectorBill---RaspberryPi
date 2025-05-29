import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
from pathlib import Path
import threading

# Initialize Firebase app
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://inspectorbill-39299-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# === Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help=" Strategically choose a lightweight model (e.g., yolov8n.pt)")
parser.add_argument("--thresh", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--res", default="320x320", help="Resolution WxH (e.g., 320x320)")
args = parser.parse_args()

# === Configs ===
min_thresh = args.thresh
resW, resH = map(int, args.res.split("x"))

# === Load Model ===
model_path = args.weights
if not Path(model_path).exists():
    print(f"[ERROR] Model not found: {model_path}")
    exit(1)

model = YOLO(model_path)

# === Open Camera ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce load

# === Random colors for bounding boxes ===
bbox_colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(100)]

# === Detection Loop ===
print("[INFO] Starting detection with optimized YOLOv8 model...")

last_firebase_time = 0
firebase_interval = 10  # Seconds between Firebase updates
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to reduce load

    # Resize frame
    frame_resized = cv2.resize(frame, (resW, resH), interpolation=cv2.INTER_AREA)

    # Run inference
    results = model.predict(source=frame_resized, conf=min_thresh, verbose=False, device='cpu')

    detected_objects = []

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = map(int, boxes[i])
                class_id = classes[i]

                # Handle unknown labels
                try:
                    class_name = model.names[class_id]
                except KeyError:
                    class_name = f"class_{class_id}"

                detected_objects.append(class_name)

    # Update Firebase periodically
    current_time = time.time()
    if detected_objects and (current_time - last_firebase_time) > firebase_interval:
        last_firebase_time = current_time
        try:
            db.reference('detections').push({
                'timestamp': current_time,
                'objects': list(set(detected_objects))
            })
        except Exception as e:
            print(f"[FIREBASE ERROR] {e}")

    # Sleep to control loop speed
    time.sleep(0.05)  # ~20 FPS max

# === Cleanup ===
cap.release()
