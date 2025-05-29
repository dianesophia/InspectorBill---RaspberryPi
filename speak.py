import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
from pathlib import Path

# === Firebase Initialization ===
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://inspectorbill-39299-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# === Command-line Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help="Path to YOLOv8 model (e.g., yolov8n.pt)")
parser.add_argument("--thresh", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--res", default="320x320", help="Resolution WxH (e.g., 320x320)")
args = parser.parse_args()

min_thresh = args.thresh
resW, resH = map(int, args.res.split("x"))

# === Load YOLOv8 Model ===
model_path = args.weights
if not Path(model_path).exists():
    print(f"[ERROR] Model not found: {model_path}")
    exit(1)

model = YOLO(model_path)

# === OpenCV Camera Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("[ERROR] Cannot access camera.")
    exit(1)

bbox_colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(100)]

print("[INFO] Starting detection...")

# === Detection Loop ===
last_firebase_time = 0
firebase_interval = 10  # Seconds between Firebase pushes
frame_skip = 2
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame_resized = cv2.resize(frame, (resW, resH), interpolation=cv2.INTER_AREA)

    # Inference
    start_time = time.time()
    results = model.predict(source=frame_resized, conf=min_thresh, verbose=False, device='cpu')
    print(f"[INFO] Inference took {time.time() - start_time:.2f} seconds")

    detected_objects = []

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = map(int, boxes[i])
                class_id = classes[i]
                class_name = model.names.get(class_id, f"class_{class_id}")

                detected_objects.append(class_name)
                color = bbox_colors[class_id % len(bbox_colors)]
                cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame_resized, class_name, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Firebase Update
    current_time = time.time()
    if detected_objects and (current_time - last_firebase_time) > firebase_interval:
        try:
            db.reference('detections').push({
                'timestamp': current_time,
                'objects': list(set(detected_objects))
            })
            print(f"[FIREBASE] Sent: {list(set(detected_objects))}")
            last_firebase_time = current_time
        except Exception as e:
            print(f"[FIREBASE ERROR] {e}")

    # Show Frame
    cv2.imshow("YOLO Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)  # Helps regulate loop speed (~20 FPS)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
