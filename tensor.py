import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
import time
import argparse
import threading
import os
from pathlib import Path
import tflite_runtime.interpreter as tflite

# Initialize Firebase app
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://inspectorbill-39299-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# === Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help="Path to custom TFLite model (e.g., model.tflite)")
parser.add_argument("--thresh", default=0.5, help="Minimum confidence threshold")
parser.add_argument("--res", default="416x416", help="Resolution WxH (e.g. 416x416)")
args = parser.parse_args()

# === Configs ===
min_thresh = float(args.thresh)
resW, resH = map(int, args.res.split("x"))

# === TTS using espeak ===
def speak(text):
    try:
        os.system(f'espeak "{text}"')
    except:
        pass

# === Load TFLite Model ===
model_path = args.weights
if not Path(model_path).exists():
    print(f"[ERROR] Model not found: {model_path}")
    exit(1)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assume model input shape: [1, height, width, 3]
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# === Open Camera ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# === Random colors for bounding boxes ===
bbox_colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(100)]

# === Function to preprocess image for TFLite ===
def preprocess(image):
    # Resize and normalize image
    img = cv2.resize(image, (input_width, input_height))
    img = img.astype(np.float32)
    img = img / 255.0  # Normalize if model expects [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# === Placeholder parse_output function ===
def parse_output(output_data, conf_threshold=min_thresh):
    # TODO: Customize this function after checking output tensors shapes and data
    print("[WARN] parse_output is a placeholder. Customize it based on your model output!")
    return []

# === Load labels file if exists ===
labels_path = "labels.txt"
if Path(labels_path).exists():
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
else:
    labels = [f"class_{i}" for i in range(100)]

# === Detection Loop ===
print("[INFO] Starting detection with TFLite model...")

last_spoken_time = 0
speak_interval = 5  # Seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible.")
        break

    # Preprocess frame for TFLite model
    input_data = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Collect output data from all output tensors
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]

    # === DEBUG: Print output tensor shapes and sample data ===
    for i, data in enumerate(output_data):
        print(f"Output tensor {i} shape: {data.shape}")
        print(f"Output tensor {i} sample data (first 10 entries): {data.flatten()[:10]}")

    detected_objects = []

    # Parse model outputs for detected boxes and classes
    detections = parse_output(output_data, conf_threshold=min_thresh)

    for detection in detections:
        xmin, ymin, xmax, ymax = detection["box"]
        class_id = detection["class_id"]

        # Convert normalized coordinates to pixel values on original frame size
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])

        # Get class name safely
        if class_id < len(labels):
            class_name = labels[class_id]
        else:
            class_name = f"class_{class_id}"

        detected_objects.append(class_name)

        # Draw bounding box and label
        color = bbox_colors[class_id % len(bbox_colors)]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(frame, class_name, (xmin, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Speak every X seconds only
    current_time = time.time()
    if detected_objects and (current_time - last_spoken_time) > speak_interval:
        last_spoken_time = current_time
        description = ', '.join(set(detected_objects))

        # === Send to Firebase ===
        try:
            db.reference('detections').push({
                'timestamp': time.time(),
                'objects': list(set(detected_objects))
            })
        except Exception as e:
            print(f"[FIREBASE ERROR] {e}")

        threading.Thread(target=speak, args=(f"I see {description}",), daemon=True).start()

    # Show frame
    cv2.imshow("TFLite Detection (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.01)  # Light FPS limit for Pi

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
