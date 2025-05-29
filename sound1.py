import os
import sys
import argparse
import glob
import time
import threading
import pyttsx3

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True, help='Image source: file, folder, video file, or USB camera index (e.g., "usb0")')
parser.add_argument('--thresh', default=0.5, type=float, help='Minimum confidence threshold (e.g., 0.4)')
parser.add_argument('--resolution', default=None, help='Resolution WxH (e.g., "640x480")')
parser.add_argument('--record', action='store_true', help='Record results from video or webcam')
args = parser.parse_args()

# Parse inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model exists
if not os.path.exists(model_path):
    print('ERROR: Model not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Check image source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid input source: {img_source}')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Setup recording if needed
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supported for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Specify resolution for recording.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Bounding box colors (Tableau 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241),
               (98,118,150), (172,176,184)]

# Audio setup
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

last_spoken = {}
cooldown = 2  # seconds

# Variables for FPS calculation
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Main loop
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Video ended.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Camera disconnected or not working.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)

    # Resize if specified
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    current_time = time.time()

    # Process detections
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(det.cls.item())
        classname = labels[classidx]
        conf = det.conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            object_count += 1

            # Audio feedback with cooldown
            if classname not in last_spoken or current_time - last_spoken[classname] > cooldown:
                threading.Thread(target=speak, args=(f'{classname} detected',)).start()
                last_spoken[classname] = current_time

    # Display FPS and object count
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects detected: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Show frame
    cv2.imshow('YOLO detection results', frame)

    if record:
        recorder.write(frame)

    # Key controls
    key = cv2.waitKey(1 if source_type in ['video', 'usb', 'picamera'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey(0)
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    # FPS calculation
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
