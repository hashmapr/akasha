import cv2
import time
from ultralytics import YOLO
from akasha_db import init_db, log_event

init_db()

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

DRINK_OBJECTS = {"bottle", "cup", "wine glass"}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Warming up camera...")
for _ in range(30):
    cap.read()
    time.sleep(0.03)

print("Watching for drinking events. Press Ctrl+C to stop.\n")

COOLDOWN_SECONDS = 5
last_event_time = 0
currently_near = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        results = model(frame, verbose=False)
        detected_objects = []
        for r in results:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name in DRINK_OBJECTS:
                    x1, y1, x2, y2 = box.xyxy[0]
                    detected_objects.append((cls_name, float(x1), float(y1), float(x2), float(y2)))

        near_face = False
        detected_label = None
        if len(faces) > 0 and detected_objects:
            fx, fy, fw, fh = faces[0]
            face_cx, face_cy = fx + fw / 2, fy + fh / 2
            margin = fw * 1.5

            for cls_name, x1, y1, x2, y2 in detected_objects:
                obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) / 2
                if abs(obj_cx - face_cx) < margin and abs(obj_cy - face_cy) < margin:
                    near_face = True
                    detected_label = cls_name
                    break

        now = time.time()
        if near_face and not currently_near and (now - last_event_time) > COOLDOWN_SECONDS:
            log_event("body", "drink_event", f"{detected_label} near face detected")
            print(f"Drink event logged ({detected_label}) at {time.strftime('%H:%M:%S')}")
            last_event_time = now
        elif near_face:
            print(f"{detected_label} near face (cooldown active)")
        else:
            print("No drink event.")

        currently_near = near_face
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")
    cap.release()
