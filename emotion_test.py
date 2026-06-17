import cv2
import time
from collections import deque, Counter
from deepface import DeepFace
from akasha_db import init_db, log_event

init_db()

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Warming up camera...")
for _ in range(30):
    cap.read()
    time.sleep(0.03)

WINDOW_SIZE = 5
recent_readings = deque(maxlen=WINDOW_SIZE)
last_logged = None

print("Detecting emotion. Press Ctrl+C to stop.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            dominant = result['dominant_emotion']
            recent_readings.append(dominant)

            if len(recent_readings) == WINDOW_SIZE:
                settled = Counter(recent_readings).most_common(1)[0][0]
                print(f"Raw: {dominant:10} | Settled (last {WINDOW_SIZE}): {settled}")

                if settled != last_logged:
                    log_event("self", "emotion", settled)
                    print(f"  -> Logged change to: {settled}")
                    last_logged = settled
            else:
                print(f"Raw: {dominant} (warming up smoothing window, {len(recent_readings)}/{WINDOW_SIZE})")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopped.")
    cap.release()
