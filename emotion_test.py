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

# Tunable — calibrate against the printed raw confidence values below
# before trusting these numbers. 50 is a starting guess, not a result.
CONFIDENCE_THRESHOLD = 50  # DeepFace returns 0-100 scale per emotion
WINDOW_SIZE = 8            # widened from 5 — only matters once low-confidence noise is filtered
MIN_AGREEMENT = 6          # require strong majority within the window, not bare plurality

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
            confidence = result['emotion'][dominant]  # 0-100

            print(f"Raw: {dominant:10} conf={confidence:5.1f}")  # leave in during calibration

            if confidence < CONFIDENCE_THRESHOLD:
                time.sleep(2)
                continue

            recent_readings.append(dominant)

            if len(recent_readings) == WINDOW_SIZE:
                settled, count = Counter(recent_readings).most_common(1)[0]
                print(f"  Settled (last {WINDOW_SIZE}): {settled} ({count}/{WINDOW_SIZE} agreement)")

                if count >= MIN_AGREEMENT and settled != last_logged:
                    log_event("self", "emotion", settled)
                    print(f"  -> Logged change to: {settled}")
                    last_logged = settled
            else:
                print(f"  (warming up smoothing window, {len(recent_readings)}/{WINDOW_SIZE})")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopped.")
    cap.release()