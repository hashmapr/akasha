"""
Run this once (re-run any time lighting/setup changes meaningfully).
Sit at a normal resting expression — not posed, not smiling, not
deliberately neutral-acting, just however you actually look when
not performing an emotion — for the full capture window.

This does NOT assume your resting face IS neutral. It just measures
what DeepFace's emotion model outputs for whatever your resting face
actually is, so that baseline can be subtracted out later. If your
resting face genuinely reads partially as "sad" to a human too,
that's a different problem this script can't fix — it only corrects
for model bias, not for ground truth you'd dispute yourself.
"""

import cv2
import time
import json
from deepface import DeepFace

CAPTURE_SECONDS = 15
OUTPUT_FILE = "emotion_baseline.json"

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Warming up camera...")
for _ in range(30):
    cap.read()
    time.sleep(0.03)

print(f"\nSit at your normal resting expression for {CAPTURE_SECONDS} seconds.")
print("Don't pose, don't perform 'neutral' — just sit how you normally sit.")
input("Press Enter when ready...")

readings = []
start = time.time()

while time.time() - start < CAPTURE_SECONDS:
    ret, frame = cap.read()
    if not ret:
        continue
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        readings.append(result['emotion'])  # dict of emotion -> 0-100 score
        print(".", end="", flush=True)
    except Exception as e:
        print(f"\nError on frame: {e}")
    time.sleep(0.3)

cap.release()
print(f"\n\nCaptured {len(readings)} readings.")

if not readings:
    print("No readings captured. Try again — check lighting/webcam.")
    exit()

emotions = readings[0].keys()
baseline = {
    emotion: float(sum(r[emotion] for r in readings) / len(readings))
    for emotion in emotions
}

print("\nBaseline (average score per emotion at rest):")
for emotion, score in sorted(baseline.items(), key=lambda x: -x[1]):
    print(f"  {emotion:10} {score:5.1f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(baseline, f, indent=2)

print(f"\nSaved to {OUTPUT_FILE}.")
print("This will be subtracted from live readings to correct for model bias.")