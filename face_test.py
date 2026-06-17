import cv2
import time

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Warming up camera...")
for _ in range(30):
    cap.read()
    time.sleep(0.03)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Looking for a face. Press Ctrl+C to stop.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            print(f"Face detected ({len(faces)} found).")
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("face_detected.jpg", frame)
        else:
            print("No face detected.")

        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
    cap.release()
