import cv2
import time

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
else:
    print("Warming up camera...")
    for _ in range(30):
        cap.read()
        time.sleep(0.03)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite("webcam_test.jpg", frame)
        print("Success. Saved snapshot to webcam_test.jpg")
    else:
        print("Webcam opened but couldn't read a frame.")
    cap.release()
