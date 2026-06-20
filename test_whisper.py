import threading
import time
import re
import cv2
import numpy as np
import sounddevice as sd
from collections import deque, Counter
from faster_whisper import WhisperModel
from deepface import DeepFace
from akasha_db import init_db, log_event

init_db()

RULES = {
    "water": ("body", "hydration", 1.0),
    "hydrate": ("body", "hydration", 1.0),
    "h2o": ("body", "hydration", 1.0),
    "coffee": ("body", "caffeine", 0.95),
    "tea": ("body", "caffeine", 0.97),
    "espresso": ("body", "caffeine", 0.9),
    "beer": ("body", "drinks", 0.85),
    "wine": ("body", "drinks", 0.55),
    "vodka": ("body", "drinks", 0.25),
    "whiskey": ("body", "drinks", 0.25),
    "soda": ("body", "drinks", 0.95),
    "sprite": ("body", "drinks", 0.95),
    "coke": ("body", "drinks", 0.95),
    "pepsi": ("body", "drinks", 0.95),
    "fanta": ("body", "drinks", 0.95),
    "mountain dew": ("body", "drinks", 0.95),
    "gatorade": ("body", "drinks", 1.0),
    "lemonade": ("body", "drinks", 0.95),
    "juice": ("body", "drinks", 0.95),
    "cocktail": ("body", "drinks", 0.45),
    "alcohol": ("body", "drinks", 0.5),
    "drank": ("body", "drinks", 0.8),
    "workout": ("body", "training", None),
    "lifted": ("body", "training", None),
    "ran": ("body", "training", None),
    "gym": ("body", "training", None),
    "exercise": ("body", "training", None),
    "jog": ("body", "training", None),
    "swam": ("body", "training", None),
    "cardio": ("body", "training", None),
    "stretched": ("body", "training", None),
    "vitamin": ("body", "supplement_log", None),
    "supplement": ("body", "supplement_log", None),
    "took my": ("body", "supplement_log", None),
    "multivitamin": ("body", "supplement_log", None),
    "fish oil": ("body", "supplement_log", None),
    "protein": ("body", "supplement_log", None),
    "creatine": ("body", "supplement_log", None),
    "exhausted": ("body", "energy", None),
    "low energy": ("body", "energy", None),
    "high energy": ("body", "energy", None),
    "energized": ("body", "energy", None),
    "drained": ("body", "energy", None),
    "stressed": ("body", "tension", None),
    "tense": ("body", "tension", None),
    "anxious": ("body", "tension", None),
    "overwhelmed": ("body", "tension", None),
    "relaxed": ("body", "tension", None),
}

UNIT_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(ounces?|oz|cups?|ml|milliliters?|liters?|l|glasses?)",
    re.IGNORECASE
)

ML_CONVERSIONS = {
    "ounce": 29.5735, "ounces": 29.5735, "oz": 29.5735,
    "cup": 236.588, "cups": 236.588,
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "liter": 1000.0, "liters": 1000.0, "l": 1000.0,
    "glass": 240.0, "glasses": 240.0,
}

def extract_amount(text):
    match = UNIT_PATTERN.search(text.lower())
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        ml = amount * ML_CONVERSIONS.get(unit, 0)
        return amount, unit, ml
    return None, None, None

def audio_loop():
    SILENCE_THRESHOLD = 0.003
    print("[Audio] Loading Whisper model...")
    model = WhisperModel("base", device="cpu")
    print("[Audio] Ready.")

    while True:
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if np.abs(audio).mean() < SILENCE_THRESHOLD:
            continue

        segments, _ = model.transcribe(audio, language="en", vad_filter=True)
        text = " ".join(segment.text for segment in segments).strip()
        if not text:
            continue

        print(f"[Audio] -> {text}")

        for keyword, (module, feature, hydration_factor) in RULES.items():
            if keyword in text.lower():
                amount, unit, ml = extract_amount(text)
                log_event(module, feature, text, amount, unit, ml, hydration_factor)
                print(f"[Audio] Logged to {module}/{feature}")
                break

# --- Emotion Detector (Social module) ---
# Reads OTHER people's faces during conversations, not the owner's.
# A persisted personal-bias baseline doesn't generalize across different
# people's faces, and persisting one tied to whoever's in frame would be
# exactly the kind of per-person behavioral profiling the project's own
# Perception Calibration design explicitly rules out. So: no file on disk,
# no identity. Each "session" (a face appearing after a gap of no face)
# gets a short silent calibration window, held in memory only, used as a
# relative offset for that session, then thrown away when the face leaves.

FACE_GAP_SECONDS = 20       # no face for this long = next face starts a new session
SESSION_CALIBRATION_READINGS = 8  # silent readings collected before logging starts
CORRECTED_FLOOR = 15        # corrected score must clear this to count as a real signal
WINDOW_SIZE = 8
MIN_AGREEMENT = 6

def vision_loop():
    print("[Vision] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Vision] Could not open webcam.")
        return

    for _ in range(30):
        cap.read()
        time.sleep(0.03)
    print("[Vision] Ready.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    last_face_seen = 0
    session_active = False
    session_calibration_readings = []
    session_baseline = {}

    recent = deque(maxlen=WINDOW_SIZE)
    last_logged = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_present = len(faces) > 0
        now = time.time()

        if not face_present:
            if session_active and (now - last_face_seen) > FACE_GAP_SECONDS:
                # Session ended — discard everything, no trace kept.
                print("[Vision] Face gone >20s — session ended, baseline discarded.")
                session_active = False
                session_calibration_readings = []
                session_baseline = {}
                recent.clear()
                last_logged = None
            time.sleep(1)
            continue

        last_face_seen = now

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            raw_scores = result['emotion']  # dict, 0-100 per emotion
        except Exception as e:
            print(f"[Vision] Error: {e}")
            time.sleep(1)
            continue

        if not session_active:
            # New session — start silent calibration on whoever this face is.
            session_active = True
            session_calibration_readings = []
            print("[Vision] New face detected — starting silent calibration...")

        if len(session_calibration_readings) < SESSION_CALIBRATION_READINGS:
            session_calibration_readings.append(raw_scores)
            print(f"[Vision] Calibrating ({len(session_calibration_readings)}/{SESSION_CALIBRATION_READINGS})...")
            if len(session_calibration_readings) == SESSION_CALIBRATION_READINGS:
                emotions = session_calibration_readings[0].keys()
                session_baseline = {
                    emotion: float(sum(r[emotion] for r in session_calibration_readings) / len(session_calibration_readings))
                    for emotion in emotions
                }
                print(f"[Vision] Session baseline set: {session_baseline}")
            time.sleep(1)
            continue

        # Calibrated — apply relative correction and log on settled change.
        corrected = {
            emotion: raw_scores[emotion] - session_baseline.get(emotion, 0)
            for emotion in raw_scores
        }
        dominant = max(corrected, key=corrected.get)
        corrected_score = corrected[dominant]

        print(f"[Vision] raw={result['dominant_emotion']}({raw_scores[result['dominant_emotion']]:.1f}) "
              f"corrected_dominant={dominant}({corrected_score:.1f})")

        if corrected_score < CORRECTED_FLOOR:
            time.sleep(2)
            continue

        recent.append(dominant)

        if len(recent) == WINDOW_SIZE:
            settled, count = Counter(recent).most_common(1)[0]
            if count >= MIN_AGREEMENT and settled != last_logged:
                log_event("social", "emotion_detector", settled)
                print(f"[Vision] Logged emotion change: {settled} ({count}/{WINDOW_SIZE} agreement)")
                last_logged = settled

        time.sleep(2)

if __name__ == "__main__":
    t1 = threading.Thread(target=audio_loop, daemon=True)
    t2 = threading.Thread(target=vision_loop, daemon=True)
    t1.start()
    t2.start()

    print("\nRunning audio + vision simultaneously. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")