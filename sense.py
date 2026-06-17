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

    WINDOW_SIZE = 5
    recent = deque(maxlen=WINDOW_SIZE)
    last_logged = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            dominant = result['dominant_emotion']
            recent.append(dominant)

            if len(recent) == WINDOW_SIZE:
                settled = Counter(recent).most_common(1)[0][0]
                if settled != last_logged:
                    log_event("self", "emotion", settled)
                    print(f"[Vision] Logged emotion change: {settled}")
                    last_logged = settled
        except Exception as e:
            print(f"[Vision] Error: {e}")
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
