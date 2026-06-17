import sounddevice as sd
from faster_whisper import WhisperModel
from akasha_db import init_db, log_event

init_db()

RULES = {
    "water": ("body", "hydration"),
    "drank": ("body", "hydration"),
    "coffee": ("body", "caffeine"),
    "tea": ("body", "caffeine"),
    "beer": ("body", "drinks"),
    "wine": ("body", "drinks"),
}

print("Recording 5 seconds... say something")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
print("Transcribing...")

model = WhisperModel("tiny", device="cpu")
segments, _ = model.transcribe(audio.flatten(), language="en")

text = " ".join(segment.text for segment in segments).strip()
print(f"-> {text}")

matched = False
for keyword, (module, feature) in RULES.items():
    if keyword in text.lower():
        log_event(module, feature, text)
        print(f"Logged to {module}/{feature}.")
        matched = True
        break

if not matched:
    print("No match. Not logged.")
