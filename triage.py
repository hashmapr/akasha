"""
AKASHA Triage — Autopilot vs Interrupt decision layer.

Decoupled from the sensing pipelines on purpose: reads from the existing
`events` table (akasha_db.py), never writes to it, never imports from
listen.py/sense.py. Runs as its own poller against its own table
(`triage_decisions`). If this breaks, it breaks alone.

Design:
- Every feature has a FEATURE_PROFILE entry: importance + whether it's
  allowed to interrupt at all. Most things are NOT interruptible — that's
  the point of Autopilot. Only a small set can ever fire.
- A daily interrupt budget caps how many INTERRUPT decisions can fire
  per day, regardless of how many high-importance events come in.
- A cooldown prevents the same feature from interrupting twice within
  a short window (no spam on "stressed... stressed... stressed").
- confidence is a 0-1 heuristic score, currently rule-based (quantity
  present, importance tier, repetition). Not ML — deliberately simple
  and inspectable until there's a reason to make it smarter.

This does NOT do anything with the interrupt yet beyond logging the
decision and printing it. Wiring an actual interrupt channel (sound,
Halo display, whatever) is a separate, later step — don't conflate
"decide" with "deliver."
"""

import sqlite3
import threading
import time
from datetime import date, datetime, timedelta

DB_PATH = "akasha.db"
_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

DAILY_INTERRUPT_BUDGET = 5
COOLDOWN_MINUTES = 30  # same feature can't interrupt again within this window

# (module, feature) -> profile
# importance: "low" | "medium" | "high"
# interruptible: can this feature EVER produce an INTERRUPT, regardless of confidence?
FEATURE_PROFILE = {
    ("body", "hydration"):       {"importance": "low",    "interruptible": False},
    ("body", "caffeine"):        {"importance": "low",    "interruptible": False},
    ("body", "drinks"):          {"importance": "medium",  "interruptible": False},
    ("body", "training"):        {"importance": "low",    "interruptible": False},
    ("body", "supplement_log"):  {"importance": "low",    "interruptible": False},
    ("body", "energy"):          {"importance": "medium", "interruptible": True},
    ("body", "tension"):         {"importance": "high",   "interruptible": True},
    ("self", "emotion"):         {"importance": "medium", "interruptible": True},
}

DEFAULT_PROFILE = {"importance": "low", "interruptible": False}


def init_triage_db():
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS triage_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                module TEXT,
                feature TEXT,
                confidence REAL,
                decision TEXT,
                reason TEXT
            )
        """)
        _conn.commit()


def _last_seen_event_id():
    with _lock:
        row = _conn.execute("SELECT MAX(event_id) FROM triage_decisions").fetchone()
    return row[0] or 0


def _new_events(since_id):
    with _lock:
        rows = _conn.execute(
            """SELECT id, timestamp, module, feature, amount, unit, ml_amount, raw_text
               FROM events WHERE id > ? ORDER BY id ASC""",
            (since_id,)
        ).fetchall()
    return rows


def _interrupts_today():
    today = date.today().isoformat()
    with _lock:
        row = _conn.execute(
            "SELECT COUNT(*) FROM triage_decisions WHERE decision = 'INTERRUPT' AND DATE(timestamp) = ?",
            (today,)
        ).fetchone()
    return row[0] or 0


def _last_interrupt_for_feature(module, feature):
    with _lock:
        row = _conn.execute(
            """SELECT timestamp FROM triage_decisions
               WHERE module = ? AND feature = ? AND decision = 'INTERRUPT'
               ORDER BY timestamp DESC LIMIT 1""",
            (module, feature)
        ).fetchone()
    return row[0] if row else None


def _record_decision(event_id, module, feature, confidence, decision, reason):
    with _lock:
        _conn.execute(
            """INSERT INTO triage_decisions (event_id, module, feature, confidence, decision, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_id, module, feature, confidence, decision, reason)
        )
        _conn.commit()


def score_confidence(module, feature, amount, profile):
    """
    Rule-based confidence, 0-1. Quantity presence and importance tier are the
    only inputs right now. Replace with something smarter once there's real
    behavioral data to ground it in — guessing past this point would be
    fabricating precision that doesn't exist yet.
    """
    base = {"low": 0.3, "medium": 0.55, "high": 0.75}[profile["importance"]]
    if amount is not None:
        base += 0.15
    return min(base, 1.0)


def decide(event):
    event_id, timestamp, module, feature, amount, unit, ml_amount, raw_text = event
    profile = FEATURE_PROFILE.get((module, feature), DEFAULT_PROFILE)
    confidence = score_confidence(module, feature, amount, profile)

    if not profile["interruptible"]:
        return confidence, "AUTOPILOT", f"{module}/{feature} is not an interruptible feature"

    if profile["importance"] != "high":
        return confidence, "AUTOPILOT", f"importance={profile['importance']}, below interrupt bar"

    if confidence < 0.7:
        return confidence, "AUTOPILOT", f"confidence {confidence:.2f} below 0.70 threshold"

    if _interrupts_today() >= DAILY_INTERRUPT_BUDGET:
        return confidence, "AUTOPILOT", f"daily interrupt budget ({DAILY_INTERRUPT_BUDGET}) exhausted"

    last = _last_interrupt_for_feature(module, feature)
    if last:
        last_dt = datetime.fromisoformat(last)
        if datetime.now() - last_dt < timedelta(minutes=COOLDOWN_MINUTES):
            return confidence, "AUTOPILOT", f"cooldown active ({COOLDOWN_MINUTES}min) for {module}/{feature}"

    return confidence, "INTERRUPT", f"high-importance, confidence {confidence:.2f}, budget and cooldown clear"


def process_once():
    """Process all events since the last run. Returns number processed."""
    since_id = _last_seen_event_id()
    events = _new_events(since_id)
    for event in events:
        event_id, timestamp, module, feature, amount, unit, ml_amount, raw_text = event
        confidence, decision, reason = decide(event)
        _record_decision(event_id, module, feature, confidence, decision, reason)
        if decision == "INTERRUPT":
            print(f"[TRIAGE] INTERRUPT — {module}/{feature} | conf={confidence:.2f} | {reason}")
            print(f"         source: \"{raw_text}\"")
        else:
            print(f"[TRIAGE] autopilot — {module}/{feature} | conf={confidence:.2f} | {reason}")
    return len(events)


def run(poll_seconds=2):
    init_triage_db()
    print(f"[TRIAGE] Running. Polling every {poll_seconds}s. Daily budget={DAILY_INTERRUPT_BUDGET}, cooldown={COOLDOWN_MINUTES}min. Ctrl+C to stop.\n")
    try:
        while True:
            process_once()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n[TRIAGE] Stopped.")


if __name__ == "__main__":
    run()