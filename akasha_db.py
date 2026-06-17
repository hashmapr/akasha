import sqlite3
import threading
from datetime import date

_lock = threading.Lock()
_conn = sqlite3.connect("akasha.db", check_same_thread=False)

def init_db():
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                module TEXT,
                feature TEXT,
                amount REAL,
                unit TEXT,
                ml_amount REAL,
                hydration_factor REAL,
                net_hydration_ml REAL,
                raw_text TEXT
            )
        """)
        _conn.commit()

def log_event(module, feature, raw_text, amount=None, unit=None, ml_amount=None, hydration_factor=None):
    net_hydration_ml = None
    if ml_amount is not None and hydration_factor is not None:
        net_hydration_ml = ml_amount * hydration_factor

    with _lock:
        _conn.execute(
            """INSERT INTO events (module, feature, amount, unit, ml_amount, hydration_factor, net_hydration_ml, raw_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (module, feature, amount, unit, ml_amount, hydration_factor, net_hydration_ml, raw_text)
        )
        _conn.commit()

def get_net_hydration_today():
    today = date.today().isoformat()
    with _lock:
        row = _conn.execute(
            "SELECT SUM(net_hydration_ml) FROM events WHERE DATE(timestamp) = ?",
            (today,)
        ).fetchone()
    return row[0] or 0
