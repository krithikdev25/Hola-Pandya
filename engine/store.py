"""
Minimal local persistence (SQLite). Raw evidence stays in data/raw/*.csv
(untouched); this store holds canonical Asset state + append-only audit
log, kept in separate concepts per PS6 Section 8 ("raw evidence != reconciled
state").
"""
import json
import os
import sqlite3
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "assets.db")

_lock = threading.Lock()


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL,
        updated_at REAL
    )""")
    return conn


def save_asset(asset_dict: dict):
    import time
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO assets (id, state_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
            (asset_dict["id"], json.dumps(asset_dict), time.time()),
        )
        conn.commit()
        conn.close()


def load_asset(asset_id: str):
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT state_json FROM assets WHERE id=?", (asset_id,)).fetchone()
        conn.close()
    return json.loads(row[0]) if row else None


def list_assets():
    with _lock:
        conn = _connect()
        rows = conn.execute("SELECT state_json FROM assets").fetchall()
        conn.close()
    return [json.loads(r[0]) for r in rows]


def reset_db():
    with _lock:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
