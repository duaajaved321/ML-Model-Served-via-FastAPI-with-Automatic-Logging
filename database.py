"""Persistent audit trail without storing sensitive questionnaire responses."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def db_path() -> Path:
    return Path(os.getenv('AUDIT_DB_PATH', str(ROOT / 'audit.db')))

def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute('PRAGMA busy_timeout = 10000')
    return connection

def initialize():
    with connect() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS prediction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL UNIQUE,
            timestamp_utc TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            http_status INTEGER NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failed')),
            predicted_class TEXT,
            error_type TEXT
        )''')

def log_request(request_id: str, http_status: int, outcome: str, predicted_class=None, error_type=None):
    with connect() as con:
        con.execute('''INSERT INTO prediction_audit
            (request_id, timestamp_utc, endpoint, http_status, outcome, predicted_class, error_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (request_id, datetime.now(timezone.utc).isoformat(), '/predict',
             http_status, outcome, predicted_class, error_type))
