import sqlite3
from pathlib import Path

DB_PATH = Path("alerts") / "cheating_alerts.db"


class AlertDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    student_id INTEGER NOT NULL,
                    behavior TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    video_time TEXT NOT NULL,
                    screenshot_path TEXT NOT NULL
                )
                """
            )

    def insert_alert(self, record: dict):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts (timestamp, student_id, behavior, confidence, video_time, screenshot_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    record["student_id"],
                    record["behavior"],
                    record["confidence"],
                    record["video_time"],
                    record["screenshot_path"],
                ),
            )

    def fetch_all(self):
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT timestamp, student_id, behavior, confidence, video_time, screenshot_path
                FROM alerts
                ORDER BY id DESC
                """
            )
            return cur.fetchall()
