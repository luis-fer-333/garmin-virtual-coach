"""Local storage using SQLite for activities and coaching history."""

import json
import sqlite3
import logging
from datetime import date
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class LocalStore:
    """SQLite-based local storage for activities and coaching responses."""

    def __init__(self, db_path: "str | None" = None) -> None:
        self.db_path = db_path or settings.local_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activities (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coaching_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_date TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save_activity(self, activity_id: str, activity_date: str, data: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO activities (id, date, data) VALUES (?, ?, ?)",
                (activity_id, activity_date, json.dumps(data)),
            )

    def save_review(self, review_date: str, prompt: str, response: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO coaching_reviews (review_date, prompt, response) "
                "VALUES (?, ?, ?)",
                (review_date, prompt, response),
            )

    def get_reviews(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT review_date, response, created_at "
                "FROM coaching_reviews ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"date": r[0], "response": r[1], "created_at": r[2]} for r in rows
        ]
