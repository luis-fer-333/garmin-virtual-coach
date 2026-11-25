"""Local SQLite storage for Garmin history and coaching output.

Layout (schema version 2):

``raw_responses``   Every Garmin payload exactly as returned, keyed by
                    (dataset, key). ``key`` is a date (YYYY-MM-DD) for daily
                    datasets, an activity id for per-activity datasets, or a
                    date range / snapshot date for the rest. ``status`` is
                    ``ok``, ``empty`` (Garmin had nothing) or ``error``.
                    This table doubles as the sync ledger: a day is done when
                    its row is ``ok`` or ``empty``.
``activities``      One typed row per activity, every sport, derived from the
                    raw activity summary (see ``src/ingestion/normalize.py``).
``daily_wellness``  One typed row per calendar day: RHR, HRV, sleep, Body
                    Battery, stress, VO2max, readiness, weight...
``coaching_reviews`` LLM review history (unchanged).

Typed tables can always be rebuilt from ``raw_responses`` without touching
Garmin (``python -m src.ingestion.sync rebuild``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from config.settings import settings
from src.ingestion.normalize import ACTIVITY_COLUMNS, WELLNESS_COLUMNS

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_REAL_COLS = {
    "distance_m", "avg_speed_mps", "max_speed_mps", "elevation_gain_m",
    "elevation_loss_m", "calories", "aerobic_te", "anaerobic_te",
    "training_load", "vo2max", "avg_power", "max_power", "norm_power",
    "avg_cadence", "max_cadence", "duration_s", "moving_duration_s",
    "elapsed_duration_s", "hr_z1_s", "hr_z2_s", "hr_z3_s", "hr_z4_s", "hr_z5_s",
    "vo2max_running", "vo2max_cycling", "weight_kg", "body_fat_pct",
    "avg_spo2", "avg_waking_respiration", "sleep_avg_respiration",
    "sleep_avg_stress", "total_kcal", "active_kcal",
}
_TEXT_COLS = {
    "start_time_local", "start_time_gmt", "date", "sport", "name",
    "hrv_status", "training_readiness_level",
}


def _col_type(c: str) -> str:
    if c in _TEXT_COLS:
        return "TEXT"
    if c in _REAL_COLS:
        return "REAL"
    return "INTEGER"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalStore:
    """SQLite-based local storage."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.local_db_path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    # ------------------------------------------------------------------ setup
    def _init_db(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)")

        # v1 had activities(id, date, data); nothing wrote to it, but keep any
        # rows a user might have by moving them aside.
        cols = {r["name"] for r in c.execute("PRAGMA table_info(activities)")}
        if cols and "data" in cols and "activity_id" not in cols:
            logger.info("Migrating legacy 'activities' table to 'activities_legacy'")
            c.execute("ALTER TABLE activities RENAME TO activities_legacy")

        c.execute("""
            CREATE TABLE IF NOT EXISTS raw_responses (
                dataset    TEXT NOT NULL,
                key        TEXT NOT NULL,
                status     TEXT NOT NULL CHECK (status IN ('ok','empty','error')),
                payload    JSON,
                error      TEXT,
                attempts   INTEGER NOT NULL DEFAULT 1,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (dataset, key)
            )
        """)

        act_cols = ",\n".join(
            f"{col} {_col_type(col)}" + (" PRIMARY KEY" if col == "activity_id" else "")
            for col in ACTIVITY_COLUMNS
        )
        c.execute(f"CREATE TABLE IF NOT EXISTS activities ({act_cols}, updated_at TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_activities_date ON activities(date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_activities_sport_date ON activities(sport, date)")

        well_cols = ",\n".join(
            f"{col} {_col_type(col)}" + (" PRIMARY KEY" if col == "date" else "")
            for col in WELLNESS_COLUMNS
        )
        c.execute(f"CREATE TABLE IF NOT EXISTS daily_wellness ({well_cols}, updated_at TEXT)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS coaching_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_date TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    # -------------------------------------------------------------- raw layer
    def put_raw(
        self,
        dataset: str,
        key: str,
        payload: Any,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO raw_responses (dataset, key, status, payload, error, attempts, fetched_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(dataset, key) DO UPDATE SET
                -- a failed refresh never clobbers data we already have
                status=CASE WHEN excluded.status='error' AND raw_responses.status IN ('ok','empty')
                            THEN raw_responses.status ELSE excluded.status END,
                payload=CASE WHEN excluded.status='error' THEN raw_responses.payload
                             ELSE excluded.payload END,
                error=excluded.error,
                attempts=CASE WHEN excluded.status='error' THEN raw_responses.attempts + 1
                              ELSE 1 END,
                fetched_at=excluded.fetched_at
            """,
            (
                dataset, str(key), status,
                json.dumps(payload, default=str) if payload is not None else None,
                error, _now(),
            ),
        )

    def get_raw(self, dataset: str, key: str) -> Any:
        row = self.conn.execute(
            "SELECT payload FROM raw_responses WHERE dataset=? AND key=?",
            (dataset, str(key)),
        ).fetchone()
        return json.loads(row["payload"]) if row and row["payload"] else None

    def done_keys(self, dataset: str, max_attempts: int | None = None) -> set[str]:
        """Keys that need no more fetching: ok/empty, or errored too often."""
        q = "SELECT key FROM raw_responses WHERE dataset=? AND (status IN ('ok','empty')"
        params: list[Any] = [dataset]
        if max_attempts:
            q += " OR attempts >= ?"
            params.append(max_attempts)
        return {r["key"] for r in self.conn.execute(q + ")", params)}

    def iter_raw(self, dataset: str) -> Iterator[tuple[str, Any]]:
        for r in self.conn.execute(
            "SELECT key, payload FROM raw_responses WHERE dataset=? AND payload IS NOT NULL",
            (dataset,),
        ):
            yield r["key"], json.loads(r["payload"])

    # ---------------------------------------------------------- typed tables
    def upsert_activity(self, row: dict) -> None:
        self._upsert("activities", ACTIVITY_COLUMNS, row)

    def upsert_wellness(self, row: dict) -> None:
        self._upsert("daily_wellness", WELLNESS_COLUMNS, row)

    def _upsert(self, table: str, columns: list[str], row: dict) -> None:
        cols = columns + ["updated_at"]
        values = [row.get(c) for c in columns] + [_now()]
        self.conn.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})",
            values,
        )

    # ------------------------------------------------------------- reading
    def activities_df(self, sport: str | Iterable[str] | None = None, since: str | None = None):
        """Raw activity payloads as a DataFrame (same shape the live API gave).

        This is what ``src/features/*`` expect (``startTimeLocal``,
        ``duration``, ``averageHR``...).
        """
        import pandas as pd

        q = (
            "SELECT r.payload FROM activities a "
            "JOIN raw_responses r ON r.dataset='activity' AND r.key=CAST(a.activity_id AS TEXT) "
            "WHERE 1=1"
        )
        params: list[Any] = []
        if sport:
            sports = [sport] if isinstance(sport, str) else list(sport)
            q += f" AND a.sport IN ({', '.join('?' * len(sports))})"
            params += sports
        if since:
            q += " AND a.date >= ?"
            params.append(since)
        q += " ORDER BY a.start_time_local"
        rows = [json.loads(r["payload"]) for r in self.conn.execute(q, params)]
        return pd.DataFrame(rows)

    def wellness_df(self, since: str | None = None):
        import pandas as pd

        q = "SELECT * FROM daily_wellness"
        params: list[Any] = []
        if since:
            q += " WHERE date >= ?"
            params.append(since)
        return pd.read_sql_query(q + " ORDER BY date", self.conn, params=params)

    def status(self) -> dict:
        c = self.conn
        out: dict[str, Any] = {}
        out["activities"] = c.execute(
            "SELECT COUNT(*) n, MIN(date) first, MAX(date) last FROM activities"
        ).fetchone()
        out["activities_by_sport"] = c.execute(
            "SELECT sport, COUNT(*) n FROM activities GROUP BY sport ORDER BY n DESC"
        ).fetchall()
        out["wellness"] = c.execute(
            "SELECT COUNT(*) n, MIN(date) first, MAX(date) last, "
            "COUNT(resting_hr) rhr, COUNT(hrv_last_night_avg) hrv, "
            "COUNT(sleep_s) sleep, COUNT(vo2max_running) vo2 FROM daily_wellness"
        ).fetchone()
        out["raw"] = c.execute(
            "SELECT dataset, status, COUNT(*) n FROM raw_responses "
            "GROUP BY dataset, status ORDER BY dataset, status"
        ).fetchall()
        return {k: ([dict(r) for r in v] if isinstance(v, list) else dict(v)) for k, v in out.items()}

    # ------------------------------------------------------ coaching reviews
    def save_activity(self, activity_id: str, activity_date: str, data: dict) -> None:
        """Back-compat helper: store one raw activity summary."""
        from src.ingestion.normalize import normalize_activity

        self.put_raw("activity", str(activity_id), data)
        row = normalize_activity(data)
        row["activity_id"] = row["activity_id"] or activity_id
        row["date"] = row["date"] or activity_date
        self.upsert_activity(row)
        self.commit()

    def save_review(self, review_date: str, prompt: str, response: str) -> None:
        self.conn.execute(
            "INSERT INTO coaching_reviews (review_date, prompt, response) VALUES (?, ?, ?)",
            (review_date, prompt, response),
        )
        self.commit()

    def get_reviews(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT review_date, response, created_at "
            "FROM coaching_reviews ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"date": r[0], "response": r[1], "created_at": r[2]} for r in rows]
