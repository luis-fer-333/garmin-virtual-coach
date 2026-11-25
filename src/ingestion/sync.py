"""Backfill and incremental sync of Garmin Connect data into SQLite.

Usage (from the repo root, after logging in once so tokens are cached):

    # One-off: pull everything since your first activity. Safe to stop with
    # Ctrl-C and re-run; it resumes where it left off.
    python -m src.ingestion.sync backfill

    # Cap one run (e.g. split a long history over several evenings)
    python -m src.ingestion.sync backfill --max-days 365

    # Daily job (cron/launchd): new activities + the last 14 days of wellness
    python -m src.ingestion.sync daily

    # What's in the DB
    python -m src.ingestion.sync status

    # Re-derive typed tables from stored raw payloads (no Garmin calls)
    python -m src.ingestion.sync rebuild

Rate limits: Garmin doesn't publish limits for this unofficial API. We space
requests (``GARMIN_REQUEST_INTERVAL``, default 1.5 s), back off on HTTP 429
and stop the run cleanly if 429s persist. Everything fetched so far is
committed per day, so the next run picks up from there.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from config.settings import settings
from src.ingestion.normalize import (
    build_wellness_row,
    normalize_activity,
    weights_by_day,
)
from src.storage.local_store import LocalStore

logger = logging.getLogger(__name__)

# Per-day wellness endpoints: dataset name -> (api, "YYYY-MM-DD") -> payload
DAILY_DATASETS: dict[str, Callable[[Any, str], Any]] = {
    "daily_summary": lambda api, d: api.get_user_summary(d),
    "sleep": lambda api, d: api.get_sleep_data(d),
    "hrv": lambda api, d: api.get_hrv_data(d),
    "max_metrics": lambda api, d: api.get_max_metrics(d),
    "training_readiness": lambda api, d: api.get_training_readiness(d),
}

# Per-activity detail endpoints (laps + Garmin's time-in-zone). Needed by the
# personal-zones step to look at HR/pace per lap rather than per activity.
ACTIVITY_DETAIL_DATASETS: dict[str, Callable[[Any, str], Any]] = {
    "activity_splits": lambda api, aid: api.get_activity_splits(aid),
    "activity_hr_zones": lambda api, aid: api.get_activity_hr_in_timezones(aid),
}

# Snapshots of current settings, stored once per sync day (cheap, 1 call each).
PROFILE_DATASETS: dict[str, Callable[[Any], Any]] = {
    "user_profile": lambda api: api.get_user_profile(),
    "userprofile_settings": lambda api: api.get_userprofile_settings(),
    "heart_rate_zones": lambda api: api.get_heart_rate_zones(),
    "lactate_threshold_latest": lambda api: api.get_lactate_threshold(latest=True),
}

ACTIVITY_PAGE_SIZE = 100
DEFAULT_HISTORY_YEARS = 10  # used only if no activities and no --since
MAX_ATTEMPTS = 3  # a key that errored this many times is skipped from then on


class RateLimited(RuntimeError):
    """Garmin kept answering 429; stop and resume later."""


class AuthFailed(RuntimeError):
    pass


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__
    return "TooManyRequests" in name or "429" in str(exc)


def _is_auth(exc: Exception) -> bool:
    return "Authentication" in type(exc).__name__


def _is_empty(payload: Any) -> bool:
    if payload is None or payload == {} or payload == []:
        return True
    return False


@dataclass
class Throttle:
    interval: float = 1.5
    backoffs: tuple[float, ...] = (60, 300, 900)
    sleep: Callable[[float], None] = time.sleep
    _last: float = field(default=0.0, init=False)
    requests: int = field(default=0, init=False)

    def call(self, fn: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait + random.uniform(0, self.interval * 0.3))
            self._last = time.monotonic()
            self.requests += 1
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - library raises many types
                if _is_auth(e):
                    raise AuthFailed(str(e)) from e
                if not _is_rate_limit(e):
                    raise
                if attempt >= len(self.backoffs):
                    raise RateLimited(str(e)) from e
                pause = self.backoffs[attempt]
                logger.warning("Rate limited by Garmin; pausing %ss", pause)
                self.sleep(pause)
                attempt += 1


@dataclass
class SyncStats:
    requests: int = 0
    days_done: int = 0
    activities_seen: int = 0
    activities_new: int = 0
    details_done: int = 0
    errors: int = 0
    stopped_reason: str | None = None


class GarminSync:
    def __init__(
        self,
        api: Any,
        store: LocalStore,
        throttle: Throttle | None = None,
        today: date | None = None,
        refresh_days: int = 3,
        max_consecutive_errors: int = 25,
    ) -> None:
        self.api = api
        self.store = store
        self.throttle = throttle or Throttle(interval=settings.garmin_request_interval)
        self.today = today or date.today()
        # Days this recent are re-fetched even if stored: sleep/HRV for a night
        # land the next morning and VO2max/readiness get recalculated.
        self.refresh_days = refresh_days
        self.max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors = 0
        self.stats = SyncStats()

    # ------------------------------------------------------------- helpers
    def _fetch(self, dataset: str, key: str, fn: Callable[[], Any]) -> Any:
        try:
            payload = self.throttle.call(fn)
        except (RateLimited, AuthFailed):
            raise
        except Exception as e:  # noqa: BLE001
            self.stats.errors += 1
            self._consecutive_errors += 1
            logger.warning("%s %s failed: %s", dataset, key, e)
            self.store.put_raw(dataset, key, None, status="error", error=str(e)[:500])
            if self._consecutive_errors >= self.max_consecutive_errors:
                raise RuntimeError(
                    f"{self._consecutive_errors} errors in a row; last: {e}"
                ) from e
            return None
        self._consecutive_errors = 0
        self.store.put_raw(dataset, key, payload, status="empty" if _is_empty(payload) else "ok")
        return payload

    def _done(self, dataset: str) -> set[str]:
        return self.store.done_keys(dataset, max_attempts=MAX_ATTEMPTS)

    # ---------------------------------------------------------- activities
    def sync_activities(self, full: bool) -> None:
        """Page through the activity list (newest first, every sport).

        full=True walks to the very first activity. Otherwise it stops at the
        first page that reaches activities older than what's stored minus the
        refresh window.
        """
        known_latest = self.store.conn.execute("SELECT MAX(date) FROM activities").fetchone()[0]
        stop_before = None
        if not full and known_latest:
            stop_before = (
                date.fromisoformat(known_latest) - timedelta(days=self.refresh_days)
            ).isoformat()
        known = {str(r[0]) for r in self.store.conn.execute("SELECT activity_id FROM activities")}

        start = 0
        while True:
            page = self.throttle.call(
                lambda: self.api.get_activities(start, ACTIVITY_PAGE_SIZE)
            )
            if not page:
                break
            reached_old = False
            for a in page:
                aid = str(a.get("activityId"))
                self.store.put_raw("activity", aid, a)
                self.store.upsert_activity(normalize_activity(a))
                self.stats.activities_seen += 1
                if aid not in known:
                    self.stats.activities_new += 1
                d = (a.get("startTimeLocal") or "")[:10]
                if stop_before and d and d < stop_before:
                    reached_old = True
            self.store.commit()
            logger.info("Activities: %d stored so far (offset %d)", self.stats.activities_seen, start)
            if reached_old or len(page) < ACTIVITY_PAGE_SIZE:
                break
            start += ACTIVITY_PAGE_SIZE

    def sync_activity_details(self, max_items: int | None = None) -> None:
        ids = [str(r[0]) for r in self.store.conn.execute(
            "SELECT activity_id FROM activities ORDER BY start_time_local DESC"
        )]
        recent_cut = (self.today - timedelta(days=self.refresh_days)).isoformat()
        recent = {str(r[0]) for r in self.store.conn.execute(
            "SELECT activity_id FROM activities WHERE date >= ?", (recent_cut,)
        )}
        n = 0
        for dataset, fn in ACTIVITY_DETAIL_DATASETS.items():
            done = self._done(dataset)
            for aid in ids:
                if aid in done and aid not in recent:
                    continue
                if max_items is not None and n >= max_items:
                    return
                self._fetch(dataset, aid, lambda aid=aid, fn=fn: fn(self.api, aid))
                n += 1
                self.stats.details_done += 1
                if n % 25 == 0:
                    self.store.commit()
                    logger.info("Activity details: %d fetched", n)
        self.store.commit()

    # ------------------------------------------------------------ wellness
    def days_to_fetch(
        self, start: date, end: date, done: dict[str, set[str]]
    ) -> list[date]:
        """Days in [start, end] with any dataset missing, newest first."""
        refresh_from = self.today - timedelta(days=self.refresh_days)
        out = []
        d = end
        while d >= start:
            k = d.isoformat()
            if d >= refresh_from or any(k not in keys for keys in done.values()):
                out.append(d)
            d -= timedelta(days=1)
        return out

    def sync_wellness(
        self,
        start: date,
        end: date,
        datasets: list[str] | None = None,
        max_days: int | None = None,
    ) -> None:
        datasets = datasets or list(DAILY_DATASETS)
        unknown = set(datasets) - set(DAILY_DATASETS)
        if unknown:
            raise ValueError(f"Unknown dataset(s): {sorted(unknown)}")
        done = {ds: self._done(ds) for ds in datasets}
        days = self.days_to_fetch(start, end, done)
        if max_days is not None:
            days = days[:max_days]
        logger.info("Wellness: %d day(s) to fetch between %s and %s", len(days), start, end)
        refresh_from = self.today - timedelta(days=self.refresh_days)
        for i, d in enumerate(days, 1):
            k = d.isoformat()
            for ds in datasets:
                if d < refresh_from and k in done[ds]:
                    continue
                self._fetch(ds, k, lambda ds=ds: DAILY_DATASETS[ds](self.api, k))
            self.rebuild_wellness_day(k)
            self.store.commit()
            self.stats.days_done += 1
            if i % 30 == 0 or i == len(days):
                logger.info(
                    "Wellness: %d/%d days (at %s), %d requests",
                    i, len(days), k, self.throttle.requests,
                )

    def sync_body_composition(self, start: date, end: date) -> None:
        """Weight/body fat by year-long ranges (one call per year)."""
        y = start.year
        done = self._done("body_composition")
        current_year = self.today.year
        while y <= end.year:
            a = max(start, date(y, 1, 1))
            b = min(end, date(y, 12, 31))
            key = f"{y}"
            if key not in done or y >= current_year:
                payload = self._fetch(
                    "body_composition", key,
                    lambda a=a, b=b: self.api.get_body_composition(a.isoformat(), b.isoformat()),
                )
                self._weights_cached = None  # include this year's payload
                for day in weights_by_day(payload):
                    self.rebuild_wellness_day(day)
            y += 1
        self.store.commit()

    def sync_profile(self) -> None:
        key = self.today.isoformat()
        for ds, fn in PROFILE_DATASETS.items():
            if key in self._done(ds):
                continue
            self._fetch(ds, key, lambda fn=fn: fn(self.api))
        self.store.commit()

    # ------------------------------------------------------------ rebuild
    def _weights(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for _, payload in self.store.iter_raw("body_composition"):
            out.update(weights_by_day(payload))
        return out

    def rebuild_wellness_day(self, day: str, weights: dict[str, dict] | None = None) -> None:
        raw = {ds: self.store.get_raw(ds, day) for ds in DAILY_DATASETS}
        if weights is None:
            weights = self._weights_cache()
        if all(v is None for v in raw.values()) and day not in weights:
            return
        self.store.upsert_wellness(build_wellness_row(day, raw, weights.get(day)))

    _weights_cached: dict[str, dict] | None = None

    def _weights_cache(self) -> dict[str, dict]:
        if self._weights_cached is None:
            self._weights_cached = self._weights()
        return self._weights_cached

    def rebuild(self) -> None:
        """Re-derive activities and daily_wellness from raw_responses."""
        n = 0
        for _, a in self.store.iter_raw("activity"):
            self.store.upsert_activity(normalize_activity(a))
            n += 1
        days: set[str] = set()
        for ds in DAILY_DATASETS:
            days.update(k for k, _ in self.store.iter_raw(ds))
        weights = self._weights()
        days.update(weights)
        for d in sorted(days):
            self.rebuild_wellness_day(d, weights)
        self.store.commit()
        logger.info("Rebuilt %d activities and %d wellness days", n, len(days))

    # ------------------------------------------------------------- flows
    def earliest_activity_date(self) -> date | None:
        v = self.store.conn.execute("SELECT MIN(date) FROM activities").fetchone()[0]
        return date.fromisoformat(v) if v else None

    def backfill(
        self,
        since: date | None = None,
        max_days: int | None = None,
        details: bool = True,
        datasets: list[str] | None = None,
    ) -> SyncStats:
        return self._run(full=True, since=since, max_days=max_days, details=details, datasets=datasets)

    def daily(self, days: int = 14, details: bool = True) -> SyncStats:
        since = self.today - timedelta(days=days - 1)
        return self._run(full=False, since=since, max_days=None, details=details, datasets=None)

    def _run(self, full, since, max_days, details, datasets) -> SyncStats:
        try:
            self.sync_profile()
            self.sync_activities(full=full)
            start = since or self.earliest_activity_date() or (
                self.today - timedelta(days=365 * DEFAULT_HISTORY_YEARS)
            )
            self.sync_body_composition(start, self.today)
            self.sync_wellness(start, self.today, datasets=datasets, max_days=max_days)
            if details:
                self.sync_activity_details()
        except RateLimited as e:
            self.stats.stopped_reason = f"rate limited ({e}); progress saved, re-run later"
        except AuthFailed as e:
            self.stats.stopped_reason = f"authentication failed ({e}); log in again"
        except KeyboardInterrupt:
            self.stats.stopped_reason = "interrupted; progress saved"
        finally:
            self.store.commit()
            self.stats.requests = self.throttle.requests
        return self.stats


# ---------------------------------------------------------------------- CLI
def _print_status(store: LocalStore) -> None:
    print(json.dumps(store.status(), indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.ingestion.sync", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="Fetch all history (resumable)")
    b.add_argument("--since", type=date.fromisoformat, help="Earliest day (default: first activity)")
    b.add_argument("--max-days", type=int, help="Max wellness days to fetch this run")
    b.add_argument("--datasets", help=f"Comma list of daily datasets ({','.join(DAILY_DATASETS)})")
    b.add_argument("--no-details", action="store_true", help="Skip per-activity laps/zones")

    d = sub.add_parser("daily", help="Incremental sync for a scheduled job")
    d.add_argument("--days", type=int, default=14, help="Wellness window to check (default 14)")
    d.add_argument("--no-details", action="store_true")

    sub.add_parser("status", help="Show what's stored")
    sub.add_parser("rebuild", help="Re-derive typed tables from raw payloads")

    p.add_argument("--db", help="SQLite path (default: LOCAL_DB_PATH)")
    args = p.parse_args(argv)

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    store = LocalStore(args.db)

    if args.cmd == "status":
        _print_status(store)
        return 0
    if args.cmd == "rebuild":
        GarminSync(api=None, store=store).rebuild()
        _print_status(store)
        return 0

    from src.ingestion.garmin_client import GarminClient

    client = GarminClient()
    client.connect()
    sync = GarminSync(client.api, store)
    if args.cmd == "backfill":
        datasets = args.datasets.split(",") if args.datasets else None
        stats = sync.backfill(
            since=args.since, max_days=args.max_days,
            details=not args.no_details, datasets=datasets,
        )
    else:
        stats = sync.daily(days=args.days, details=not args.no_details)

    print(json.dumps(stats.__dict__, indent=2))
    _print_status(store)
    return 0 if stats.stopped_reason is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
