"""Tests for the Garmin backfill / incremental sync (no network)."""

from datetime import date, timedelta

import pytest

from src.ingestion.sync import GarminSync, Throttle
from src.storage.local_store import LocalStore

TODAY = date(2026, 11, 25)


class GarminConnectTooManyRequestsError(Exception):
    pass


def make_activity(aid: int, day: date, sport: str = "running") -> dict:
    return {
        "activityId": aid,
        "activityName": f"{sport} {aid}",
        "startTimeLocal": f"{day.isoformat()} 07:00:00",
        "startTimeGMT": f"{day.isoformat()} 06:00:00",
        "activityType": {"typeKey": sport, "parentTypeId": 17},
        "duration": 1800.0,
        "distance": 5000.0,
        "averageHR": 150.0,
        "maxHR": 175.0,
        "averageSpeed": 2.78,
        "hrTimeInZone_2": 600.0,
        "vO2MaxValue": 52.0,
    }


class FakeGarmin:
    """Stands in for garminconnect.Garmin; counts calls."""

    def __init__(self, activities, first_wellness_day=None, rate_limit_after=None):
        # newest first, like Garmin
        self.activities = sorted(activities, key=lambda a: a["startTimeLocal"], reverse=True)
        self.first_wellness_day = first_wellness_day
        self.calls: list[tuple] = []
        self.rate_limit_after = rate_limit_after

    def _hit(self, *call):
        self.calls.append(call)
        if self.rate_limit_after is not None and len(self.calls) > self.rate_limit_after:
            raise GarminConnectTooManyRequestsError("429")

    def get_activities(self, start, limit):
        self._hit("activities", start)
        return self.activities[start:start + limit]

    def _has(self, d):
        return self.first_wellness_day is None or d >= self.first_wellness_day.isoformat()

    def get_user_summary(self, d):
        self._hit("summary", d)
        return {"restingHeartRate": 48, "totalSteps": 9000, "averageStressLevel": -1,
                "bodyBatteryHighestValue": 90} if self._has(d) else {}

    def get_sleep_data(self, d):
        self._hit("sleep", d)
        if not self._has(d):
            return {}
        return {"dailySleepDTO": {"sleepTimeSeconds": 27000,
                                  "sleepScores": {"overall": {"value": 82}}}}

    def get_hrv_data(self, d):
        self._hit("hrv", d)
        if d == "2026-11-20":
            raise ValueError("boom")
        return {"hrvSummary": {"lastNightAvg": 61, "status": "BALANCED",
                               "baseline": {"balancedLow": 55, "balancedUpper": 70}}} if self._has(d) else None

    def get_max_metrics(self, d):
        self._hit("max", d)
        return [{"generic": {"vo2MaxPreciseValue": 52.4}, "cycling": None}]

    def get_training_readiness(self, d):
        self._hit("readiness", d)
        return [{"score": 70, "level": "HIGH", "timestampLocal": f"{d}T09:00:00"},
                {"score": 55, "level": "MODERATE", "timestampLocal": f"{d}T06:00:00"}]

    def get_body_composition(self, a, b):
        self._hit("bodycomp", a, b)
        return {"dateWeightList": [{"calendarDate": "2026-11-22", "weight": 72500.0, "bodyFat": 14.2}]}

    def get_activity_splits(self, aid):
        self._hit("splits", aid)
        return {"lapDTOs": [{"averageHR": 150}]}

    def get_activity_hr_in_timezones(self, aid):
        self._hit("hrzones", aid)
        return [{"zoneNumber": 1, "secsInZone": 100}]

    def get_user_profile(self):
        self._hit("profile")
        return {"userData": {"gender": "MALE"}}

    def get_userprofile_settings(self):
        self._hit("settings")
        return {"userData": {"lactateThresholdHeartRate": 172}}

    def get_heart_rate_zones(self):
        self._hit("hrz")
        return []

    def get_lactate_threshold(self, latest=True):
        self._hit("lt")
        return {"speed_and_heart_rate": {"heartRate": 172}}


def no_sleep_throttle(**kw):
    return Throttle(interval=0, sleep=lambda s: None, **kw)


@pytest.fixture
def store():
    s = LocalStore(":memory:")
    yield s
    s.close()


def make_sync(api, store, **kw):
    return GarminSync(api, store, throttle=no_sleep_throttle(), today=TODAY, **kw)


def test_backfill_stores_all_sports_and_wellness(store):
    acts = [make_activity(i, TODAY - timedelta(days=i), sport)
            for i, sport in [(1, "running"), (3, "cycling"), (9, "lap_swimming")]]
    api = FakeGarmin(acts)
    stats = make_sync(api, store).backfill()

    assert stats.stopped_reason is None
    st = store.status()
    assert st["activities"]["n"] == 3
    assert {r["sport"] for r in st["activities_by_sport"]} == {"running", "cycling", "lap_swimming"}
    # Wellness from first activity (9 days ago) through today = 10 days
    assert st["wellness"]["n"] == 10
    assert st["wellness"]["first"] == (TODAY - timedelta(days=9)).isoformat()

    row = store.conn.execute("SELECT * FROM daily_wellness WHERE date='2026-11-22'").fetchone()
    assert row["resting_hr"] == 48
    assert row["avg_stress"] is None  # -1 means "no data"
    assert row["sleep_score"] == 82
    assert row["hrv_last_night_avg"] == 61
    assert row["vo2max_running"] == 52.4
    assert row["training_readiness"] == 55  # morning value
    assert row["weight_kg"] == 72.5

    # features still get the raw API shape
    df = store.activities_df(sport="running")
    assert list(df["activityId"]) == [1]
    assert "startTimeLocal" in df.columns


def test_backfill_is_resumable_and_skips_done_days(store):
    acts = [make_activity(1, TODAY - timedelta(days=30))]
    api = FakeGarmin(acts)
    s1 = make_sync(api, store)
    s1.backfill(max_days=10, details=False)
    assert store.status()["wellness"]["n"] == 10

    api.calls.clear()
    s2 = make_sync(api, store)
    s2.backfill(details=False)
    assert store.status()["wellness"]["n"] == 31
    summary_days = {c[1] for c in api.calls if c[0] == "summary"}
    # old, already-complete days are not fetched again (except refresh window)
    fetched_before = {(TODAY - timedelta(days=i)).isoformat() for i in range(4, 10)}
    assert not (summary_days & fetched_before)


def test_failed_day_is_retried_but_keeps_other_data(store):
    api = FakeGarmin([make_activity(1, TODAY - timedelta(days=10))])
    make_sync(api, store).backfill(details=False)
    row = store.conn.execute("SELECT status, attempts FROM raw_responses "
                             "WHERE dataset='hrv' AND key='2026-11-20'").fetchone()
    assert row["status"] == "error"
    assert "2026-11-20" not in store.done_keys("hrv", max_attempts=3)

    api.calls.clear()
    make_sync(api, store).backfill(details=False)
    assert ("hrv", "2026-11-20") in api.calls
    assert ("summary", "2026-11-20") not in api.calls


def test_rate_limit_stops_cleanly_and_saves_progress(store):
    api = FakeGarmin([make_activity(1, TODAY - timedelta(days=60))], rate_limit_after=40)
    sync = GarminSync(api, store, today=TODAY,
                      throttle=Throttle(interval=0, backoffs=(0,), sleep=lambda s: None))
    stats = sync.backfill(details=False)
    assert stats.stopped_reason and "rate limited" in stats.stopped_reason
    saved = store.status()["wellness"]["n"]
    assert 0 < saved < 61

    api.rate_limit_after = None
    make_sync(api, store).backfill(details=False)
    assert store.status()["wellness"]["n"] == 61


def test_daily_only_pages_new_activities(store):
    old = [make_activity(i, TODAY - timedelta(days=i)) for i in range(5, 400, 2)]
    api = FakeGarmin(old)
    make_sync(api, store).backfill(since=TODAY - timedelta(days=20), details=False)
    assert store.status()["activities"]["n"] == len(old)

    api.activities.insert(0, make_activity(9999, TODAY))
    api.calls.clear()
    stats = make_sync(api, store).daily(details=False)
    assert stats.activities_new == 1
    assert [c for c in api.calls if c[0] == "activities"] == [("activities", 0)]
    # 14-day window, all but the 3-day refresh window already stored
    assert len({c[1] for c in api.calls if c[0] == "summary"}) == 4


def test_activity_details_fetched_once(store):
    api = FakeGarmin([make_activity(1, TODAY - timedelta(days=20)),
                      make_activity(2, TODAY - timedelta(days=15))])
    make_sync(api, store).backfill()
    assert len([c for c in api.calls if c[0] == "splits"]) == 2
    api.calls.clear()
    make_sync(api, store).backfill()
    assert not [c for c in api.calls if c[0] in ("splits", "hrzones")]


def test_rebuild_from_raw(store):
    api = FakeGarmin([make_activity(1, TODAY - timedelta(days=3))])
    make_sync(api, store).backfill(details=False)
    store.conn.execute("DELETE FROM daily_wellness")
    store.conn.execute("DELETE FROM activities")
    GarminSync(None, store, today=TODAY).rebuild()
    assert store.status()["activities"]["n"] == 1
    assert store.status()["wellness"]["n"] == 4


def test_legacy_activities_table_is_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "coach.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE activities (id TEXT PRIMARY KEY, date TEXT, data JSON, created_at TIMESTAMP)")
    c.execute("INSERT INTO activities VALUES ('1','2024-01-01','{}',NULL)")
    c.commit(); c.close()
    s = LocalStore(str(db))
    assert s.conn.execute("SELECT COUNT(*) FROM activities_legacy").fetchone()[0] == 1
    assert s.status()["activities"]["n"] == 0
    s.close()


def test_weights_across_years(store):
    api = FakeGarmin([make_activity(1, date(2025, 12, 20))])
    payloads = {
        "2025-12-20": {"dateWeightList": [{"calendarDate": "2025-12-24", "weight": 74000.0}]},
        "2026-01-01": {"dateWeightList": [{"calendarDate": "2026-01-02", "weight": 73000.0}]},
    }
    api.get_body_composition = lambda a, b: payloads[a]
    make_sync(api, store).backfill(details=False)
    w = dict(store.conn.execute("SELECT date, weight_kg FROM daily_wellness WHERE weight_kg IS NOT NULL"))
    assert w == {"2025-12-24": 74.0, "2026-01-02": 73.0}
