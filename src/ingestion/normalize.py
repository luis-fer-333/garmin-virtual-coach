"""Turn raw Garmin Connect payloads into flat rows for the SQLite store.

Every raw payload is kept in ``raw_responses`` exactly as Garmin returned it.
The functions here only *derive* the typed columns of ``activities`` and
``daily_wellness`` from those payloads, so if a field name below turns out to
be wrong for your account/device, fix it here and run
``python -m src.ingestion.sync rebuild`` — no re-download needed.

Field names come from the garminconnect library docs and community examples;
the ones marked ``# GUESS`` have not been checked against a real payload yet.
All lookups are defensive: a missing key yields ``None``, never an exception.
"""

from __future__ import annotations

from typing import Any


def _get(d: Any, *path: str) -> Any:
    """Safe nested lookup: _get(d, "a", "b") == d["a"]["b"] or None."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _first(d: Any, *keys: str) -> Any:
    """Return the first non-None value among several candidate keys."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
# Activities
# --------------------------------------------------------------------------- #

ACTIVITY_COLUMNS = [
    "activity_id", "start_time_local", "start_time_gmt", "date",
    "sport", "sport_parent_id", "name",
    "duration_s", "moving_duration_s", "elapsed_duration_s", "distance_m",
    "avg_hr", "max_hr", "avg_speed_mps", "max_speed_mps",
    "elevation_gain_m", "elevation_loss_m", "calories",
    "aerobic_te", "anaerobic_te", "training_load", "vo2max",
    "avg_power", "max_power", "norm_power",
    "avg_cadence", "max_cadence",
    "hr_z1_s", "hr_z2_s", "hr_z3_s", "hr_z4_s", "hr_z5_s",
    "lactate_threshold_bpm", "device_id", "manual",
]


def normalize_activity(a: dict) -> dict:
    """Flatten one activity summary (from the activity list endpoint)."""
    start_local = a.get("startTimeLocal")  # "YYYY-MM-DD HH:MM:SS"
    return {
        "activity_id": a.get("activityId"),
        "start_time_local": start_local,
        "start_time_gmt": a.get("startTimeGMT"),
        "date": start_local[:10] if isinstance(start_local, str) else None,
        "sport": _get(a, "activityType", "typeKey"),
        "sport_parent_id": _get(a, "activityType", "parentTypeId"),
        "name": a.get("activityName"),
        "duration_s": a.get("duration"),
        "moving_duration_s": a.get("movingDuration"),
        "elapsed_duration_s": a.get("elapsedDuration"),
        "distance_m": a.get("distance"),
        "avg_hr": a.get("averageHR"),
        "max_hr": a.get("maxHR"),
        "avg_speed_mps": a.get("averageSpeed"),
        "max_speed_mps": a.get("maxSpeed"),
        "elevation_gain_m": a.get("elevationGain"),
        "elevation_loss_m": a.get("elevationLoss"),
        "calories": a.get("calories"),
        "aerobic_te": a.get("aerobicTrainingEffect"),
        "anaerobic_te": a.get("anaerobicTrainingEffect"),
        "training_load": a.get("activityTrainingLoad"),
        "vo2max": a.get("vO2MaxValue"),
        "avg_power": a.get("avgPower"),
        "max_power": a.get("maxPower"),
        "norm_power": a.get("normPower"),
        "avg_cadence": _first(
            a,
            "averageRunningCadenceInStepsPerMinute",
            "averageBikingCadenceInRevPerMinute",
            "averageSwimCadenceInStrokesPerMinute",  # GUESS
        ),
        "max_cadence": _first(
            a,
            "maxRunningCadenceInStepsPerMinute",
            "maxBikingCadenceInRevPerMinute",
        ),
        # Time in HR zones as Garmin computed them with the zones set on the
        # watch at the time. Step 2 (personal zones) recomputes from splits.
        "hr_z1_s": a.get("hrTimeInZone_1"),
        "hr_z2_s": a.get("hrTimeInZone_2"),
        "hr_z3_s": a.get("hrTimeInZone_3"),
        "hr_z4_s": a.get("hrTimeInZone_4"),
        "hr_z5_s": a.get("hrTimeInZone_5"),
        "lactate_threshold_bpm": a.get("lactateThresholdBpm"),  # GUESS
        "device_id": a.get("deviceId"),
        "manual": int(bool(a.get("manualActivity"))),
    }


# --------------------------------------------------------------------------- #
# Daily wellness
# --------------------------------------------------------------------------- #

WELLNESS_COLUMNS = [
    "date",
    # from daily summary
    "resting_hr", "min_hr", "max_hr", "steps", "distance_m",
    "total_kcal", "active_kcal", "avg_stress", "max_stress",
    "body_battery_high", "body_battery_low",
    "body_battery_charged", "body_battery_drained",
    "intensity_min_moderate", "intensity_min_vigorous",
    "avg_spo2", "avg_waking_respiration",
    # from sleep
    "sleep_start_local", "sleep_end_local",
    "sleep_s", "deep_sleep_s", "light_sleep_s", "rem_sleep_s", "awake_s",
    "sleep_score", "sleep_avg_respiration", "sleep_avg_stress",
    # from HRV
    "hrv_last_night_avg", "hrv_last_night_5min_high", "hrv_weekly_avg",
    "hrv_status", "hrv_baseline_low", "hrv_baseline_high",
    # from max metrics / readiness / body composition
    "vo2max_running", "vo2max_cycling", "fitness_age",
    "training_readiness", "training_readiness_level",
    "weight_kg", "body_fat_pct",
]


def wellness_from_summary(s: dict | None) -> dict:
    s = s or {}
    return {
        "resting_hr": s.get("restingHeartRate"),
        "min_hr": s.get("minHeartRate"),
        "max_hr": s.get("maxHeartRate"),
        "steps": s.get("totalSteps"),
        "distance_m": s.get("totalDistanceMeters"),
        "total_kcal": s.get("totalKilocalories"),
        "active_kcal": s.get("activeKilocalories"),
        # Garmin uses -1/-2 for "not enough data"
        "avg_stress": _nonneg(s.get("averageStressLevel")),
        "max_stress": _nonneg(s.get("maxStressLevel")),
        "body_battery_high": s.get("bodyBatteryHighestValue"),
        "body_battery_low": s.get("bodyBatteryLowestValue"),
        "body_battery_charged": s.get("bodyBatteryChargedValue"),
        "body_battery_drained": s.get("bodyBatteryDrainedValue"),
        "intensity_min_moderate": s.get("moderateIntensityMinutes"),
        "intensity_min_vigorous": s.get("vigorousIntensityMinutes"),
        "avg_spo2": s.get("averageSpo2"),
        "avg_waking_respiration": s.get("avgWakingRespirationValue"),
    }


def wellness_from_sleep(sl: dict | None) -> dict:
    d = _get(sl, "dailySleepDTO") or {}
    return {
        "sleep_start_local": d.get("sleepStartTimestampLocal"),
        "sleep_end_local": d.get("sleepEndTimestampLocal"),
        "sleep_s": d.get("sleepTimeSeconds"),
        "deep_sleep_s": d.get("deepSleepSeconds"),
        "light_sleep_s": d.get("lightSleepSeconds"),
        "rem_sleep_s": d.get("remSleepSeconds"),
        "awake_s": d.get("awakeSleepSeconds"),
        "sleep_score": _get(d, "sleepScores", "overall", "value"),
        "sleep_avg_respiration": d.get("averageRespirationValue"),
        "sleep_avg_stress": d.get("avgSleepStress"),
    }


def wellness_from_hrv(h: dict | None) -> dict:
    s = _get(h, "hrvSummary") or {}
    b = s.get("baseline") or {}
    return {
        "hrv_last_night_avg": s.get("lastNightAvg"),
        "hrv_last_night_5min_high": s.get("lastNight5MinHigh"),
        "hrv_weekly_avg": s.get("weeklyAvg"),
        "hrv_status": s.get("status"),
        "hrv_baseline_low": b.get("balancedLow"),
        "hrv_baseline_high": b.get("balancedUpper"),
    }


def wellness_from_max_metrics(m: Any) -> dict:
    # Endpoint returns a list with one entry per day; generic == running.
    entry = m[0] if isinstance(m, list) and m else (m if isinstance(m, dict) else {})
    return {
        "vo2max_running": _first(entry.get("generic") or {}, "vo2MaxPreciseValue", "vo2MaxValue"),
        "vo2max_cycling": _first(entry.get("cycling") or {}, "vo2MaxPreciseValue", "vo2MaxValue"),
        "fitness_age": _get(entry, "generic", "fitnessAge"),
    }


def wellness_from_readiness(r: Any) -> dict:
    # Several entries per day are possible (recalculated after activities);
    # keep the morning one, i.e. the earliest timestamp.  # GUESS on ordering key
    entries = r if isinstance(r, list) else ([r] if isinstance(r, dict) else [])
    entries = [e for e in entries if isinstance(e, dict) and e.get("score") is not None]
    if not entries:
        return {"training_readiness": None, "training_readiness_level": None}
    entries.sort(key=lambda e: str(e.get("timestampLocal") or e.get("timestamp") or ""))
    first = entries[0]
    return {
        "training_readiness": first.get("score"),
        "training_readiness_level": first.get("level"),
    }


def weights_by_day(body_comp: dict | None) -> dict[str, dict]:
    """Map calendarDate -> {weight_kg, body_fat_pct} from a date-range payload."""
    out: dict[str, dict] = {}
    for e in (body_comp or {}).get("dateWeightList") or []:
        day = e.get("calendarDate")
        if not day:
            continue
        w = e.get("weight")  # grams
        out[day] = {
            "weight_kg": round(w / 1000, 2) if isinstance(w, (int, float)) else None,
            "body_fat_pct": e.get("bodyFat"),
        }
    return out


def _nonneg(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and v >= 0 else None


def build_wellness_row(day: str, raw: dict[str, Any], weight: dict | None = None) -> dict:
    """Merge all per-day raw payloads for ``day`` into one wellness row.

    ``raw`` maps dataset name -> payload (any may be missing).
    """
    row = {c: None for c in WELLNESS_COLUMNS}
    row["date"] = day
    row.update(wellness_from_summary(raw.get("daily_summary")))
    row.update(wellness_from_sleep(raw.get("sleep")))
    row.update(wellness_from_hrv(raw.get("hrv")))
    row.update(wellness_from_max_metrics(raw.get("max_metrics")))
    row.update(wellness_from_readiness(raw.get("training_readiness")))
    # The daily summary's RHR is sometimes missing; sleep carries one too.
    if row["resting_hr"] is None:
        row["resting_hr"] = _get(raw.get("sleep"), "restingHeartRate")
    if weight:
        row.update(weight)
    return row
