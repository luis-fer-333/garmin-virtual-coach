"""Sleep and recovery scoring from Garmin sleep + HRV data."""

import logging

logger = logging.getLogger(__name__)


def extract_sleep_score(sleep_data: dict) -> dict:
    """Extract key sleep metrics from Garmin sleep API response.

    Returns:
        Dict with sleep_score, total_sleep_hours, deep_sleep_hours,
        rem_sleep_hours, and awake_minutes.
    """
    daily = sleep_data.get("dailySleepDTO", {})

    total_sec = daily.get("sleepTimeSeconds", 0) or 0
    deep_sec = daily.get("deepSleepSeconds", 0) or 0
    rem_sec = daily.get("remSleepSeconds", 0) or 0
    awake_sec = daily.get("awakeSleepSeconds", 0) or 0

    return {
        "sleep_score": daily.get("sleepScores", {}).get("overall", {}).get("value"),
        "total_sleep_hours": round(total_sec / 3600, 2),
        "deep_sleep_hours": round(deep_sec / 3600, 2),
        "rem_sleep_hours": round(rem_sec / 3600, 2),
        "awake_minutes": round(awake_sec / 60, 1),
    }


def compute_recovery_score(
    sleep_score: "float | None",
    hrv_status: "float | None",
    resting_hr: "float | None" = None,
    resting_hr_baseline: float = 55,
) -> dict:
    """Compute a simple composite recovery score (0-100).

    Combines sleep quality and HRV status. Higher = better recovered.
    This is a simplified model; Garmin's own Body Battery is more
    sophisticated but not always available via API.
    """
    components = {}
    weights = {}

    if sleep_score is not None:
        components["sleep"] = min(sleep_score, 100)
        weights["sleep"] = 0.5

    if hrv_status is not None:
        # Normalize HRV: assume baseline ~50ms, good >60ms
        hrv_score = min((hrv_status / 60) * 100, 100)
        components["hrv"] = hrv_score
        weights["hrv"] = 0.35

    if resting_hr is not None:
        # Lower resting HR = better recovery
        hr_score = max(0, 100 - (resting_hr - resting_hr_baseline) * 5)
        components["resting_hr"] = min(hr_score, 100)
        weights["resting_hr"] = 0.15

    if not weights:
        return {"recovery_score": None, "components": {}}

    total_weight = sum(weights.values())
    score = sum(
        components[k] * (weights[k] / total_weight) for k in components
    )

    return {"recovery_score": round(score, 1), "components": components}
