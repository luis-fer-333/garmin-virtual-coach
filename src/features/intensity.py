"""Intensity distribution in *your* zones (how much easy vs hard you train).

For each activity, time per zone is estimated from its laps: each lap's
duration is assigned to the zone of the lap's average HR. Lap averages smooth
out short surges, so short intervals inside a long lap are under-counted;
the per-second HR stream would be exact but costs one more Garmin call per
activity, so it is not fetched yet.

Fallbacks, per activity: no laps stored -> whole activity at its average HR
-> no HR at all -> skipped (counted in ``activities_without_hr``).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from src.athlete.profile import ZONE_NAMES, AthleteProfile

BELOW = "below Z1"


def laps_from_splits(splits: Any) -> list[tuple[float, float | None]]:
    """[(duration_s, avg_hr)] from a get_activity_splits() payload."""
    laps = (splits or {}).get("lapDTOs") if isinstance(splits, dict) else None
    out = []
    for lap in laps or []:
        dur = lap.get("duration") or lap.get("elapsedDuration")
        if isinstance(dur, (int, float)) and dur > 0:
            out.append((float(dur), lap.get("averageHR")))
    return out


def activity_zone_seconds(
    profile: AthleteProfile,
    activity: dict,
    splits: Any = None,
) -> tuple[dict[str, float], str]:
    """Seconds per personal zone for one activity, plus how it was estimated."""
    secs = {BELOW: 0.0, **{z: 0.0 for z in ZONE_NAMES}}
    laps = [(d, hr) for d, hr in laps_from_splits(splits) if hr]
    if laps:
        for dur, hr in laps:
            secs[profile.zone_for(hr)] += dur
        return secs, "laps"
    avg, dur = activity.get("averageHR"), activity.get("duration")
    if avg and dur:
        secs[profile.zone_for(avg)] += float(dur)
        return secs, "activity_avg"
    return secs, "no_hr"


def intensity_distribution(
    profile: AthleteProfile,
    activities: Iterable[dict],
    splits_by_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate minutes and % per zone over a set of activities.

    ``easy_pct`` = below Z1 + Z1 + Z2, ``moderate_pct`` = Z3,
    ``hard_pct`` = Z4 + Z5 (the 80/20 view).
    """
    splits_by_id = splits_by_id or {}
    total = {BELOW: 0.0, **{z: 0.0 for z in ZONE_NAMES}}
    how: Counter[str] = Counter()
    for a in activities:
        s, method = activity_zone_seconds(profile, a, splits_by_id.get(str(a.get("activityId"))))
        how[method] += 1
        for k, v in s.items():
            total[k] += v
    all_s = sum(total.values())
    minutes = {k: round(v / 60, 1) for k, v in total.items()}
    pct = {k: (round(100 * v / all_s, 1) if all_s else 0.0) for k, v in total.items()}
    return {
        "minutes": minutes,
        "pct": pct,
        "easy_pct": round(pct[BELOW] + pct[ZONE_NAMES[0]] + pct[ZONE_NAMES[1]], 1),
        "moderate_pct": pct[ZONE_NAMES[2]],
        "hard_pct": round(pct[ZONE_NAMES[3]] + pct[ZONE_NAMES[4]], 1),
        "activities_from_laps": how["laps"],
        "activities_from_avg_hr": how["activity_avg"],
        "activities_without_hr": how["no_hr"],
    }


def intensity_from_store(store, profile: AthleteProfile, activities_df) -> dict[str, Any]:
    """Convenience: distribution for the activities in ``activities_df`` using stored laps."""
    if activities_df is None or activities_df.empty:
        return intensity_distribution(profile, [])
    acts = activities_df.to_dict("records")
    ids = [str(a.get("activityId")) for a in acts if a.get("activityId") is not None]
    splits: dict[str, Any] = {}
    if ids:
        q = (
            "SELECT key, payload FROM raw_responses WHERE dataset='activity_splits' "
            f"AND payload IS NOT NULL AND key IN ({','.join('?' * len(ids))})"
        )
        splits = {r[0]: json.loads(r[1]) for r in store.conn.execute(q, ids)}
    return intensity_distribution(profile, acts, splits)
