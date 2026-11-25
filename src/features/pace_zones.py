"""HR zone classification and time-in-zone analysis.

Pass an ``AthleteProfile`` (``src.athlete.load_profile``) to use your personal
zones in bpm. Without one, the old generic % of max HR zones apply.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# Default zones as % of max HR (can be customized per athlete)
DEFAULT_HR_ZONES = {
    "Z1 - Recovery": (0.50, 0.60),
    "Z2 - Easy Aerobic": (0.60, 0.70),
    "Z3 - Aerobic": (0.70, 0.80),
    "Z4 - Threshold": (0.80, 0.90),
    "Z5 - VO2max": (0.90, 1.00),
}


def classify_hr_zone(
    hr: float,
    max_hr: float = 190,
    zones: "dict | None" = None,
    profile=None,
) -> str:
    """Classify a heart rate value into a training zone.

    With ``profile``, returns the profile's zone name (or "below Z1").
    """
    if profile is not None:
        return profile.zone_for(hr)
    zones = zones or DEFAULT_HR_ZONES
    hr_pct = hr / max_hr

    for zone_name, (low, high) in zones.items():
        if low <= hr_pct < high:
            return zone_name
    return "Z5 - VO2max" if hr_pct >= 1.0 else "Z1 - Recovery"


def time_in_zones(
    hr_series: pd.Series,
    max_hr: float = 190,
    sample_interval_sec: float = 1.0,
    profile=None,
) -> dict[str, float]:
    """Compute time spent in each HR zone (in minutes).

    Args:
        hr_series: Series of heart rate values (one per sample).
        max_hr: Athlete's maximum heart rate (ignored with ``profile``).
        sample_interval_sec: Time between samples in seconds.
        profile: Optional ``AthleteProfile`` for personal bpm zones.

    Returns:
        Dict mapping zone name to minutes spent in that zone.
    """
    if profile is not None:
        zones = {"below Z1": 0.0, **{z.name: 0.0 for z in profile.hr_zones}}
    else:
        zones = {name: 0.0 for name in DEFAULT_HR_ZONES}
    for hr in hr_series.dropna():
        zone = classify_hr_zone(hr, max_hr, profile=profile)
        zones[zone] += sample_interval_sec / 60.0
    return zones
