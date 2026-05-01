"""Pace zone classification and time-in-zone analysis.

Zones are based on percentage of threshold pace (or max HR).
Default 5-zone model aligned with Garmin's zone system.
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
    hr: float, max_hr: float = 190, zones: "dict | None" = None
) -> str:
    """Classify a heart rate value into a training zone."""
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
) -> dict[str, float]:
    """Compute time spent in each HR zone (in minutes).

    Args:
        hr_series: Series of heart rate values (one per sample).
        max_hr: Athlete's maximum heart rate.
        sample_interval_sec: Time between samples in seconds.

    Returns:
        Dict mapping zone name to minutes spent in that zone.
    """
    zones = {name: 0.0 for name in DEFAULT_HR_ZONES}
    for hr in hr_series.dropna():
        zone = classify_hr_zone(hr, max_hr)
        zones[zone] += sample_interval_sec / 60.0
    return zones
