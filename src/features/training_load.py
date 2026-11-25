"""Training load calculations: TRIMP, acute/chronic workload ratio (ACWR).

The acute:chronic workload ratio is a key metric for injury prevention
and performance optimization. It compares recent load (acute, ~7 days)
to longer-term load (chronic, ~28 days).

- ACWR < 0.8  → undertrained / detraining
- ACWR 0.8–1.3 → sweet spot
- ACWR > 1.5  → injury risk zone
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.athlete.profile import AthleteProfile


def compute_trimp(
    duration_minutes: float,
    avg_hr: float,
    resting_hr: float = 60,
    max_hr: float = 190,
    gender: str = "male",
) -> float:
    """Compute Training Impulse (TRIMP) for a single session.

    Uses Banister's TRIMP formula:
        TRIMP = duration * HRr * 0.64 * e^(1.92 * HRr)  [male]
        TRIMP = duration * HRr * 0.86 * e^(1.67 * HRr)  [female]

    where HRr = (avg_hr - resting_hr) / (max_hr - resting_hr)
    """
    if max_hr <= resting_hr:
        raise ValueError(f"max_hr ({max_hr}) must exceed resting_hr ({resting_hr})")
    hr_reserve = (avg_hr - resting_hr) / (max_hr - resting_hr)
    hr_reserve = np.clip(hr_reserve, 0, 1)

    if gender == "male":
        return duration_minutes * hr_reserve * 0.64 * np.exp(1.92 * hr_reserve)
    return duration_minutes * hr_reserve * 0.86 * np.exp(1.67 * hr_reserve)


def compute_daily_load(
    activities_df: pd.DataFrame,
    profile: "AthleteProfile | None" = None,
    max_hr: "float | None" = None,
    resting_hr: "float | None" = None,
    gender: "str | None" = None,
) -> pd.DataFrame:
    """Compute daily TRIMP from a DataFrame of activities.

    Expects columns: 'startTimeLocal' (or 'date'), 'duration' (seconds),
    'averageHR'.

    HR parameters come from ``profile`` (see ``src.athlete``): its max HR,
    gender and the resting HR you had around each activity's date. Explicit
    ``max_hr`` / ``resting_hr`` / ``gender`` win over the profile; without
    either, the old 190/60/male defaults apply.

    Activities without an average HR contribute 0 TRIMP (they used to be
    scored as if at 140 bpm). ``n_no_hr`` counts them per day.

    Returns:
        DataFrame with columns ['date', 'daily_trimp', 'n_no_hr'].
    """
    if activities_df.empty:
        return pd.DataFrame(columns=["date", "daily_trimp", "n_no_hr"])

    df = activities_df.copy()

    # Normalize date column
    if "startTimeLocal" in df.columns:
        df["date"] = pd.to_datetime(df["startTimeLocal"]).dt.date
    elif "date" not in df.columns:
        raise ValueError("Activities must have 'startTimeLocal' or 'date' column")

    mx = max_hr or (profile.max_hr if profile else 190)
    sex = gender or (profile.gender if profile else "male")

    def rest_on(day) -> float:
        if resting_hr is not None:
            return resting_hr
        return profile.resting_hr_on(str(day)) if profile else 60

    if "averageHR" not in df.columns:
        df["averageHR"] = np.nan
    df["duration_min"] = df["duration"] / 60.0
    df["no_hr"] = df["averageHR"].isna()
    df["trimp"] = df.apply(
        lambda r: 0.0 if r["no_hr"] else compute_trimp(
            duration_minutes=r["duration_min"],
            avg_hr=r["averageHR"],
            resting_hr=rest_on(r["date"]),
            max_hr=mx,
            gender=sex,
        ),
        axis=1,
    )

    daily = df.groupby("date").agg(
        daily_trimp=("trimp", "sum"), n_no_hr=("no_hr", "sum")
    ).reset_index()
    return daily.sort_values("date")


def compute_acwr(
    daily_load: pd.DataFrame,
    acute_window: int = 7,
    chronic_window: int = 28,
    end: "date | str | None" = None,
) -> pd.DataFrame:
    """Compute Acute:Chronic Workload Ratio using EWMA.

    Args:
        daily_load: DataFrame with ['date', 'daily_trimp'].
        acute_window: Days for acute load (default 7).
        chronic_window: Days for chronic load (default 28).
        end: Extend the series with rest days up to this date (e.g. today),
            so the latest row reflects days off since the last session.

    Returns:
        DataFrame with added columns: 'acute_load', 'chronic_load', 'acwr'.
    """
    df = daily_load.copy().sort_values("date")

    # Rest days must count as zero load, otherwise the EWMA only "sees"
    # training days and both loads are inflated. Fill the calendar.
    if not df.empty:
        dates = pd.to_datetime(df["date"])
        last = max(dates.max(), pd.Timestamp(end)) if end is not None else dates.max()
        full = pd.date_range(dates.min(), last, freq="D")
        df = (
            df.assign(date=dates).set_index("date").reindex(full)
            .rename_axis("date").reset_index()
        )
        df["daily_trimp"] = df["daily_trimp"].fillna(0.0)
        for col in df.columns.difference(["date", "daily_trimp"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
        df["date"] = df["date"].dt.date

    df["acute_load"] = (
        df["daily_trimp"].ewm(span=acute_window, min_periods=1).mean()
    )
    df["chronic_load"] = (
        df["daily_trimp"].ewm(span=chronic_window, min_periods=1).mean()
    )
    df["acwr"] = df["acute_load"] / df["chronic_load"].replace(0, np.nan)

    return df
