"""Training load calculations: TRIMP, acute/chronic workload ratio (ACWR).

The acute:chronic workload ratio is a key metric for injury prevention
and performance optimization. It compares recent load (acute, ~7 days)
to longer-term load (chronic, ~28 days).

- ACWR < 0.8  → undertrained / detraining
- ACWR 0.8–1.3 → sweet spot
- ACWR > 1.5  → injury risk zone
"""

import pandas as pd
import numpy as np


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
    hr_reserve = (avg_hr - resting_hr) / (max_hr - resting_hr)
    hr_reserve = np.clip(hr_reserve, 0, 1)

    if gender == "male":
        return duration_minutes * hr_reserve * 0.64 * np.exp(1.92 * hr_reserve)
    return duration_minutes * hr_reserve * 0.86 * np.exp(1.67 * hr_reserve)


def compute_daily_load(activities_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily TRIMP from a DataFrame of activities.

    Expects columns: 'startTimeLocal' (or 'date'), 'duration' (seconds),
    'averageHR', and optionally 'maxHR'.

    Returns:
        DataFrame with columns ['date', 'daily_trimp'].
    """
    if activities_df.empty:
        return pd.DataFrame(columns=["date", "daily_trimp"])

    df = activities_df.copy()

    # Normalize date column
    if "startTimeLocal" in df.columns:
        df["date"] = pd.to_datetime(df["startTimeLocal"]).dt.date
    elif "date" not in df.columns:
        raise ValueError("Activities must have 'startTimeLocal' or 'date' column")

    df["duration_min"] = df["duration"] / 60.0
    df["trimp"] = df.apply(
        lambda r: compute_trimp(
            duration_minutes=r["duration_min"],
            avg_hr=r.get("averageHR", 140),
        ),
        axis=1,
    )

    daily = df.groupby("date")["trimp"].sum().reset_index()
    daily.columns = ["date", "daily_trimp"]
    return daily.sort_values("date")


def compute_acwr(
    daily_load: pd.DataFrame,
    acute_window: int = 7,
    chronic_window: int = 28,
) -> pd.DataFrame:
    """Compute Acute:Chronic Workload Ratio using EWMA.

    Args:
        daily_load: DataFrame with ['date', 'daily_trimp'].
        acute_window: Days for acute load (default 7).
        chronic_window: Days for chronic load (default 28).

    Returns:
        DataFrame with added columns: 'acute_load', 'chronic_load', 'acwr'.
    """
    df = daily_load.copy().sort_values("date")

    df["acute_load"] = (
        df["daily_trimp"].ewm(span=acute_window, min_periods=1).mean()
    )
    df["chronic_load"] = (
        df["daily_trimp"].ewm(span=chronic_window, min_periods=1).mean()
    )
    df["acwr"] = df["acute_load"] / df["chronic_load"].replace(0, np.nan)

    return df
