"""Weekly and monthly volume trends for training progression analysis."""

import pandas as pd


def weekly_summary(activities_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly aggregates from activities.

    Expects columns: 'startTimeLocal', 'distance' (meters), 'duration' (seconds),
    'averageHR', 'averageSpeed' (m/s).

    Returns:
        DataFrame with weekly totals and averages.
    """
    if activities_df.empty:
        return pd.DataFrame()

    df = activities_df.copy()
    df["date"] = pd.to_datetime(df["startTimeLocal"])
    df["week"] = df["date"].dt.isocalendar().week
    df["year"] = df["date"].dt.isocalendar().year

    weekly = df.groupby(["year", "week"]).agg(
        num_runs=("distance", "count"),
        total_distance_km=("distance", lambda x: x.sum() / 1000),
        total_duration_min=("duration", lambda x: x.sum() / 60),
        avg_pace_min_km=(
            "averageSpeed",
            lambda x: (1000 / 60) / x.mean() if x.mean() > 0 else None,
        ),
        avg_hr=("averageHR", "mean"),
    ).reset_index()

    return weekly.sort_values(["year", "week"])


def week_over_week_change(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Add week-over-week percentage change columns."""
    df = weekly_df.copy()
    df["distance_change_pct"] = df["total_distance_km"].pct_change() * 100
    df["duration_change_pct"] = df["total_duration_min"].pct_change() * 100
    return df
