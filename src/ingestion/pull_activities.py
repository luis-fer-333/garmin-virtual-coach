"""CLI entry point: pull activities from Garmin Connect and store locally."""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from config.settings import settings
from src.ingestion.garmin_client import GarminClient

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def main() -> None:
    """Pull recent activities and daily stats, save to local JSON."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    client = GarminClient()
    client.connect()

    end = date.today()
    start = end - timedelta(days=settings.lookback_days)

    # Pull activities
    activities_df = client.get_activities(start, end)
    activities_path = data_dir / f"activities_{start}_{end}.json"
    activities_df.to_json(activities_path, orient="records", indent=2)
    logger.info("Saved %d activities to %s", len(activities_df), activities_path)

    # Pull daily stats and sleep for each day
    daily_stats = []
    sleep_data = []
    for day_offset in range(settings.lookback_days):
        day = end - timedelta(days=day_offset)
        try:
            stats = client.get_daily_stats(day)
            stats["date"] = day.isoformat()
            daily_stats.append(stats)
        except Exception as e:
            logger.warning("Failed to get stats for %s: %s", day, e)
        try:
            sleep = client.get_sleep_data(day)
            sleep["date"] = day.isoformat()
            sleep_data.append(sleep)
        except Exception as e:
            logger.warning("Failed to get sleep for %s: %s", day, e)

    stats_path = data_dir / f"daily_stats_{start}_{end}.json"
    with open(stats_path, "w") as f:
        json.dump(daily_stats, f, indent=2, default=str)
    logger.info("Saved %d daily stat records to %s", len(daily_stats), stats_path)

    sleep_path = data_dir / f"sleep_{start}_{end}.json"
    with open(sleep_path, "w") as f:
        json.dump(sleep_data, f, indent=2, default=str)
    logger.info("Saved %d sleep records to %s", len(sleep_data), sleep_path)


if __name__ == "__main__":
    main()
