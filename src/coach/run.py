"""CLI entry point: run the virtual coach for a weekly review."""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from config.settings import settings
from src.ingestion.garmin_client import GarminClient
from src.features.training_load import compute_daily_load, compute_acwr
from src.features.trends import weekly_summary
from src.features.sleep_recovery import extract_sleep_score, compute_recovery_score
from src.coach.prompt_builder import SYSTEM_PROMPT, build_weekly_review_prompt
from src.coach.llm_client import LLMClient

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the weekly coaching review pipeline."""
    logger.info("Starting weekly coaching review...")

    # 1. Fetch data
    client = GarminClient()
    client.connect()

    end = date.today()
    start = end - timedelta(days=settings.lookback_days)

    activities = client.get_activities(start, end)
    if activities.empty:
        logger.warning("No running activities found in the last %d days", settings.lookback_days)
        print("No running activities found. Go for a run first! 🏃")
        return

    # 2. Compute features
    daily_load = compute_daily_load(activities)
    acwr_df = compute_acwr(daily_load)
    weekly = weekly_summary(activities)

    # Latest ACWR values
    latest_acwr = acwr_df.iloc[-1].to_dict() if not acwr_df.empty else {}

    # Latest week summary
    latest_week = weekly.iloc[-1].to_dict() if not weekly.empty else {}

    # Sleep/recovery (best effort — may not be available)
    sleep_recovery = None
    try:
        sleep_data = client.get_sleep_data(end)
        sleep_metrics = extract_sleep_score(sleep_data)
        hrv_data = client.get_hrv_data(end)
        hrv_value = hrv_data.get("hrvSummary", {}).get("lastNightAvg")
        recovery = compute_recovery_score(
            sleep_score=sleep_metrics.get("sleep_score"),
            hrv_status=hrv_value,
        )
        sleep_recovery = {**sleep_metrics, **recovery}
    except Exception as e:
        logger.warning("Could not fetch sleep/recovery data: %s", e)

    # 3. Build prompt
    user_prompt = build_weekly_review_prompt(
        weekly_summary=latest_week,
        acwr_data=latest_acwr,
        sleep_recovery=sleep_recovery,
    )

    logger.info("Prompt built (%d chars), calling LLM...", len(user_prompt))

    # 4. Call LLM
    llm = LLMClient()
    response = llm.generate(SYSTEM_PROMPT, user_prompt)

    # 5. Output
    print("\n" + "=" * 60)
    print("🏃 GARMIN VIRTUAL COACH — Weekly Review")
    print("=" * 60 + "\n")
    print(response)
    print("\n" + "=" * 60)

    # Save response
    output_dir = Path("data/coaching")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"review_{end.isoformat()}.md"
    with open(output_path, "w") as f:
        f.write(f"# Weekly Coaching Review — {end.isoformat()}\n\n")
        f.write(response)
    logger.info("Review saved to %s", output_path)


if __name__ == "__main__":
    main()
