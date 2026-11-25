"""CLI entry point: run the virtual coach for a weekly review."""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from config.settings import settings
from src.ingestion.garmin_client import GarminClient
from src.ingestion.sync import GarminSync
from src.storage.local_store import LocalStore
from src.athlete import load_profile
from src.features.intensity import intensity_from_store
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

    # 1. Sync new Garmin data into the local store, then read from it
    client = GarminClient()
    client.connect()
    store = LocalStore()
    stats = GarminSync(client.api, store).daily(details=False)
    if stats.stopped_reason:
        logger.warning("Sync stopped early: %s (using stored data)", stats.stopped_reason)

    end = date.today()
    start = end - timedelta(days=settings.lookback_days)
    # Chronic load needs a few months of history to settle (EWMA span 28d)
    load_start = end - timedelta(days=max(settings.lookback_days, 120))

    profile = load_profile(store, today=end)
    logger.info(
        "Athlete profile: max HR %s (%s), resting HR %s (%s), zones by %s",
        profile.max_hr, profile.sources.get("max_hr"),
        profile.resting_hr, profile.sources.get("resting_hr"), profile.zone_method,
    )
    for note in profile.notes:
        logger.info("Profile note: %s", note)

    run_sports = [
        r[0] for r in store.conn.execute("SELECT DISTINCT sport FROM activities")
        if r[0] and "run" in r[0]
    ]
    history = store.activities_df(sport=run_sports, since=load_start.isoformat()) if run_sports else None
    if history is None or history.empty:
        logger.warning("No running activities found in the last %d days", (end - load_start).days)
        print("No running activities found. Go for a run first! 🏃")
        return
    recent = history[history["startTimeLocal"].str[:10] >= start.isoformat()]
    if recent.empty:
        logger.warning("No running activities found in the last %d days", settings.lookback_days)
        print("No running activities found. Go for a run first! 🏃")
        return

    # 2. Compute features with personal HR values
    daily_load = compute_daily_load(history, profile=profile)
    acwr_df = compute_acwr(daily_load, end=end)
    weekly = weekly_summary(recent)
    intensity = intensity_from_store(
        store, profile, recent[recent["startTimeLocal"].str[:10] >= (end - timedelta(days=28)).isoformat()]
    )

    # Latest ACWR values
    latest_acwr = acwr_df.iloc[-1].to_dict() if not acwr_df.empty else {}

    # Latest week summary
    latest_week = weekly.iloc[-1].to_dict() if not weekly.empty else {}

    # Sleep/recovery from stored payloads (last night may not be in yet)
    sleep_recovery = None
    for day in (end, end - timedelta(days=1)):
        sleep_data = store.get_raw("sleep", day.isoformat())
        if sleep_data:
            sleep_metrics = extract_sleep_score(sleep_data)
            hrv_data = store.get_raw("hrv", day.isoformat()) or {}
            hrv_value = (hrv_data.get("hrvSummary") or {}).get("lastNightAvg")
            rhr_row = store.conn.execute(
                "SELECT resting_hr FROM daily_wellness WHERE date=?", (day.isoformat(),)
            ).fetchone()
            recovery = compute_recovery_score(
                sleep_score=sleep_metrics.get("sleep_score"),
                hrv_status=hrv_value,
                resting_hr=rhr_row[0] if rhr_row else None,
                resting_hr_baseline=profile.resting_hr,
                hrv_baseline=profile.hrv_baseline_ms,
            )
            sleep_recovery = {**sleep_metrics, **recovery}
            break

    # 3. Build prompt
    user_prompt = build_weekly_review_prompt(
        weekly_summary=latest_week,
        acwr_data=latest_acwr,
        sleep_recovery=sleep_recovery,
        athlete_profile=profile.to_prompt_dict(),
        intensity_distribution=intensity,
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
