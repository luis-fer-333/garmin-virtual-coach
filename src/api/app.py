"""FastAPI application for the Garmin Virtual Coach."""

import logging
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings
from src.ingestion.garmin_client import GarminClient
from src.features.training_load import compute_daily_load, compute_acwr
from src.features.trends import weekly_summary
from src.features.sleep_recovery import extract_sleep_score, compute_recovery_score
from src.coach.prompt_builder import SYSTEM_PROMPT, build_weekly_review_prompt
from src.coach.llm_client import LLMClient
from src.storage.local_store import LocalStore

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Garmin Virtual Coach",
    description="AI-powered running coach using Garmin data",
    version="0.1.0",
)

store = LocalStore()


class CoachingResponse(BaseModel):
    date: str
    review: str
    acwr: "float | None" = None
    weekly_distance_km: "float | None" = None
    recovery_score: "float | None" = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/coach/weekly", response_model=CoachingResponse)
def weekly_review():
    """Generate a weekly coaching review from recent Garmin data."""
    try:
        client = GarminClient()
        client.connect()

        end = date.today()
        start = end - timedelta(days=settings.lookback_days)

        activities = client.get_activities(start, end)
        if activities.empty:
            raise HTTPException(404, "No running activities found")

        daily_load = compute_daily_load(activities)
        acwr_df = compute_acwr(daily_load)
        weekly = weekly_summary(activities)

        latest_acwr = acwr_df.iloc[-1].to_dict() if not acwr_df.empty else {}
        latest_week = weekly.iloc[-1].to_dict() if not weekly.empty else {}

        # Sleep/recovery (best effort)
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
            logger.warning("Sleep/recovery data unavailable: %s", e)

        # Build prompt and call LLM
        user_prompt = build_weekly_review_prompt(
            weekly_summary=latest_week,
            acwr_data=latest_acwr,
            sleep_recovery=sleep_recovery,
        )
        llm = LLMClient()
        review_text = llm.generate(SYSTEM_PROMPT, user_prompt)

        # Persist
        store.save_review(end.isoformat(), user_prompt, review_text)

        return CoachingResponse(
            date=end.isoformat(),
            review=review_text,
            acwr=latest_acwr.get("acwr"),
            weekly_distance_km=latest_week.get("total_distance_km"),
            recovery_score=(
                sleep_recovery.get("recovery_score") if sleep_recovery else None
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating weekly review")
        raise HTTPException(500, f"Internal error: {e}")


@app.get("/coach/history")
def coaching_history(limit: int = 10):
    """Retrieve past coaching reviews."""
    return store.get_reviews(limit=limit)
