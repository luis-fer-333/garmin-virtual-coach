"""Build structured prompts for the LLM coach.

The prompt builder takes computed features (training load, trends, sleep)
and formats them into a context-rich prompt that the LLM can use to
generate personalized coaching advice.
"""

import json
from datetime import date

SYSTEM_PROMPT = """You are an expert running coach with deep knowledge of \
exercise physiology, periodization, and injury prevention. You analyze \
training data from a Garmin device and provide personalized, actionable \
coaching advice.

Your coaching style:
- Evidence-based, referencing training principles (progressive overload, \
  80/20 polarized training, acute:chronic workload ratio)
- Concise and direct — athletes want clear guidance, not lectures
- Encouraging but honest — flag risks (overtraining, injury) when you see them
- Always explain the "why" behind your recommendations

You receive structured training data and respond with:
1. A brief assessment of the current training week
2. Key observations (positive and concerning)
3. Specific recommendations for the next 7 days
4. One recovery or lifestyle tip based on sleep/HRV data if available

Keep responses under 500 words. Use bullet points for clarity."""


def build_weekly_review_prompt(
    weekly_summary: dict,
    acwr_data: dict,
    sleep_recovery: "dict | None" = None,
    athlete_profile: "dict | None" = None,
) -> str:
    """Build the user prompt for a weekly training review.

    Args:
        weekly_summary: Dict with weekly volume, runs, pace, etc.
        acwr_data: Dict with acute load, chronic load, ACWR ratio.
        sleep_recovery: Optional dict with sleep scores and recovery.
        athlete_profile: Optional dict with athlete info (age, goals, etc.).
    """
    sections = []

    # Athlete context
    if athlete_profile:
        sections.append(f"## Athlete Profile\n{json.dumps(athlete_profile, indent=2)}")

    sections.append(f"## Date: {date.today().isoformat()}")

    # Training volume
    sections.append(
        f"""## This Week's Training
{json.dumps(weekly_summary, indent=2, default=str)}"""
    )

    # Workload ratio
    sections.append(
        f"""## Acute:Chronic Workload Ratio
{json.dumps(acwr_data, indent=2, default=str)}"""
    )

    # Sleep and recovery
    if sleep_recovery:
        sections.append(
            f"""## Sleep & Recovery (last 7 days avg)
{json.dumps(sleep_recovery, indent=2, default=str)}"""
        )

    sections.append(
        "## Request\n"
        "Based on the data above, provide your weekly coaching review "
        "and recommendations for next week."
    )

    return "\n\n".join(sections)


def build_single_run_prompt(activity_data: dict) -> str:
    """Build a prompt for post-run analysis of a single activity."""
    return f"""## Post-Run Analysis Request

Here is the data from my latest run:
{json.dumps(activity_data, indent=2, default=str)}

Please analyze this run and tell me:
1. How was the effort relative to the intended zone?
2. Any pacing issues (positive/negative splits)?
3. One thing I did well and one thing to improve next time."""
