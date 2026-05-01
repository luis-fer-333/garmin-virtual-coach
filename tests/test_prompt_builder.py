"""Tests for prompt builder."""

from src.coach.prompt_builder import (
    build_weekly_review_prompt,
    build_single_run_prompt,
    SYSTEM_PROMPT,
)


def test_system_prompt_exists():
    assert len(SYSTEM_PROMPT) > 100
    assert "running coach" in SYSTEM_PROMPT.lower()


def test_weekly_review_prompt_structure():
    prompt = build_weekly_review_prompt(
        weekly_summary={"num_runs": 4, "total_distance_km": 32.5},
        acwr_data={"acwr": 1.1, "acute_load": 55, "chronic_load": 50},
        sleep_recovery={"sleep_score": 78, "recovery_score": 72.3},
    )
    assert "Training" in prompt
    assert "Workload" in prompt
    assert "Sleep" in prompt
    assert "32.5" in prompt


def test_weekly_review_without_sleep():
    prompt = build_weekly_review_prompt(
        weekly_summary={"num_runs": 3},
        acwr_data={"acwr": 0.9},
        sleep_recovery=None,
    )
    assert "Sleep" not in prompt
    assert "Training" in prompt


def test_single_run_prompt():
    prompt = build_single_run_prompt({"distance": 10000, "duration": 3000})
    assert "Post-Run" in prompt
    assert "10000" in prompt
