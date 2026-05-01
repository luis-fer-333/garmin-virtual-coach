"""Tests for training load calculations."""

import numpy as np
import pandas as pd
import pytest

from src.features.training_load import compute_trimp, compute_daily_load, compute_acwr


class TestComputeTrimp:
    def test_moderate_effort(self):
        """A 30-min run at 150bpm with default HR params."""
        trimp = compute_trimp(duration_minutes=30, avg_hr=150)
        assert trimp > 0
        assert trimp < 200  # Sanity check

    def test_zero_duration(self):
        trimp = compute_trimp(duration_minutes=0, avg_hr=150)
        assert trimp == 0.0

    def test_resting_hr_equals_avg(self):
        """If avg HR equals resting HR, TRIMP should be ~0."""
        trimp = compute_trimp(duration_minutes=30, avg_hr=60, resting_hr=60)
        assert trimp == pytest.approx(0.0, abs=0.01)

    def test_higher_hr_gives_higher_trimp(self):
        low = compute_trimp(duration_minutes=30, avg_hr=130)
        high = compute_trimp(duration_minutes=30, avg_hr=170)
        assert high > low

    def test_female_coefficient(self):
        male = compute_trimp(duration_minutes=30, avg_hr=150, gender="male")
        female = compute_trimp(duration_minutes=30, avg_hr=150, gender="female")
        # Different coefficients, both should be positive
        assert male > 0
        assert female > 0


class TestComputeACWR:
    def test_basic_acwr(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        daily = pd.DataFrame({
            "date": dates,
            "daily_trimp": np.random.uniform(30, 80, size=30),
        })
        result = compute_acwr(daily)
        assert "acwr" in result.columns
        assert "acute_load" in result.columns
        assert "chronic_load" in result.columns
        assert len(result) == 30
