"""Garmin Connect API client wrapper.

Handles authentication (with token caching) and data extraction
for activities, daily summaries, sleep, and HRV.
"""

from __future__ import annotations

import getpass
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from garminconnect import Garmin
from garminconnect.exceptions import GarminConnectAuthenticationError

from config.settings import settings

logger = logging.getLogger(__name__)


class GarminClient:
    """Thin wrapper around the garminconnect library with token persistence."""

    def __init__(self) -> None:
        self.token_dir = Path(settings.garmin_token_dir).expanduser()
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self._api: Garmin | None = None

    def connect(self) -> None:
        """Authenticate to Garmin Connect, reusing saved tokens when possible.

        Order: cached tokens in ``token_dir`` -> GARMIN_EMAIL/GARMIN_PASSWORD
        (env or .env) -> interactive prompt (password hidden via getpass).
        garminconnect>=0.3 saves fresh tokens to ``token_dir`` after a
        credential login, so later runs (e.g. a scheduled daily sync) need
        no password.
        """
        email = settings.garmin_email or None
        password = settings.garmin_password or None
        try:
            self._login(email, password)
        except GarminConnectAuthenticationError as e:
            if email and password:
                raise
            if not sys.stdin or not sys.stdin.isatty():
                raise GarminConnectAuthenticationError(
                    f"No saved Garmin tokens in {self.token_dir} and no "
                    "GARMIN_EMAIL/GARMIN_PASSWORD set. Run the sync once "
                    "interactively to log in and save tokens."
                ) from e
            print(f"No saved Garmin tokens in {self.token_dir}; logging in once.")
            email = email or input("Garmin email: ").strip()
            password = password or getpass.getpass("Garmin password: ")
            self._login(email, password)
        logger.info("Logged in to Garmin Connect (tokens in %s)", self.token_dir)

    def _login(self, email: str | None, password: str | None) -> None:
        self._api = Garmin(
            email,
            password,
            prompt_mfa=lambda: input("Garmin MFA code: ").strip(),
        )
        self._api.login(str(self.token_dir))
        garth = getattr(self._api, "garth", None)
        if garth is not None:  # garminconnect < 0.3
            garth.dump(str(self.token_dir))

    @property
    def api(self) -> Garmin:
        if self._api is None:
            self.connect()
        return self._api

    def get_activities(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        """Fetch running activities within a date range.

        Args:
            start_date: Start of range (default: 30 days ago).
            end_date: End of range (default: today).

        Returns:
            DataFrame with one row per activity.
        """
        end_date = end_date or date.today()
        start_date = start_date or end_date - timedelta(days=settings.lookback_days)

        activities = self.api.get_activities_by_date(
            start_date.isoformat(), end_date.isoformat(), "running"
        )
        df = pd.DataFrame(activities)
        logger.info(
            "Fetched %d running activities (%s to %s)",
            len(df), start_date, end_date,
        )
        return df

    def get_daily_stats(self, day: date | None = None) -> dict:
        """Fetch daily summary stats (steps, calories, stress, etc.)."""
        day = day or date.today()
        return self.api.get_stats(day.isoformat())

    def get_sleep_data(self, day: date | None = None) -> dict:
        """Fetch sleep data for a given night."""
        day = day or date.today()
        return self.api.get_sleep_data(day.isoformat())

    def get_hrv_data(self, day: date | None = None) -> dict:
        """Fetch heart rate variability data."""
        day = day or date.today()
        return self.api.get_hrv_data(day.isoformat())

    def get_training_status(self) -> dict:
        """Fetch current training status (VO2max, training load, etc.)."""
        return self.api.get_training_status(date.today().isoformat())
