"""Parse .fit files exported directly from a Garmin device.

This is the offline fallback when the Connect API is unavailable.
Uses the fitparse library to extract records from .fit binary files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fitparse import FitFile

logger = logging.getLogger(__name__)


def parse_fit_file(filepath: str | Path) -> pd.DataFrame:
    """Parse a single .fit file into a DataFrame of records.

    Args:
        filepath: Path to the .fit file.

    Returns:
        DataFrame with columns for each field in the record messages
        (timestamp, heart_rate, speed, cadence, position, etc.).
    """
    fit = FitFile(str(filepath))
    records = []

    for record in fit.get_messages("record"):
        row = {}
        for field in record:
            row[field.name] = field.value
        records.append(row)

    df = pd.DataFrame(records)
    logger.info("Parsed %d records from %s", len(df), filepath)
    return df


def parse_fit_directory(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Parse all .fit files in a directory.

    Returns:
        Dict mapping filename to its parsed DataFrame.
    """
    directory = Path(directory)
    results = {}
    for fit_path in sorted(directory.glob("*.fit")):
        results[fit_path.name] = parse_fit_file(fit_path)
    logger.info("Parsed %d .fit files from %s", len(results), directory)
    return results
