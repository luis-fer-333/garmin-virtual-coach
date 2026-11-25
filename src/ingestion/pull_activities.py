"""Legacy entry point, kept for the README's old instructions.

Data now lives in SQLite (``LOCAL_DB_PATH``). This runs the incremental
sync; use ``python -m src.ingestion.sync backfill`` for full history.
"""

from src.ingestion.sync import main

if __name__ == "__main__":
    raise SystemExit(main(["daily"]))
