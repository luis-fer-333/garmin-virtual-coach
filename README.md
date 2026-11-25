# Garmin Virtual Coach 🏃‍♂️🤖

AI-powered virtual running coach that ingests raw Garmin data and delivers personalized training insights using LLMs.

## Overview

This project connects to Garmin Connect, extracts training and health metrics, computes derived analytics (training load, fatigue, pace zones), and feeds structured context to an LLM that acts as a personalized virtual coach.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Garmin Connect  │────▶│  Data Ingestion   │────▶│   Raw Storage    │
│  (API / .fit)    │     │  (Python client)  │     │   (S3 / local)   │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Coach Response  │◀────│   LLM Engine      │◀────│  Feature Engine  │
│  (natural lang)  │     │  (Bedrock/OpenAI) │     │  (analytics)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

### Components

1. **Data Ingestion** (`src/ingestion/`) — Pulls activities, daily stats, sleep, and HRV from Garmin Connect API
2. **Feature Engine** (`src/features/`) — Computes training load (TRIMP/EWMA), pace zones, weekly volume trends, fatigue ratios
3. **LLM Coach** (`src/coach/`) — Builds structured prompts with athlete context + recent metrics, calls LLM for coaching advice
4. **API / Interface** (`src/api/`) — FastAPI service exposing coaching endpoints
5. **Storage** (`src/storage/`) — Abstraction layer for local (SQLite/JSON) and cloud (S3/DynamoDB) storage

## Quick Start

```bash
# 1. Clone and set up environment
cd garmin-virtual-coach
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env with your Garmin Connect credentials and LLM API key

# 4. Pull your full history into SQLite (resumable; see "Syncing Garmin data")
python -m src.ingestion.sync backfill

# 5. Run the coach
python -m src.coach.run
```

## Syncing Garmin data

All Garmin data lands in one SQLite file (`LOCAL_DB_PATH`, default `data/coach.db`).

```bash
python -m src.ingestion.sync backfill                # everything since your first activity
python -m src.ingestion.sync backfill --max-days 365 # cap a single run (resume later)
python -m src.ingestion.sync backfill --since 2019-01-01
python -m src.ingestion.sync daily                   # incremental: new activities + last 14 days
python -m src.ingestion.sync status                  # counts and date ranges
python -m src.ingestion.sync rebuild                 # re-derive tables from raw payloads, no Garmin calls
```

**What is fetched**

| Dataset | Endpoint | Cost |
|---|---|---|
| Activities, every sport | activity list, 100 per page | ~1 call / 100 activities |
| Laps + time in HR zones | per activity | 2 calls / activity (`--no-details` to skip) |
| Daily summary (RHR, steps, stress, Body Battery, intensity min, SpO2) | per day | 1 call / day |
| Sleep, HRV, VO2max/fitness age, training readiness | per day | 4 calls / day |
| Weight / body fat | per year | 1 call / year |
| Profile, user settings, HR zones, lactate threshold | snapshot per sync day | 4 calls |

**Tables**

- `raw_responses` — every payload exactly as Garmin returned it, keyed by `(dataset, key)`, with a status (`ok` / `empty` / `error`). It is also the sync ledger: a day counts as done once each dataset is `ok` or `empty`. Keys that fail 3 times are skipped.
- `activities` — one typed row per activity (sport, duration, distance, HR, pace, power, cadence, training effect/load, time in Garmin HR zones…).
- `daily_wellness` — one typed row per day (resting HR, HRV + baseline, sleep stages + score, Body Battery, stress, VO2max, readiness, weight…).
- `coaching_reviews` — saved LLM reviews.

Column mapping lives in `src/ingestion/normalize.py`. If a column comes out empty for your watch, fix the field name there and run `rebuild`.

**Rate limits.** Requests are spaced by `GARMIN_REQUEST_INTERVAL` (default 1.5 s). On HTTP 429 the sync backs off (1, 5, 15 min) and then stops cleanly. Progress is committed per day, so just re-run. Roughly: 1 year of history is about 1,825 wellness calls, so around 50 minutes.

**Running it daily** (example cron line, 06:30 every day):

```cron
30 6 * * * cd /path/to/garmin-virtual-coach && .venv/bin/python -m src.ingestion.sync daily >> data/sync.log 2>&1
```

The daily run re-fetches the last 3 days because sleep, HRV and VO2max for a day are often finalized the next morning.

## Project Structure

```
garmin-virtual-coach/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py              # Centralized configuration
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── garmin_client.py     # Garmin Connect API wrapper
│   │   ├── fit_parser.py        # .fit file parser (offline fallback)
│   │   └── pull_activities.py   # CLI entry point for data pull
│   ├── features/
│   │   ├── __init__.py
│   │   ├── training_load.py     # TRIMP, acute/chronic workload ratio
│   │   ├── pace_zones.py        # Zone classification and time-in-zone
│   │   ├── trends.py            # Weekly/monthly volume and progression
│   │   └── sleep_recovery.py    # Sleep score + HRV-based recovery
│   ├── coach/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py    # Builds structured LLM prompts
│   │   ├── llm_client.py        # LLM API abstraction (Bedrock/OpenAI)
│   │   └── run.py               # CLI entry point for coaching
│   ├── storage/
│   │   ├── __init__.py
│   │   └── local_store.py       # SQLite/JSON local storage
│   └── api/
│       ├── __init__.py
│       └── app.py               # FastAPI application
├── tests/
│   ├── __init__.py
│   ├── test_garmin_client.py
│   ├── test_training_load.py
│   └── test_prompt_builder.py
├── notebooks/
│   └── exploration.ipynb        # Data exploration and feature prototyping
└── docs/
    └── architecture.md          # Detailed architecture decisions
```

## V1 Scope (Running Only)

- **Input:** Last 30 days of running activities + daily stats + sleep
- **Features:** Weekly mileage trend, avg pace by zone, training load (acute vs chronic), sleep/recovery score
- **Output:** Weekly training summary + next week recommendation in natural language

## Future Enhancements

- [ ] Multi-sport support (cycling, swimming)
- [ ] Telegram/WhatsApp bot interface
- [ ] Race prediction model
- [ ] Injury risk scoring based on load spikes
- [ ] Cloud deployment (Lambda + API Gateway + DynamoDB)
- [ ] User auth + multi-user support

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Garmin API | `garminconnect` (unofficial) |
| Analytics | pandas, numpy |
| LLM | OpenAI API / AWS Bedrock |
| API | FastAPI |
| Storage | SQLite (local) → DynamoDB (cloud) |
| Testing | pytest |

## License

MIT
