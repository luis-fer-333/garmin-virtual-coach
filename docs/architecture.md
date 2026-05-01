# Architecture Decisions

## Overview

The Garmin Virtual Coach follows a pipeline architecture with clear separation
between data ingestion, feature computation, and LLM-based coaching.

## Design Principles

1. **Offline-first** — All data is stored locally after ingestion. The coach
   can run without an active Garmin connection (using cached data).
2. **LLM-agnostic** — The `LLMClient` abstraction supports both OpenAI and
   AWS Bedrock. Adding new providers requires only a new method.
3. **Feature layer does the thinking** — The LLM receives pre-computed
   analytics (ACWR, trends, recovery scores), not raw data. This keeps
   prompts focused and token-efficient.
4. **Progressive enhancement** — V1 is CLI + local SQLite. The same code
   can be deployed as a Lambda behind API Gateway with DynamoDB.

## Data Flow

```
Garmin Connect API
       │
       ▼
  pull_activities.py  ──▶  data/raw/*.json  (raw cache)
       │
       ▼
  Feature Engine
  ├── training_load.py   → TRIMP, ACWR
  ├── pace_zones.py      → time-in-zone
  ├── trends.py          → weekly volume
  └── sleep_recovery.py  → recovery score
       │
       ▼
  prompt_builder.py  ──▶  Structured prompt (JSON sections)
       │
       ▼
  llm_client.py  ──▶  OpenAI / Bedrock
       │
       ▼
  Coach response  ──▶  Terminal / API / Bot
```

## Key Metrics Explained

### TRIMP (Training Impulse)
Banister's formula combining duration and heart rate intensity.
Higher TRIMP = harder session.

### ACWR (Acute:Chronic Workload Ratio)
- Acute = last 7 days (EWMA)
- Chronic = last 28 days (EWMA)
- Sweet spot: 0.8–1.3
- Danger zone: >1.5 (injury risk)

### Recovery Score
Composite of sleep quality (50%), HRV (35%), resting HR (15%).
Simplified model — Garmin's Body Battery is more sophisticated
but not reliably available via the unofficial API.

## Cloud Deployment Path (V2)

```
API Gateway → Lambda → DynamoDB (storage)
                  ↓
              Bedrock (LLM)
                  ↓
              S3 (raw data archive)
```

## Security Considerations

- Garmin credentials stored in `.env` (never committed)
- Auth tokens cached in `~/.garminconnect/` with restricted permissions
- Health data is sensitive — multi-user version needs encryption at rest
  and proper auth (OAuth2 / Cognito)
