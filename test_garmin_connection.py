"""Standalone Garmin Connect test — v0.3.3+ (new JWT auth).

Usage:
    1. Create a .env file with GARMIN_EMAIL and GARMIN_PASSWORD
    2. Activate the venv: source .venv/bin/activate
    3. Run: python test_garmin_connection.py

This script tests the connection, pulls recent data, and saves it locally
so you can explore what fields are available before building features.
"""

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

# Load .env manually (no extra dependency needed)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from garminconnect import Garmin

EMAIL = os.environ.get("GARMIN_EMAIL", "")
PASSWORD = os.environ.get("GARMIN_PASSWORD", "")
TOKEN_DIR = str(Path.home() / ".garminconnect")


def save_json(data, filename):
    """Save data to data/test/ as pretty JSON."""
    out_dir = Path("data/test")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  💾 Saved to {path}")


def main():
    if not EMAIL or not PASSWORD:
        print("❌ Set GARMIN_EMAIL and GARMIN_PASSWORD in .env first")
        return

    print(f"📧 Connecting as: {EMAIL}")
    print("-" * 50)

    # --- Step 1: Authenticate ---
    print("\n🔐 Step 1: Authentication")
    Path(TOKEN_DIR).mkdir(parents=True, exist_ok=True)

    api = Garmin(EMAIL, PASSWORD)

    # Check if we have cached tokens
    token_file = Path(TOKEN_DIR) / "garmin_tokens.json"
    if token_file.exists():
        print("  Found cached tokens, trying to reuse...")
        try:
            api.login(tokenstore=TOKEN_DIR)
            print("  ✅ Logged in with cached tokens")
        except Exception as e:
            print(f"  ⚠️  Cached tokens failed: {e}")
            print("  Waiting 15s before fresh login (rate limit cooldown)...")
            time.sleep(15)
            try:
                api.login()
                # Save tokens manually via the client
                api.client.dump(TOKEN_DIR)
                print("  ✅ Fresh login succeeded, tokens saved")
            except Exception as e2:
                if "429" in str(e2):
                    print(f"\n  ❌ Rate limited by Garmin (429)")
                    print("  Wait 10-15 minutes and try again.")
                    return
                raise
    else:
        print("  No cached tokens, doing first-time login...")
        try:
            api.login()
            api.client.dump(TOKEN_DIR)
            print("  ✅ Logged in and tokens cached for next time")
        except Exception as e:
            if "429" in str(e):
                print(f"\n  ❌ Rate limited by Garmin (429)")
                print("  This is normal after multiple attempts.")
                print("  Wait 10-15 minutes and try again.")
                print("  Tip: don't retry rapidly — each attempt extends the cooldown.")
                return
            raise

    # --- Step 2: Basic profile ---
    print("\n👤 Step 2: Profile info")
    try:
        full_name = api.get_full_name()
        print(f"  Name: {full_name}")
    except Exception as e:
        print(f"  ⚠️  Could not get name: {e}")

    try:
        units = api.get_unit_system()
        print(f"  Units: {units}")
    except Exception as e:
        print(f"  ⚠️  Could not get units: {e}")

    # --- Step 3: Today's daily stats ---
    print("\n📊 Step 3: Today's daily stats")
    today = date.today().isoformat()
    try:
        stats = api.get_stats(today)
        print(f"  Steps: {stats.get('totalSteps', 'N/A')}")
        print(f"  Calories: {stats.get('totalKilocalories', 'N/A')}")
        print(f"  Resting HR: {stats.get('restingHeartRate', 'N/A')}")
        print(f"  Stress: {stats.get('averageStressLevel', 'N/A')}")
        print(f"  Body Battery: {stats.get('bodyBatteryChargedValue', 'N/A')}")
        save_json(stats, "daily_stats_today.json")
    except Exception as e:
        print(f"  ⚠️  Could not get daily stats: {e}")

    # --- Step 4: Recent activities ---
    print("\n🏃 Step 4: Recent running activities (last 30 days)")
    start = (date.today() - timedelta(days=30)).isoformat()
    try:
        activities = api.get_activities_by_date(start, today, "running")
        print(f"  Found {len(activities)} running activities")
        for i, act in enumerate(activities[:5]):
            dist_km = round(act.get("distance", 0) / 1000, 2)
            dur_min = round(act.get("duration", 0) / 60, 1)
            avg_hr = act.get("averageHR", "N/A")
            print(f"  [{i+1}] {act.get('startTimeLocal', '?')} — "
                  f"{dist_km} km, {dur_min} min, avg HR {avg_hr}")
        save_json(activities, "recent_activities.json")
    except Exception as e:
        print(f"  ⚠️  Could not get activities: {e}")

    # Also get ALL activity types
    try:
        all_activities = api.get_activities_by_date(start, today)
        print(f"\n  Total activities (all types): {len(all_activities)}")
        types = set(
            a.get("activityType", {}).get("typeKey", "?")
            for a in all_activities
        )
        print(f"  Activity types found: {', '.join(sorted(types))}")
        save_json(all_activities, "all_activities.json")
    except Exception as e:
        print(f"  ⚠️  Could not get all activities: {e}")

    # --- Step 5: Sleep data ---
    print("\n😴 Step 5: Last night's sleep")
    try:
        sleep = api.get_sleep_data(today)
        daily_sleep = sleep.get("dailySleepDTO", {})
        total_h = round((daily_sleep.get("sleepTimeSeconds", 0) or 0) / 3600, 1)
        deep_h = round((daily_sleep.get("deepSleepSeconds", 0) or 0) / 3600, 1)
        rem_h = round((daily_sleep.get("remSleepSeconds", 0) or 0) / 3600, 1)
        score = daily_sleep.get("sleepScores", {}).get(
            "overall", {}
        ).get("value", "N/A")
        print(f"  Total sleep: {total_h}h")
        print(f"  Deep: {deep_h}h | REM: {rem_h}h")
        print(f"  Sleep score: {score}")
        save_json(sleep, "sleep_last_night.json")
    except Exception as e:
        print(f"  ⚠️  Could not get sleep data: {e}")

    # --- Step 6: HRV ---
    print("\n💓 Step 6: HRV data")
    try:
        hrv = api.get_hrv_data(today)
        summary = hrv.get("hrvSummary", {})
        print(f"  Last night avg HRV: {summary.get('lastNightAvg', 'N/A')} ms")
        print(f"  Weekly avg HRV: {summary.get('weeklyAvg', 'N/A')} ms")
        print(f"  HRV status: {summary.get('status', 'N/A')}")
        save_json(hrv, "hrv_today.json")
    except Exception as e:
        print(f"  ⚠️  Could not get HRV data: {e}")

    # --- Step 7: Training status ---
    print("\n🏋️ Step 7: Training status")
    try:
        training = api.get_training_status(today)
        print(f"  VO2max: {training.get('vo2MaxValue', 'N/A')}")
        print(f"  Training load (7d): {training.get('trainingLoad7Day', 'N/A')}")
        save_json(training, "training_status.json")
    except Exception as e:
        print(f"  ⚠️  Could not get training status: {e}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("✅ Connection test complete!")
    print(f"📁 Raw data saved in data/test/")
    print("   Explore the JSON files to see what fields are available.")
    print("=" * 50)


if __name__ == "__main__":
    main()
