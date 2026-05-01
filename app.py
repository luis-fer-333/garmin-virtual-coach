"""
Garmin Virtual Coach — Streamlit App
=====================================
Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to Streamlit Cloud

Users enter their Garmin credentials in the sidebar.
Data is fetched live and stays in the session (not persisted).
"""

import json
import sys
import time
import logging
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from src.coach.prompt_builder import SYSTEM_PROMPT
from src.coach.llm_client import LLMClient
from src.analytics import inject_ga4, track_event

logger = logging.getLogger(__name__)

# --- Page config ---
st.set_page_config(
    page_title="Garmin Virtual Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="darkgrid", palette="deep")

# --- Analytics ---
GA4_ID = st.secrets.get("GA4_MEASUREMENT_ID", "") if hasattr(st, "secrets") else ""
inject_ga4(GA4_ID)


# ============================================================
# GARMIN DATA FETCHING (live, per-user session)
# ============================================================

def fetch_garmin_data(email: str, password: str, lookback_days: int = 90) -> dict:
    """Connect to Garmin and pull all available data for the user."""
    from garminconnect import Garmin

    token_dir = str(Path.home() / ".garminconnect" / email.split("@")[0])
    Path(token_dir).mkdir(parents=True, exist_ok=True)

    api = Garmin(email, password)
    api.login(tokenstore=token_dir)

    today = date.today()
    today_str = today.isoformat()
    start_str = (today - timedelta(days=lookback_days)).isoformat()

    data = {"fetch_date": today_str, "lookback_days": lookback_days}

    def safe_fetch(name, func, *args, **kwargs):
        try:
            data[name] = func(*args, **kwargs)
        except Exception as e:
            logger.warning("%s fetch failed: %s", name, e)
            data[name] = None

    # --- Profile ---
    safe_fetch("full_name", api.get_full_name)
    safe_fetch("user_profile", api.get_user_profile)
    safe_fetch("device_info", api.get_device_last_used)

    # --- Activities (all types, full range) ---
    safe_fetch("all_activities", api.get_activities_by_date, start_str, today_str)

    # --- Daily stats ---
    safe_fetch("daily_stats", api.get_stats, today_str)
    safe_fetch("stats_and_body", api.get_stats_and_body, today_str)

    # --- Heart rate ---
    safe_fetch("heart_rates", api.get_heart_rates, today_str)
    safe_fetch("resting_hr", api.get_rhr_day, today_str)

    # --- Sleep ---
    safe_fetch("sleep", api.get_sleep_data, today_str)

    # --- HRV ---
    safe_fetch("hrv", api.get_hrv_data, today_str)

    # --- Stress ---
    safe_fetch("stress", api.get_stress_data, today_str)
    safe_fetch("all_day_stress", api.get_all_day_stress, today_str)

    # --- Body Battery ---
    safe_fetch("body_battery", api.get_body_battery, today_str)

    # --- Body composition (weight, BMI, body fat) ---
    safe_fetch("body_composition", api.get_body_composition, today_str)
    safe_fetch("weigh_ins", api.get_daily_weigh_ins, today_str)

    # --- Steps ---
    safe_fetch("steps_data", api.get_steps_data, today_str)
    safe_fetch("daily_steps", api.get_daily_steps, today_str)

    # --- Respiration ---
    safe_fetch("respiration", api.get_respiration_data, today_str)

    # --- SpO2 ---
    safe_fetch("spo2", api.get_spo2_data, today_str)

    # --- Floors ---
    safe_fetch("floors", api.get_floors, today_str)

    # --- Hydration ---
    safe_fetch("hydration", api.get_hydration_data, today_str)

    # --- Training metrics ---
    safe_fetch("training_readiness", api.get_training_readiness, today_str)
    safe_fetch("training_status", api.get_training_status, today_str)
    safe_fetch("endurance_score", api.get_endurance_score, today_str)
    safe_fetch("hill_score", api.get_hill_score, today_str)
    safe_fetch("max_metrics", api.get_max_metrics, today_str)
    safe_fetch("race_predictions", api.get_race_predictions)
    safe_fetch("fitness_age", api.get_fitnessage_data, today_str)

    # --- Intensity minutes ---
    safe_fetch("intensity_minutes", api.get_intensity_minutes_data, today_str)
    safe_fetch("weekly_intensity", api.get_weekly_intensity_minutes)

    # --- Personal records ---
    safe_fetch("personal_records", api.get_personal_record)

    return data

# ============================================================
# DATA PROCESSING
# ============================================================

def get_activities_df(data: dict) -> pd.DataFrame:
    activities = data.get("all_activities", [])
    if not activities:
        return pd.DataFrame()
    df = pd.DataFrame(activities)
    df["date"] = pd.to_datetime(df["startTimeLocal"])
    df["sport"] = df["activityType"].apply(lambda x: x["typeKey"])
    df["distance_km"] = df["distance"] / 1000
    df["duration_min"] = df["duration"] / 60
    df["calories_active"] = df["calories"] - df["bmrCalories"]
    sport_map = {
        "resort_skiing": "Skiing", "indoor_cardio": "Indoor Cardio",
        "mountain_biking": "MTB", "running": "Running", "hiking": "Hiking",
        "cycling": "Cycling", "swimming": "Swimming",
        "strength_training": "Strength", "yoga": "Yoga",
        "trail_running": "Trail Running", "walking": "Walking",
    }
    df["sport_label"] = df["sport"].map(sport_map).fillna(df["sport"])
    return df


def build_athlete_context(data: dict, df_act: pd.DataFrame) -> str:
    sections = []

    # Activities summary
    if not df_act.empty:
        sections.append(f"## Recent Activities (last {data.get('lookback_days', 30)} days)")
        for _, row in df_act.iterrows():
            sections.append(
                f"- {row['date'].strftime('%b %d')} | {row['sport_label']} | "
                f"{row['distance_km']:.1f}km | {row['duration_min']:.0f}min | "
                f"avg HR {row.get('averageHR', 'N/A')} | "
                f"training load {row.get('activityTrainingLoad', 'N/A')}"
            )

    # Daily stats
    stats = data.get("daily_stats") or {}
    if stats:
        sections.append(f"""## Today's Wellness
- Resting HR: {stats.get('restingHeartRate', 'N/A')} bpm (7d avg: {stats.get('lastSevenDaysAvgRestingHeartRate', 'N/A')})
- Body Battery: {stats.get('bodyBatteryMostRecentValue', 'N/A')}/100 (highest: {stats.get('bodyBatteryHighestValue', 'N/A')}, lowest: {stats.get('bodyBatteryLowestValue', 'N/A')})
- Stress: avg {stats.get('averageStressLevel', 'N/A')}, max {stats.get('maxStressLevel', 'N/A')}
- Steps: {stats.get('totalSteps', 'N/A')} / {stats.get('dailyStepGoal', 'N/A')} goal
- Active calories: {stats.get('activeKilocalories', 'N/A')} kcal
- Intensity minutes: moderate {stats.get('moderateIntensityMinutes', 'N/A')}, vigorous {stats.get('vigorousIntensityMinutes', 'N/A')}
- Floors climbed: {stats.get('floorsAscended', 'N/A')}
- Respiration: avg {stats.get('avgWakingRespirationValue', 'N/A')} breaths/min""")

    # Sleep
    sleep = data.get("sleep") or {}
    if sleep:
        ds = sleep.get("dailySleepDTO", {})
        total_h = round((ds.get("sleepTimeSeconds", 0) or 0) / 3600, 1)
        deep_h = round((ds.get("deepSleepSeconds", 0) or 0) / 3600, 1)
        light_h = round((ds.get("lightSleepSeconds", 0) or 0) / 3600, 1)
        rem_h = round((ds.get("remSleepSeconds", 0) or 0) / 3600, 1)
        awake_min = round((ds.get("awakeSleepSeconds", 0) or 0) / 60, 0)
        score = ds.get("sleepScores", {}).get("overall", {}).get("value", "N/A")
        sections.append(f"""## Last Night's Sleep
- Total: {total_h}h | Deep: {deep_h}h | Light: {light_h}h | REM: {rem_h}h | Awake: {awake_min}min
- Sleep score: {score}/100
- Avg stress during sleep: {ds.get('avgSleepStress', 'N/A')}
- Respiration: avg {ds.get('averageRespirationValue', 'N/A')} breaths/min""")

    # HRV
    hrv = data.get("hrv") or {}
    if hrv:
        summary = hrv.get("hrvSummary", {})
        baseline = summary.get("baseline", {})
        sections.append(f"""## HRV
- Last night avg: {summary.get('lastNightAvg', 'N/A')}ms
- Last night 5-min high: {summary.get('lastNight5MinHigh', 'N/A')}ms
- Weekly avg: {summary.get('weeklyAvg', 'N/A')}ms
- Status: {summary.get('status', 'N/A')}
- Balanced range: {baseline.get('balancedLow', 'N/A')}–{baseline.get('balancedUpper', 'N/A')}ms""")

    # Training readiness
    readiness = data.get("training_readiness")
    if readiness and isinstance(readiness, dict):
        sections.append(f"""## Training Readiness
- Score: {readiness.get('score', 'N/A')}
- Level: {readiness.get('level', 'N/A')}""")

    # Training status
    tstatus = data.get("training_status")
    if tstatus and isinstance(tstatus, dict):
        sections.append(f"""## Training Status
- VO2max: {tstatus.get('vo2MaxValue', 'N/A')}
- Training load (7d): {tstatus.get('trainingLoad7Day', 'N/A')}
- Training load balance: {tstatus.get('trainingLoadBalance', 'N/A')}""")

    # Race predictions
    preds = data.get("race_predictions")
    if preds and isinstance(preds, list) and len(preds) > 0:
        sections.append("## Race Predictions")
        for p in preds:
            dist = p.get("raceName", p.get("raceDistanceKey", "?"))
            time_sec = p.get("predictedTime", 0)
            if time_sec:
                mins = int(time_sec // 60)
                secs = int(time_sec % 60)
                sections.append(f"- {dist}: {mins}:{secs:02d}")

    # Fitness age
    fitage = data.get("fitness_age")
    if fitage and isinstance(fitage, dict):
        sections.append(f"## Fitness Age: {fitage.get('fitnessAge', 'N/A')} (chronological: {fitage.get('chronologicalAge', 'N/A')})")

    # Body composition
    body = data.get("body_composition")
    if body and isinstance(body, dict):
        sections.append(f"""## Body Composition
- Weight: {body.get('weight', 'N/A')} kg
- BMI: {body.get('bmi', 'N/A')}
- Body fat: {body.get('bodyFat', 'N/A')}%""")

    # Personal records
    records = data.get("personal_records")
    if records and isinstance(records, list) and len(records) > 0:
        sections.append("## Personal Records")
        for r in records[:5]:
            sections.append(f"- {r.get('typeKey', '?')}: {r.get('value', 'N/A')}")

    return "\n\n".join(sections)


# ============================================================
# CHART FUNCTIONS
# ============================================================

def chart_hr_zones(df_act):
    zone_cols = ["hrTimeInZone_1", "hrTimeInZone_2", "hrTimeInZone_3",
                 "hrTimeInZone_4", "hrTimeInZone_5"]
    zone_labels = ["Z1 Recovery", "Z2 Easy", "Z3 Aerobic", "Z4 Threshold", "Z5 VO2max"]
    zone_colors = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 4))
    zone_data = df_act[zone_cols].fillna(0) / 60
    labels = df_act["sport_label"] + " (" + df_act["date"].dt.strftime("%b %d") + ")"
    zone_data.index = labels

    bottom = np.zeros(len(zone_data))
    for col, label, color in zip(zone_cols, zone_labels, zone_colors):
        values = zone_data[col].values
        ax.barh(zone_data.index, values, left=bottom, label=label, color=color)
        bottom += values

    ax.set_xlabel("Time in Zone (minutes)")
    ax.set_title("Heart Rate Zone Distribution")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def chart_training_load(df_act):
    sport_colors = {
        "Skiing": "#2196F3", "Indoor Cardio": "#FF5722",
        "MTB": "#4CAF50", "Running": "#FF9800", "Hiking": "#795548",
        "Cycling": "#00BCD4", "Trail Running": "#FF9800",
        "Strength": "#9C27B0", "Swimming": "#03A9F4",
    }
    df_sorted = df_act.sort_values("date")
    colors = [sport_colors.get(s, "#9E9E9E") for s in df_sorted["sport_label"]]
    labels = df_sorted["sport_label"] + "\n" + df_sorted["date"].dt.strftime("%b %d")

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, df_sorted["activityTrainingLoad"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df_sorted["activityTrainingLoad"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Training Load")
    ax.set_title("Training Load per Activity")
    plt.tight_layout()
    return fig


def chart_sleep(data):
    sleep = data.get("sleep")
    if not sleep or not isinstance(sleep, dict):
        return None
    ds = sleep.get("dailySleepDTO", {})
    if not ds:
        return None
    stages = {
        "Deep": (ds.get("deepSleepSeconds", 0) or 0) / 3600,
        "Light": (ds.get("lightSleepSeconds", 0) or 0) / 3600,
        "REM": (ds.get("remSleepSeconds", 0) or 0) / 3600,
        "Awake": (ds.get("awakeSleepSeconds", 0) or 0) / 3600,
    }
    colors = ["#1a237e", "#5c6bc0", "#7e57c2", "#ef5350"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(stages.values(), labels=stages.keys(), colors=colors,
           autopct=lambda p: f"{p:.0f}%", startangle=90, pctdistance=0.75,
           wedgeprops=dict(width=0.4))
    score = ds.get("sleepScores", {}).get("overall", {}).get("value", "?")
    ax.text(0, 0, f"{score}", ha="center", va="center",
            fontsize=28, fontweight="bold", color="#1a237e")
    ax.set_title("Sleep Architecture")
    return fig


def chart_hrv(data):
    hrv = data.get("hrv")
    if not hrv or not isinstance(hrv, dict):
        return None
    readings = pd.DataFrame(hrv.get("hrvReadings", []))
    if readings.empty:
        return None
    readings["time"] = pd.to_datetime(readings["readingTimeLocal"])
    baseline = hrv.get("hrvSummary", {}).get("baseline", {})

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(readings["time"], readings["hrvValue"],
            color="#7e57c2", linewidth=1.5, marker="o", markersize=2)
    if baseline:
        ax.axhspan(baseline.get("balancedLow", 0), baseline.get("balancedUpper", 100),
                    alpha=0.15, color="#2ecc71", label="Balanced range")
    avg = hrv.get("hrvSummary", {}).get("lastNightAvg")
    if avg:
        ax.axhline(y=avg, color="#3498db", linestyle="-.", alpha=0.7,
                    label=f"Avg: {avg}ms")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("HRV (ms)")
    ax.set_title("Overnight HRV")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    # --- Sidebar: Login ---
    with st.sidebar:
        st.title("🏃 Garmin Virtual Coach")
        st.caption("AI-powered training insights")
        st.divider()

        # Check if user is already connected
        if "garmin_data" not in st.session_state:
            st.subheader("Connect your Garmin")
            st.caption("Your credentials are used only to fetch data and are not stored.")

            garmin_email = st.text_input("Garmin Email", placeholder="you@email.com")
            garmin_password = st.text_input("Garmin Password", type="password")
            lookback = st.slider("Days of history", 7, 365, 90, step=7)

            if st.button("Connect", type="primary", use_container_width=True):
                if not garmin_email or not garmin_password:
                    st.error("Please enter both email and password")
                else:
                    with st.spinner("Connecting to Garmin (fetching all data)..."):
                        try:
                            data = fetch_garmin_data(garmin_email, garmin_password, lookback)
                            st.session_state.garmin_data = data
                            st.session_state.garmin_email = garmin_email
                            track_event(GA4_ID, "garmin_login", {
                                "lookback_days": str(lookback),
                                "num_activities": str(len(data.get("all_activities") or [])),
                            })
                            st.rerun()
                        except Exception as e:
                            if "429" in str(e):
                                st.error(
                                    "Rate limited by Garmin. Wait 5-10 minutes "
                                    "and try again. Don't retry rapidly."
                                )
                            else:
                                st.error(f"Login failed: {e}")

            st.divider()
            st.info(
                "**How it works:**\n"
                "1. Enter your Garmin Connect credentials\n"
                "2. We fetch your last 30 days of data\n"
                "3. Chat with your AI coach or explore the dashboard\n\n"
                "Your data stays in your browser session only."
            )
            return  # Stop here until logged in

        # --- User is connected ---
        data = st.session_state.garmin_data
        name = data.get("full_name", "Athlete")
        st.success(f"Connected as **{name}**")

        if st.button("Disconnect", use_container_width=True):
            del st.session_state.garmin_data
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()

        if st.button("Refresh Data", use_container_width=True):
            with st.spinner("Refreshing..."):
                try:
                    data = fetch_garmin_data(
                        st.session_state.garmin_email,
                        st.session_state.get("garmin_password", ""),
                    )
                    st.session_state.garmin_data = data
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

        st.divider()

        # Quick stats
        stats = data.get("daily_stats") or {}
        if stats:
            col1, col2 = st.columns(2)
            col1.metric("Resting HR", f"{stats.get('restingHeartRate', '—')} bpm")
            col2.metric("Body Battery", f"{stats.get('bodyBatteryMostRecentValue', '—')}")
            col1, col2 = st.columns(2)
            col1.metric("Stress", f"{stats.get('averageStressLevel', '—')}")
            col2.metric("Steps", f"{stats.get('totalSteps', 0):,}")

        hrv_summary = data.get("hrv", {}).get("hrvSummary", {}) if data.get("hrv") else {}
        if hrv_summary:
            st.metric("HRV", f"{hrv_summary.get('lastNightAvg', '—')}ms "
                       f"({hrv_summary.get('status', '')})")

        sleep_ds = data.get("sleep", {}).get("dailySleepDTO", {}) if data.get("sleep") else {}
        if sleep_ds:
            score = sleep_ds.get("sleepScores", {}).get("overall", {}).get("value", "—")
            total_h = round((sleep_ds.get("sleepTimeSeconds", 0) or 0) / 3600, 1)
            st.metric("Sleep", f"{total_h}h (score: {score})")

        # Training readiness
        readiness = data.get("training_readiness")
        if readiness and isinstance(readiness, dict) and readiness.get("score"):
            st.metric("Training Readiness", f"{readiness['score']}")

        # Fitness age
        fitage = data.get("fitness_age")
        if fitage and isinstance(fitage, dict) and fitage.get("fitnessAge"):
            st.metric("Fitness Age", f"{fitage['fitnessAge']}")

        st.divider()
        n_activities = len(data.get("all_activities") or [])
        st.caption(f"Activities: {n_activities} | Last {data.get('lookback_days', 30)} days")

    # --- Process data ---
    df_act = get_activities_df(data)
    athlete_context = build_athlete_context(data, df_act)

    # --- Tabs ---
    tab_coach, tab_dashboard = st.tabs(["💬 Coach Chat", "📊 Dashboard"])

    # ========== COACH CHAT ==========
    with tab_coach:
        st.header("Talk to your Coach")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # --- Suggested prompts (shown when chat is empty) ---
        if not st.session_state.messages:
            st.markdown(
                f"Hey **{name}**! I'm your AI coach. I can see all your Garmin data "
                "— activities, sleep, HRV, stress, body battery, and more.\n\n"
                "Pick a topic below or type your own question."
            )

            st.subheader("🔍 Evaluate")
            eval_cols = st.columns(3)
            eval_prompts = [
                ("📊 Weekly Review", "Give me a detailed weekly training review. Analyze my volume, intensity, recovery, and tell me what went well and what needs attention."),
                ("🏃 Performance Check", "Evaluate my overall fitness and performance based on all available data. How am I doing? Am I improving or declining?"),
                ("😴 Sleep Analysis", "Analyze my sleep quality in detail. Look at deep sleep, REM, sleep score, and how it's affecting my recovery and readiness."),
            ]
            for col, (label, prompt) in zip(eval_cols, eval_prompts):
                if col.button(label, use_container_width=True):
                    track_event(GA4_ID, "suggested_prompt", {"category": "evaluate", "prompt": label})
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.rerun()

            st.subheader("💡 Improve")
            improve_cols = st.columns(3)
            improve_prompts = [
                ("🎯 Where to Focus", "Based on my data, what are the top 3 areas I should focus on to improve? Be specific with numbers and actionable steps."),
                ("⚡ Quick Wins", "What are some quick wins I can implement this week to improve my training, recovery, or overall health? Look at my weakest metrics."),
                ("🔄 Recovery Tips", "Analyze my recovery metrics (HRV, sleep, stress, body battery) and give me specific tips to recover better."),
            ]
            for col, (label, prompt) in zip(improve_cols, improve_prompts):
                if col.button(label, use_container_width=True):
                    track_event(GA4_ID, "suggested_prompt", {"category": "improve", "prompt": label})
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.rerun()

            st.subheader("📋 Custom Plan")
            plan_cols = st.columns(3)
            plan_prompts = [
                ("🗓️ Build My Plan", "I want a custom training plan. Ask me about my goals, available days, current fitness level, and any constraints before creating the plan."),
                ("🏁 Race Prep", "I'm preparing for a race. Ask me about the race distance, date, and my current fitness, then create a preparation plan."),
                ("🔥 Get Stronger", "I want to improve my overall fitness. Ask me what sports I do, what my goals are, and how many hours per week I can train, then build a plan."),
            ]
            for col, (label, prompt) in zip(plan_cols, plan_prompts):
                if col.button(label, use_container_width=True):
                    track_event(GA4_ID, "suggested_prompt", {"category": "plan", "prompt": label})
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.rerun()

            st.subheader("❓ Ask Anything")
            st.caption("Or type your own question below — I have access to all your Garmin data.")

        # --- Display chat history ---
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # --- Generate response for pending user message ---
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            coach_system = SYSTEM_PROMPT + f"""

You have access to this athlete's current data:

{athlete_context}

Use this data to give specific, personalized advice. Reference actual numbers
from their data (HR, distances, sleep scores, HRV) in your responses.

When the user asks you to build a custom plan, DO NOT create it immediately.
First ask them 3-5 questions to understand their goals, constraints, and
preferences. Only after they answer, create a detailed plan.

Keep responses conversational and under 400 words unless they ask for a
detailed review or plan."""

            conversation = ""
            for msg in st.session_state.messages[-8:]:
                role = "User" if msg["role"] == "user" else "Coach"
                conversation += f"\n{role}: {msg['content']}"

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        llm = LLMClient()
                        response = llm.generate(coach_system, conversation)
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        track_event(GA4_ID, "coach_response", {
                            "message_count": str(len(st.session_state.messages)),
                            "response_length": str(len(response)),
                        })
                    except Exception as e:
                        st.error(f"Sorry, I couldn't generate a response: {e}")

        # --- Chat input (always visible) ---
        if prompt := st.chat_input("Ask your coach..."):
            track_event(GA4_ID, "chat_message", {"type": "free_text"})
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        # --- Clear chat button ---
        if st.session_state.messages:
            if st.button("🗑️ Clear chat", use_container_width=False):
                st.session_state.messages = []
                st.rerun()

    # ========== DASHBOARD ==========
    with tab_dashboard:
        st.header("Training Dashboard")
        track_event(GA4_ID, "view_dashboard")

        if df_act.empty:
            st.info("No activities found in the last 30 days. Get moving!")
            return

        # Activity cards
        st.subheader("Recent Activities")
        n_cols = min(len(df_act), 5)
        cols = st.columns(n_cols)
        for col, (_, row) in zip(cols, df_act.head(n_cols).iterrows()):
            with col:
                st.metric(
                    row["sport_label"],
                    f"{row['distance_km']:.1f} km",
                    f"{row['duration_min']:.0f} min",
                )
                st.caption(row["date"].strftime("%b %d"))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("HR Zone Distribution")
            fig = chart_hr_zones(df_act)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            st.subheader("Training Load")
            fig = chart_training_load(df_act)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sleep Architecture")
            fig = chart_sleep(data)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No sleep data available")
        with col2:
            st.subheader("Overnight HRV")
            fig = chart_hrv(data)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No HRV data available")

        # Details table
        st.divider()
        st.subheader("Activity Details")
        display_cols = [
            "date", "sport_label", "distance_km", "duration_min",
            "averageHR", "maxHR", "calories_active", "activityTrainingLoad",
            "elevationGain", "locationName",
        ]
        available = [c for c in display_cols if c in df_act.columns]
        st.dataframe(
            df_act[available].rename(columns={
                "sport_label": "Sport", "distance_km": "Distance (km)",
                "duration_min": "Duration (min)", "averageHR": "Avg HR",
                "maxHR": "Max HR", "calories_active": "Active Cal",
                "activityTrainingLoad": "Training Load",
                "elevationGain": "Elevation (m)", "locationName": "Location",
                "date": "Date",
            }),
            use_container_width=True, hide_index=True,
        )


if __name__ == "__main__":
    main()
