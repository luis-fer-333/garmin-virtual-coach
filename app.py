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

logger = logging.getLogger(__name__)

# --- Page config ---
st.set_page_config(
    page_title="Garmin Virtual Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="darkgrid", palette="deep")


# ============================================================
# GARMIN DATA FETCHING (live, per-user session)
# ============================================================

def fetch_garmin_data(email: str, password: str) -> dict:
    """Connect to Garmin and pull all relevant data for the user."""
    from garminconnect import Garmin

    token_dir = str(Path.home() / ".garminconnect" / email.split("@")[0])
    Path(token_dir).mkdir(parents=True, exist_ok=True)

    api = Garmin(email, password)
    api.login(tokenstore=token_dir)

    today = date.today().isoformat()
    start = (date.today() - timedelta(days=30)).isoformat()

    data = {}

    # Activities
    try:
        data["all_activities"] = api.get_activities_by_date(start, today)
    except Exception as e:
        logger.warning("Activities fetch failed: %s", e)
        data["all_activities"] = []

    # Daily stats
    try:
        data["daily_stats"] = api.get_stats(today)
    except Exception as e:
        logger.warning("Stats fetch failed: %s", e)
        data["daily_stats"] = {}

    # Sleep
    try:
        data["sleep"] = api.get_sleep_data(today)
    except Exception as e:
        logger.warning("Sleep fetch failed: %s", e)
        data["sleep"] = {}

    # HRV
    try:
        data["hrv"] = api.get_hrv_data(today)
    except Exception as e:
        logger.warning("HRV fetch failed: %s", e)
        data["hrv"] = {}

    # Profile name
    try:
        data["full_name"] = api.get_full_name()
    except Exception:
        data["full_name"] = email.split("@")[0]

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

    if not df_act.empty:
        sections.append("## Recent Activities (last 30 days)")
        for _, row in df_act.iterrows():
            sections.append(
                f"- {row['date'].strftime('%b %d')} | {row['sport_label']} | "
                f"{row['distance_km']:.1f}km | {row['duration_min']:.0f}min | "
                f"avg HR {row.get('averageHR', 'N/A')} | "
                f"training load {row.get('activityTrainingLoad', 'N/A')}"
            )

    stats = data.get("daily_stats", {})
    if stats:
        sections.append(f"""## Today's Wellness
- Resting HR: {stats.get('restingHeartRate', 'N/A')} bpm
- Body Battery: {stats.get('bodyBatteryMostRecentValue', 'N/A')}/100
- Stress: avg {stats.get('averageStressLevel', 'N/A')}
- Steps: {stats.get('totalSteps', 'N/A')}""")

    sleep = data.get("sleep", {})
    if sleep:
        ds = sleep.get("dailySleepDTO", {})
        total_h = round((ds.get("sleepTimeSeconds", 0) or 0) / 3600, 1)
        deep_h = round((ds.get("deepSleepSeconds", 0) or 0) / 3600, 1)
        rem_h = round((ds.get("remSleepSeconds", 0) or 0) / 3600, 1)
        score = ds.get("sleepScores", {}).get("overall", {}).get("value", "N/A")
        sections.append(f"""## Last Night's Sleep
- Total: {total_h}h | Deep: {deep_h}h | REM: {rem_h}h
- Sleep score: {score}/100""")

    hrv = data.get("hrv", {})
    if hrv:
        summary = hrv.get("hrvSummary", {})
        sections.append(f"""## HRV
- Last night avg: {summary.get('lastNightAvg', 'N/A')}ms
- Weekly avg: {summary.get('weeklyAvg', 'N/A')}ms
- Status: {summary.get('status', 'N/A')}""")

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
    ds = data.get("sleep", {}).get("dailySleepDTO", {})
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
    hrv = data.get("hrv", {})
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

            if st.button("Connect", type="primary", use_container_width=True):
                if not garmin_email or not garmin_password:
                    st.error("Please enter both email and password")
                else:
                    with st.spinner("Connecting to Garmin..."):
                        try:
                            data = fetch_garmin_data(garmin_email, garmin_password)
                            st.session_state.garmin_data = data
                            st.session_state.garmin_email = garmin_email
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
        stats = data.get("daily_stats", {})
        if stats:
            col1, col2 = st.columns(2)
            col1.metric("Resting HR", f"{stats.get('restingHeartRate', '—')} bpm")
            col2.metric("Body Battery", f"{stats.get('bodyBatteryMostRecentValue', '—')}")
            col1, col2 = st.columns(2)
            col1.metric("Stress", f"{stats.get('averageStressLevel', '—')}")
            col2.metric("Steps", f"{stats.get('totalSteps', 0):,}")

        hrv_summary = data.get("hrv", {}).get("hrvSummary", {})
        if hrv_summary:
            st.metric("HRV", f"{hrv_summary.get('lastNightAvg', '—')}ms "
                       f"({hrv_summary.get('status', '')})")

        sleep_ds = data.get("sleep", {}).get("dailySleepDTO", {})
        if sleep_ds:
            score = sleep_ds.get("sleepScores", {}).get("overall", {}).get("value", "—")
            total_h = round((sleep_ds.get("sleepTimeSeconds", 0) or 0) / 3600, 1)
            st.metric("Sleep", f"{total_h}h (score: {score})")

    # --- Process data ---
    df_act = get_activities_df(data)
    athlete_context = build_athlete_context(data, df_act)

    # --- Tabs ---
    tab_coach, tab_dashboard = st.tabs(["💬 Coach Chat", "📊 Dashboard"])

    # ========== COACH CHAT ==========
    with tab_coach:
        st.header("Talk to your Coach")
        st.caption("Ask about your training, recovery, sleep, or get a weekly review.")

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hey {name}! I'm your virtual coach. I can see your "
                        "Garmin data — activities, sleep, HRV, and daily stats.\n\n"
                        "Try asking me:\n"
                        "- *How was my week?*\n"
                        "- *Analyze my sleep*\n"
                        "- *Should I train hard today?*\n"
                        "- *Give me a weekly review*"
                    ),
                }
            ]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your coach..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            coach_system = SYSTEM_PROMPT + f"""

You have access to this athlete's current data:

{athlete_context}

Use this data to give specific, personalized advice. Reference actual numbers
from their data (HR, distances, sleep scores, HRV) in your responses.
Keep responses conversational and under 300 words unless they ask for a
detailed review."""

            conversation = ""
            for msg in st.session_state.messages[-6:]:
                role = "User" if msg["role"] == "user" else "Coach"
                conversation += f"\n{role}: {msg['content']}"

            full_prompt = conversation + f"\nUser: {prompt}\nCoach:"

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        llm = LLMClient()
                        response = llm.generate(coach_system, full_prompt)
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        st.error(f"Sorry, I couldn't generate a response: {e}")

    # ========== DASHBOARD ==========
    with tab_dashboard:
        st.header("Training Dashboard")

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
