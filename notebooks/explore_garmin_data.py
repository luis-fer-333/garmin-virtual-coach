"""
Garmin Data Exploration — Visual Dashboard
==========================================
Generates meaningful charts from your Garmin Connect data.
Run from the garmin-virtual-coach directory:
    python notebooks/explore_garmin_data.py

Charts are saved to notebooks/charts/
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- Setup ---
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

CHARTS_DIR = Path("notebooks/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/test")


def load_json(filename):
    with open(DATA_DIR / filename) as f:
        return json.load(f)


# ============================================================
# 1. ACTIVITY OVERVIEW — Multi-sport comparison
# ============================================================
print("📊 Chart 1: Activity Overview")
activities = load_json("all_activities.json")
df_act = pd.DataFrame(activities)
df_act["date"] = pd.to_datetime(df_act["startTimeLocal"])
df_act["sport"] = df_act["activityType"].apply(lambda x: x["typeKey"])
df_act["distance_km"] = df_act["distance"] / 1000
df_act["duration_min"] = df_act["duration"] / 60
df_act["calories_active"] = df_act["calories"] - df_act["bmrCalories"]

# Sport labels for display
sport_labels = {
    "resort_skiing": "⛷️ Skiing",
    "indoor_cardio": "🏋️ Indoor Cardio",
    "mountain_biking": "🚵 MTB",
    "running": "🏃 Running",
    "hiking": "🥾 Hiking",
}
df_act["sport_label"] = df_act["sport"].map(sport_labels).fillna(df_act["sport"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Activity Overview — Last 30 Days", fontsize=16, fontweight="bold", y=1.02)

# 1a. Distance by sport
colors = sns.color_palette("husl", len(df_act))
ax = axes[0]
bars = ax.barh(df_act["sport_label"], df_act["distance_km"], color=colors)
ax.set_xlabel("Distance (km)")
ax.set_title("Distance per Activity")
for bar, val in zip(bars, df_act["distance_km"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=10)

# 1b. Duration by sport
ax = axes[1]
bars = ax.barh(df_act["sport_label"], df_act["duration_min"], color=colors)
ax.set_xlabel("Duration (min)")
ax.set_title("Duration per Activity")
for bar, val in zip(bars, df_act["duration_min"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=10)

# 1c. Active calories by sport
ax = axes[2]
bars = ax.barh(df_act["sport_label"], df_act["calories_active"], color=colors)
ax.set_xlabel("Active Calories (kcal)")
ax.set_title("Active Calories per Activity")
for bar, val in zip(bars, df_act["calories_active"]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "01_activity_overview.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 01_activity_overview.png")

# ============================================================
# 2. HEART RATE ZONES — Stacked bar per activity
# ============================================================
print("📊 Chart 2: HR Zone Distribution")

zone_cols = ["hrTimeInZone_1", "hrTimeInZone_2", "hrTimeInZone_3",
             "hrTimeInZone_4", "hrTimeInZone_5"]
zone_labels = ["Z1 Recovery", "Z2 Easy", "Z3 Aerobic", "Z4 Threshold", "Z5 VO2max"]
zone_colors = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

fig, ax = plt.subplots(figsize=(14, 6))

# Convert seconds to minutes
zone_data = df_act[zone_cols].fillna(0) / 60
zone_data.index = df_act["sport_label"] + "\n" + df_act["date"].dt.strftime("%b %d")

bottom = np.zeros(len(zone_data))
for i, (col, label, color) in enumerate(zip(zone_cols, zone_labels, zone_colors)):
    values = zone_data[col].values
    ax.barh(zone_data.index, values, left=bottom, label=label, color=color)
    bottom += values

ax.set_xlabel("Time in Zone (minutes)")
ax.set_title("Heart Rate Zone Distribution per Activity", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "02_hr_zones.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 02_hr_zones.png")


# ============================================================
# 3. TRAINING EFFECT — Aerobic vs Anaerobic scatter
# ============================================================
print("📊 Chart 3: Training Effect")

fig, ax = plt.subplots(figsize=(8, 8))

for sport, group in df_act.groupby("sport_label"):
    ax.scatter(
        group["aerobicTrainingEffect"],
        group["anaerobicTrainingEffect"],
        s=group["duration_min"] * 3,  # Size = duration
        label=sport,
        alpha=0.8,
        edgecolors="white",
        linewidth=1.5,
    )

# Add quadrant lines
ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Aerobic Training Effect", fontsize=12)
ax.set_ylabel("Anaerobic Training Effect", fontsize=12)
ax.set_title("Training Effect — Aerobic vs Anaerobic\n(bubble size = duration)",
             fontsize=14, fontweight="bold")
ax.set_xlim(-0.2, 5.5)
ax.set_ylim(-0.2, 5.5)
ax.legend(fontsize=10)

# Quadrant labels
ax.text(1.0, 4.5, "Speed\nFocused", ha="center", fontsize=9, color="gray")
ax.text(4.0, 4.5, "High\nIntensity", ha="center", fontsize=9, color="gray")
ax.text(1.0, 0.5, "Recovery /\nLight", ha="center", fontsize=9, color="gray")
ax.text(4.0, 0.5, "Endurance\nFocused", ha="center", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "03_training_effect.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 03_training_effect.png")

# ============================================================
# 4. SLEEP ARCHITECTURE — Last night breakdown
# ============================================================
print("📊 Chart 4: Sleep Architecture")

sleep = load_json("sleep_last_night.json")
daily_sleep = sleep["dailySleepDTO"]

sleep_stages = {
    "Deep Sleep": daily_sleep.get("deepSleepSeconds", 0) / 3600,
    "Light Sleep": daily_sleep.get("lightSleepSeconds", 0) / 3600,
    "REM Sleep": daily_sleep.get("remSleepSeconds", 0) / 3600,
    "Awake": daily_sleep.get("awakeSleepSeconds", 0) / 3600,
}
stage_colors = ["#1a237e", "#5c6bc0", "#7e57c2", "#ef5350"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4a. Donut chart
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    sleep_stages.values(),
    labels=sleep_stages.keys(),
    colors=stage_colors,
    autopct=lambda p: f"{p:.0f}%\n({p * sum(sleep_stages.values()) / 100:.1f}h)",
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.4),
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Sleep Architecture — Last Night", fontsize=13, fontweight="bold")

# Add sleep score in center
score = daily_sleep.get("sleepScores", {}).get("overall", {}).get("value", "?")
ax.text(0, 0, f"Score\n{score}", ha="center", va="center",
        fontsize=20, fontweight="bold", color="#1a237e")

# 4b. Sleep quality scores breakdown
ax = axes[1]
scores = daily_sleep.get("sleepScores", {})
score_items = {}
for key in ["remPercentage", "deepPercentage", "lightPercentage"]:
    if key in scores and "qualifierKey" in scores[key]:
        label = key.replace("Percentage", "").replace("light", "Light")\
                    .replace("deep", "Deep").replace("rem", "REM")
        score_items[label] = scores[key].get("value", 0)

# Add overall
score_items["Overall"] = scores.get("overall", {}).get("value", 0)

qualifier_map = {
    "EXCELLENT": "#2ecc71", "GOOD": "#3498db",
    "FAIR": "#f1c40f", "POOR": "#e74c3c"
}

bar_colors = []
for key in ["remPercentage", "deepPercentage", "lightPercentage"]:
    q = scores.get(key, {}).get("qualifierKey", "FAIR")
    bar_colors.append(qualifier_map.get(q, "#95a5a6"))
bar_colors.append(qualifier_map.get(
    scores.get("overall", {}).get("qualifierKey", "FAIR"), "#95a5a6"))

bars = ax.barh(list(score_items.keys()), list(score_items.values()), color=bar_colors)
ax.set_xlabel("Score / Percentage")
ax.set_title("Sleep Quality Breakdown", fontsize=13, fontweight="bold")
ax.set_xlim(0, 100)
for bar, val in zip(bars, score_items.values()):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val}", va="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "04_sleep_architecture.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 04_sleep_architecture.png")

# ============================================================
# 5. HRV OVERNIGHT — Time series with baseline bands
# ============================================================
print("📊 Chart 5: HRV Overnight")

hrv = load_json("hrv_today.json")
hrv_readings = pd.DataFrame(hrv["hrvReadings"])
hrv_readings["time"] = pd.to_datetime(hrv_readings["readingTimeLocal"])
hrv_readings["hour"] = hrv_readings["time"].dt.strftime("%H:%M")

baseline = hrv["hrvSummary"]["baseline"]

fig, ax = plt.subplots(figsize=(14, 5))

# Plot HRV readings
ax.plot(hrv_readings["time"], hrv_readings["hrvValue"],
        color="#7e57c2", linewidth=1.5, marker="o", markersize=3, label="HRV (ms)")

# Baseline bands
ax.axhspan(baseline["balancedLow"], baseline["balancedUpper"],
           alpha=0.15, color="#2ecc71", label="Balanced range")
ax.axhline(y=baseline["lowUpper"], color="#e74c3c",
           linestyle="--", alpha=0.6, label=f"Low threshold ({baseline['lowUpper']}ms)")

# Average line
avg = hrv["hrvSummary"]["lastNightAvg"]
ax.axhline(y=avg, color="#3498db", linestyle="-.", alpha=0.7,
           label=f"Last night avg ({avg}ms)")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time (local)")
ax.set_ylabel("HRV (ms)")
ax.set_title(f"Overnight HRV — {hrv['hrvSummary']['calendarDate']}  |  "
             f"Status: {hrv['hrvSummary']['status']}",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "05_hrv_overnight.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 05_hrv_overnight.png")


# ============================================================
# 6. DAILY WELLNESS DASHBOARD — Body Battery + Stress + HR
# ============================================================
print("📊 Chart 6: Daily Wellness Dashboard")

stats = load_json("daily_stats_today.json")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f"Daily Wellness — {stats['calendarDate']}",
             fontsize=15, fontweight="bold", y=1.05)

# 6a. Body Battery gauge
ax = axes[0]
bb_val = stats.get("bodyBatteryMostRecentValue", 0)
bb_color = "#2ecc71" if bb_val >= 60 else "#f1c40f" if bb_val >= 30 else "#e74c3c"
ax.barh(["Body Battery"], [bb_val], color=bb_color, height=0.5)
ax.barh(["Body Battery"], [100 - bb_val], left=[bb_val],
        color="#ecf0f1", height=0.5)
ax.set_xlim(0, 100)
ax.text(bb_val / 2, 0, f"{bb_val}", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")
ax.set_title("Body Battery", fontsize=12)

# 6b. Resting HR
ax = axes[1]
rhr = stats.get("restingHeartRate", 0)
rhr_7d = stats.get("lastSevenDaysAvgRestingHeartRate", 0)
ax.bar(["Today", "7-day avg"], [rhr, rhr_7d],
       color=["#e74c3c", "#95a5a6"], width=0.5)
ax.set_ylabel("BPM")
ax.set_title("Resting Heart Rate", fontsize=12)
for i, v in enumerate([rhr, rhr_7d]):
    ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

# 6c. Stress breakdown
ax = axes[2]
stress_data = {
    "Rest": stats.get("restStressDuration", 0) / 60,
    "Low": stats.get("lowStressDuration", 0) / 60,
    "Medium": stats.get("mediumStressDuration", 0) / 60,
    "High": stats.get("highStressDuration", 0) / 60,
    "Activity": stats.get("activityStressDuration", 0) / 60,
}
stress_colors = ["#2ecc71", "#3498db", "#f1c40f", "#e74c3c", "#9b59b6"]
ax.pie(stress_data.values(), labels=stress_data.keys(), colors=stress_colors,
       autopct=lambda p: f"{p:.0f}%" if p > 3 else "", startangle=90)
ax.set_title(f"Stress (avg: {stats.get('averageStressLevel', '?')})", fontsize=12)

# 6d. Steps progress
ax = axes[3]
steps = stats.get("totalSteps", 0)
goal = stats.get("dailyStepGoal", 6740)
pct = min(steps / goal * 100, 100)
ax.barh(["Steps"], [pct], color="#3498db", height=0.5)
ax.barh(["Steps"], [100 - pct], left=[pct], color="#ecf0f1", height=0.5)
ax.set_xlim(0, 100)
ax.text(50, 0, f"{steps:,} / {goal:,}", ha="center", va="center",
        fontsize=12, fontweight="bold")
ax.set_title("Steps Progress", fontsize=12)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "06_daily_wellness.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 06_daily_wellness.png")

# ============================================================
# 7. ACTIVITY TIMELINE — Calendar heatmap style
# ============================================================
print("📊 Chart 7: Activity Timeline")

fig, ax = plt.subplots(figsize=(14, 4))

sport_y = {s: i for i, s in enumerate(df_act["sport_label"].unique())}
sport_colors_map = {
    "⛷️ Skiing": "#2196F3",
    "🏋️ Indoor Cardio": "#FF5722",
    "🚵 MTB": "#4CAF50",
    "🏃 Running": "#FF9800",
    "🥾 Hiking": "#795548",
}

for _, row in df_act.iterrows():
    sport = row["sport_label"]
    ax.scatter(
        row["date"], sport_y[sport],
        s=row["calories_active"] * 1.5,
        c=sport_colors_map.get(sport, "#9E9E9E"),
        alpha=0.8, edgecolors="white", linewidth=1.5, zorder=3,
    )
    ax.annotate(
        f"{row['distance_km']:.1f}km",
        (row["date"], sport_y[sport]),
        textcoords="offset points", xytext=(0, 15),
        ha="center", fontsize=8, color="#555",
    )

ax.set_yticks(list(sport_y.values()))
ax.set_yticklabels(list(sport_y.keys()))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_title("Activity Timeline (bubble size = active calories)",
             fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "07_activity_timeline.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 07_activity_timeline.png")


# ============================================================
# 8. TRAINING LOAD — Garmin's activityTrainingLoad
# ============================================================
print("📊 Chart 8: Training Load")

fig, ax = plt.subplots(figsize=(12, 5))

df_sorted = df_act.sort_values("date")
colors_tl = [sport_colors_map.get(s, "#9E9E9E") for s in df_sorted["sport_label"]]

bars = ax.bar(
    df_sorted["sport_label"] + "\n" + df_sorted["date"].dt.strftime("%b %d"),
    df_sorted["activityTrainingLoad"],
    color=colors_tl, edgecolor="white", linewidth=1.5,
)

for bar, val in zip(bars, df_sorted["activityTrainingLoad"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("Training Load (Garmin)")
ax.set_title("Training Load per Activity", fontsize=14, fontweight="bold")
ax.axhline(y=50, color="#e74c3c", linestyle="--", alpha=0.5, label="High load threshold")
ax.legend()

plt.tight_layout()
plt.savefig(CHARTS_DIR / "08_training_load.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 08_training_load.png")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("✅ All charts generated!")
print(f"📁 Find them in: {CHARTS_DIR.resolve()}")
print("=" * 50)
print(f"\nActivities analyzed: {len(df_act)}")
print(f"Sports: {', '.join(df_act['sport_label'].unique())}")
print(f"Date range: {df_act['date'].min().strftime('%b %d')} — {df_act['date'].max().strftime('%b %d, %Y')}")
print(f"Total active calories: {df_act['calories_active'].sum():.0f} kcal")
print(f"Sleep score: {score}/100")
print(f"HRV status: {hrv['hrvSummary']['status']} (avg {avg}ms)")
print(f"Resting HR: {stats.get('restingHeartRate', '?')} bpm")
