"""Build the athlete profile from stored Garmin history plus manual overrides.

Every value in the profile records where it came from (``profile.sources``):

``override``            set by you with ``python -m src.athlete set ...``
``garmin:<dataset>``    read from a Garmin settings snapshot the sync stored
``history:<how>``       derived from your stored activities / daily wellness
``estimate:<formula>``  formula fallback (e.g. age-predicted max HR)
``default``             nothing better available

Precedence, per value (first one that exists wins):

max HR      override > max(Garmin zone setting, observed max in last 3 years)
            > Tanaka 208 - 0.7*age > 190
resting HR  override > median of last 30 days of daily RHR (>= 7 days)
            > Garmin zone setting > 60
LTHR        override > Garmin latest lactate threshold > Garmin zone setting
            > user settings
zones       override floors > zone_method (override, else ``lthr`` when an
            LTHR is known, else ``hrr``)

Garmin field names below come from the python-garminconnect docs; the ones
marked ``# GUESS`` have not been checked against a real payload yet.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

ZONE_NAMES = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 VO2max"]
ZONE_METHODS = ("lthr", "hrr", "max", "garmin")

# Zone floors per method, as fractions. Z5 runs up to max HR.
_HRR_FLOORS = (0.50, 0.60, 0.70, 0.80, 0.90)       # Karvonen, % of HR reserve
_MAX_FLOORS = (0.50, 0.60, 0.70, 0.80, 0.90)       # % of max HR
# Friel running zones collapsed to 5: Z1 <85%, Z2 85-89, Z3 90-94, Z4 95-99, Z5 100+
_LTHR_FLOORS = (None, 0.85, 0.90, 0.95, 1.00)       # % of LTHR (Z1 floor = 50% HRR)

DEFAULT_MAX_HR = 190
DEFAULT_RESTING_HR = 60
MAX_HR_LOOKBACK_DAYS = 3 * 365
RHR_WINDOW_DAYS = 30
RHR_MIN_DAYS = 7
HR_PLAUSIBLE = (100, 230)

RACE_DISTANCES_KM = {
    "5k": 5.0, "10k": 10.0, "15k": 15.0, "half": 21.0975, "half_marathon": 21.0975,
    "hm": 21.0975, "marathon": 42.195, "m": 42.195, "50k": 50.0,
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class HRZone:
    name: str
    low_bpm: int    # inclusive
    high_bpm: int   # exclusive, except Z5 which includes max HR

    def contains(self, hr: float) -> bool:
        return self.low_bpm <= hr < self.high_bpm


@dataclass
class GoalRace:
    name: str | None = None
    date: str | None = None          # YYYY-MM-DD
    distance_km: float | None = None
    target_time: str | None = None   # H:MM:SS

    @property
    def target_seconds(self) -> int | None:
        return _parse_duration(self.target_time) if self.target_time else None

    @property
    def target_pace_s_per_km(self) -> float | None:
        s = self.target_seconds
        return round(s / self.distance_km, 1) if s and self.distance_km else None

    def weeks_out(self, today: date) -> float | None:
        if not self.date:
            return None
        return round((date.fromisoformat(self.date) - today).days / 7, 1)


@dataclass
class AthleteProfile:
    as_of: str
    max_hr: int
    resting_hr: int
    lthr: int | None = None
    lt_pace_s_per_km: float | None = None
    gender: str = "male"
    age: int | None = None
    weight_kg: float | None = None
    vo2max_running: float | None = None
    hrv_baseline_ms: float | None = None
    zone_method: str = "hrr"
    hr_zones: list[HRZone] = field(default_factory=list)
    garmin_hr_zones: list[HRZone] | None = None
    goal: GoalRace | None = None
    training_days_per_week: int | None = None
    long_run_day: str | None = None
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Daily RHR (30-day rolling median) so old activities are scored with the
    # resting HR you had back then. Not serialised.
    resting_hr_by_date: dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ helpers
    def zone_for(self, hr: float | None) -> str | None:
        if hr is None or hr != hr:  # None or NaN
            return None
        if hr < self.hr_zones[0].low_bpm:
            return "below Z1"
        for z in self.hr_zones:
            if z.contains(hr):
                return z.name
        return self.hr_zones[-1].name

    def zone_index(self, hr: float | None) -> int | None:
        """0..4 for Z1..Z5, -1 below Z1, None when HR is missing."""
        name = self.zone_for(hr)
        if name is None:
            return None
        return -1 if name == "below Z1" else ZONE_NAMES.index(name)

    def resting_hr_on(self, day: str | date) -> float:
        """Resting HR as it was around ``day`` (falls back to the current value)."""
        key = day.isoformat() if isinstance(day, date) else str(day)[:10]
        if self.sources.get("resting_hr") == "override" or not self.resting_hr_by_date:
            return self.resting_hr
        if key in self.resting_hr_by_date:
            return self.resting_hr_by_date[key]
        # nearest earlier day we have
        earlier = [d for d in self.resting_hr_by_date if d <= key]
        return self.resting_hr_by_date[max(earlier)] if earlier else self.resting_hr

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able dict (for prompts, the API and the planner)."""
        d = asdict(self)
        d.pop("resting_hr_by_date", None)
        d["hr_zones"] = [asdict(z) for z in self.hr_zones]
        if self.goal:
            d["goal"].update(
                weeks_out=self.goal.weeks_out(date.fromisoformat(self.as_of)),
                target_pace_s_per_km=self.goal.target_pace_s_per_km,
            )
        return d

    def to_prompt_dict(self) -> dict[str, Any]:
        """Compact version for the LLM prompt."""
        d = {
            "max_hr": self.max_hr,
            "resting_hr": self.resting_hr,
            "lthr": self.lthr,
            "lt_pace": _fmt_pace(self.lt_pace_s_per_km),
            "gender": self.gender,
            "age": self.age,
            "weight_kg": self.weight_kg,
            "vo2max_running": self.vo2max_running,
            "hr_zones_bpm": {z.name: f"{z.low_bpm}-{z.high_bpm}" for z in self.hr_zones},
            "zone_method": self.zone_method,
        }
        if self.goal:
            g = self.to_dict()["goal"]
            d["goal_race"] = {
                "name": g["name"], "date": g["date"], "distance_km": g["distance_km"],
                "target_time": g["target_time"], "weeks_out": g["weeks_out"],
                "target_pace": _fmt_pace(g["target_pace_s_per_km"]),
            }
        if self.training_days_per_week:
            d["training_days_per_week"] = self.training_days_per_week
        if self.long_run_day:
            d["long_run_day"] = self.long_run_day
        return {k: v for k, v in d.items() if v is not None}


# --------------------------------------------------------------------------- #
# Overrides file
# --------------------------------------------------------------------------- #

OVERRIDE_KEYS: dict[str, str] = {
    "max_hr": "int bpm",
    "resting_hr": "int bpm",
    "lthr": "int bpm (lactate threshold HR)",
    "lt_pace": "M:SS per km",
    "gender": "male | female",
    "birth_date": "YYYY-MM-DD",
    "weight_kg": "float",
    "zone_method": " | ".join(ZONE_METHODS),
    "hr_zone_floors": "5 comma-separated bpm floors, e.g. 120,135,150,162,172",
    "goal_race_name": "text",
    "goal_race_date": "YYYY-MM-DD",
    "goal_race_distance": "km, or 5k|10k|half|marathon",
    "goal_target_time": "H:MM:SS",
    "training_days_per_week": "int",
    "long_run_day": "monday..sunday",
}


def parse_override(key: str, value: str) -> Any:
    """Validate and convert a CLI value for ``key``. Raises ValueError."""
    if key not in OVERRIDE_KEYS:
        raise ValueError(f"unknown key {key!r}; one of: {', '.join(OVERRIDE_KEYS)}")
    v = value.strip()
    if key in ("max_hr", "resting_hr", "lthr"):
        n = int(v)
        if not 25 <= n <= 230:
            raise ValueError(f"{key} {n} bpm is not plausible")
        return n
    if key == "training_days_per_week":
        n = int(v)
        if not 1 <= n <= 7:
            raise ValueError("training_days_per_week must be 1..7")
        return n
    if key == "weight_kg":
        return float(v)
    if key == "lt_pace":
        return _parse_duration(v)  # seconds per km
    if key == "gender":
        if v.lower() not in ("male", "female"):
            raise ValueError("gender must be male or female (sets TRIMP coefficients)")
        return v.lower()
    if key in ("birth_date", "goal_race_date"):
        return date.fromisoformat(v).isoformat()
    if key == "zone_method":
        if v not in ZONE_METHODS:
            raise ValueError(f"zone_method must be one of {ZONE_METHODS}")
        return v
    if key == "hr_zone_floors":
        floors = [int(x) for x in re.split(r"[,\s]+", v) if x]
        if len(floors) != 5 or floors != sorted(floors):
            raise ValueError("need 5 increasing floors, Z1..Z5")
        return floors
    if key == "goal_race_distance":
        return RACE_DISTANCES_KM.get(v.lower().replace("-", "_").replace(" ", "_")) or float(v)
    if key == "goal_target_time":
        _parse_duration(v)
        return v
    if key == "long_run_day":
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if v.lower() not in days:
            raise ValueError("long_run_day must be a weekday name")
        return v.lower()
    return v


def overrides_path(path: str | None = None) -> Path:
    from config.settings import settings

    return Path(path or settings.athlete_profile_path)


def load_overrides(path: str | None = None) -> dict[str, Any]:
    p = overrides_path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} is not valid JSON: {e}") from e
    unknown = set(data) - set(OVERRIDE_KEYS)
    if unknown:
        logger.warning("Ignoring unknown keys in %s: %s", p, ", ".join(sorted(unknown)))
    return {k: v for k, v in data.items() if k in OVERRIDE_KEYS and v is not None}


def save_overrides(data: dict[str, Any], path: str | None = None) -> Path:
    p = overrides_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return p


# --------------------------------------------------------------------------- #
# Reading inputs from the store
# --------------------------------------------------------------------------- #

def gather_inputs(store, today: date | None = None) -> dict[str, Any]:
    """Pull everything the profile needs out of a ``LocalStore``."""
    today = today or date.today()
    c = store.conn
    since = (today - timedelta(days=MAX_HR_LOOKBACK_DAYS)).isoformat()
    act_max = [
        (r[0], r[1]) for r in c.execute(
            "SELECT date, max_hr FROM activities WHERE max_hr IS NOT NULL AND date >= ?",
            (since,),
        )
    ]
    rhr = [(r[0], r[1]) for r in c.execute(
        "SELECT date, resting_hr FROM daily_wellness WHERE resting_hr IS NOT NULL ORDER BY date"
    )]
    hrv = [(r[0], r[1]) for r in c.execute(
        "SELECT date, hrv_last_night_avg FROM daily_wellness "
        "WHERE hrv_last_night_avg IS NOT NULL AND date >= ? ORDER BY date",
        ((today - timedelta(days=RHR_WINDOW_DAYS)).isoformat(),),
    )]

    def latest_value(col: str) -> Any:
        r = c.execute(
            f"SELECT {col} FROM daily_wellness WHERE {col} IS NOT NULL ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return r[0] if r else None

    return {
        "today": today,
        "activity_max_hr": act_max,
        "resting_hr_daily": rhr,
        "hrv_daily": hrv,
        "vo2max_running": latest_value("vo2max_running"),
        "weight_kg": latest_value("weight_kg"),
        "heart_rate_zones": _latest_snapshot(store, "heart_rate_zones"),
        "lactate_threshold": _latest_snapshot(store, "lactate_threshold_latest"),
        "user_profile": _latest_snapshot(store, "user_profile"),
    }


def _latest_snapshot(store, dataset: str) -> Any:
    r = store.conn.execute(
        "SELECT payload FROM raw_responses WHERE dataset=? AND status='ok' "
        "AND payload IS NOT NULL ORDER BY key DESC LIMIT 1",
        (dataset,),
    ).fetchone()
    return json.loads(r[0]) if r else None


# --------------------------------------------------------------------------- #
# Derivation (pure functions, unit-tested)
# --------------------------------------------------------------------------- #

def garmin_zone_entry(zones_payload: Any) -> dict | None:
    """Pick the running zone set from get_heart_rate_zones(), else the default one."""
    if not isinstance(zones_payload, list):
        return None
    by_sport = {str(z.get("sport") or "").upper(): z for z in zones_payload if isinstance(z, dict)}
    return by_sport.get("RUNNING") or by_sport.get("DEFAULT") or (
        next(iter(by_sport.values())) if by_sport else None
    )


def observed_max_hr(values: Iterable[float], confirm_within: int = 4) -> int | None:
    """Highest activity max HR that a second activity comes within ``confirm_within`` bpm of.

    A single spike (strap glitch, optical cadence lock) is ignored; a real
    max shows up more than once over three years.
    """
    v = sorted(
        (float(x) for x in values if x is not None and HR_PLAUSIBLE[0] <= float(x) <= HR_PLAUSIBLE[1]),
        reverse=True,
    )
    for i in range(len(v) - 1):
        if v[i] - v[i + 1] <= confirm_within:
            return int(round(v[i]))
    # nothing confirmed: the second-highest is the safer guess than a lone spike
    return int(round(v[1])) if len(v) >= 2 else (int(round(v[0])) if v else None)


def rolling_resting_hr(daily: list[tuple[str, float]], window: int = RHR_WINDOW_DAYS) -> dict[str, float]:
    """Trailing ``window``-day median RHR for every day that has data."""
    import pandas as pd

    if not daily:
        return {}
    s = pd.Series({d: float(v) for d, v in daily if v and v > 20})
    if s.empty:
        return {}
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    med = s.rolling(f"{window}D", min_periods=1).median()
    return {d.date().isoformat(): round(float(v), 1) for d, v in med.items()}


def lt_speed_to_pace(speed: Any) -> float | None:
    """Garmin LT speed -> seconds per km.

    Garmin seems to store LT speed in units of 10 m/s (0.34 == 3.4 m/s).  # GUESS
    Values >= 1 are taken as m/s already.
    """
    if not isinstance(speed, (int, float)) or speed <= 0:
        return None
    mps = speed * 10 if speed < 1 else speed
    return round(1000 / mps, 1)


def zones_from(method: str, max_hr: int, resting_hr: int, lthr: int | None) -> list[HRZone]:
    if method == "lthr":
        if not lthr:
            raise ValueError("lthr method needs an LTHR")
        z1_floor = resting_hr + 0.5 * (max_hr - resting_hr)
        floors = [z1_floor] + [lthr * f for f in _LTHR_FLOORS[1:]]
    elif method == "hrr":
        floors = [resting_hr + f * (max_hr - resting_hr) for f in _HRR_FLOORS]
    elif method == "max":
        floors = [f * max_hr for f in _MAX_FLOORS]
    else:
        raise ValueError(f"unknown zone method {method!r}")
    return zones_from_floors([int(round(f)) for f in floors], max_hr)


def zones_from_floors(floors: list[int], max_hr: int) -> list[HRZone]:
    top = max(max_hr, floors[-1] + 1) + 1  # Z5 includes max HR
    highs = floors[1:] + [top]
    return [HRZone(n, lo, hi) for n, lo, hi in zip(ZONE_NAMES, floors, highs)]


def garmin_zones(entry: dict | None) -> list[HRZone] | None:
    if not entry:
        return None
    floors = [entry.get(f"zone{i}Floor") for i in range(1, 6)]
    mx = entry.get("maxHeartRateUsed")
    if any(not isinstance(f, (int, float)) for f in floors) or not mx:
        return None
    return zones_from_floors([int(f) for f in floors], int(mx))


def build_profile(inputs: dict[str, Any], overrides: dict[str, Any] | None = None) -> AthleteProfile:
    """Combine stored data and overrides into an ``AthleteProfile``."""
    o = dict(overrides or {})
    today: date = inputs.get("today") or date.today()
    src: dict[str, str] = {}
    notes: list[str] = []

    zentry = garmin_zone_entry(inputs.get("heart_rate_zones"))
    lt = (inputs.get("lactate_threshold") or {}).get("speed_and_heart_rate") or {}
    user = (inputs.get("user_profile") or {}).get("userData") or {}  # GUESS: shape

    # ---- age / gender / weight
    age = None
    birth = o.get("birth_date") or user.get("birthDate")  # GUESS
    if birth:
        try:
            b = date.fromisoformat(str(birth)[:10])
            age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
            src["age"] = "override" if "birth_date" in o else "garmin:user_profile"
        except ValueError:
            pass

    gender = o.get("gender")
    if gender:
        src["gender"] = "override"
    elif str(user.get("gender") or "").upper() in ("MALE", "FEMALE"):  # GUESS
        gender = str(user["gender"]).lower()
        src["gender"] = "garmin:user_profile"
    else:
        gender = "male"
        src["gender"] = "default"
        notes.append("Gender unknown; TRIMP uses male coefficients. Set it with `set gender`.")

    weight = o.get("weight_kg")
    if weight:
        src["weight_kg"] = "override"
    elif inputs.get("weight_kg"):
        weight = inputs["weight_kg"]
        src["weight_kg"] = "history:latest weigh-in"
    elif isinstance(user.get("weight"), (int, float)):  # GUESS: grams
        weight = round(user["weight"] / 1000, 1)
        src["weight_kg"] = "garmin:user_profile"

    # ---- max HR
    garmin_max = zentry.get("maxHeartRateUsed") if zentry else None
    seen_max = observed_max_hr(v for _, v in inputs.get("activity_max_hr") or [])
    if o.get("max_hr"):
        max_hr = int(o["max_hr"])
        src["max_hr"] = "override"
    elif garmin_max or seen_max:
        if garmin_max and (not seen_max or garmin_max >= seen_max):
            max_hr, src["max_hr"] = int(garmin_max), "garmin:heart_rate_zones"
        else:
            max_hr, src["max_hr"] = int(seen_max), "history:highest confirmed activity max HR, last 3 years"
        if garmin_max and seen_max and abs(garmin_max - seen_max) >= 5:
            notes.append(
                f"Garmin's max HR setting is {garmin_max} but your activities reached {seen_max}; "
                f"using {max_hr}. Override with `set max_hr` if you know better."
            )
    elif age:
        max_hr, src["max_hr"] = int(round(208 - 0.7 * age)), "estimate:Tanaka 208-0.7*age"
    else:
        max_hr, src["max_hr"] = DEFAULT_MAX_HR, "default"
        notes.append("No max HR data; using 190. Run the backfill or `set max_hr`.")

    # ---- resting HR
    rhr_by_date = rolling_resting_hr(inputs.get("resting_hr_daily") or [])
    recent_cut = (today - timedelta(days=RHR_WINDOW_DAYS)).isoformat()
    recent = sorted(v for d, v in (inputs.get("resting_hr_daily") or []) if d >= recent_cut and v)
    if o.get("resting_hr"):
        resting_hr, src["resting_hr"] = int(o["resting_hr"]), "override"
    elif len(recent) >= RHR_MIN_DAYS:
        resting_hr = int(round(_median(recent)))
        src["resting_hr"] = f"history:median of last {RHR_WINDOW_DAYS} days ({len(recent)} days)"
    elif zentry and zentry.get("restingHeartRateUsed"):
        resting_hr, src["resting_hr"] = int(zentry["restingHeartRateUsed"]), "garmin:heart_rate_zones"
    else:
        resting_hr, src["resting_hr"] = DEFAULT_RESTING_HR, "default"
        notes.append("No recent resting HR; using 60. Run the sync or `set resting_hr`.")

    # ---- lactate threshold
    lthr = None
    if o.get("lthr"):
        lthr, src["lthr"] = int(o["lthr"]), "override"
    elif lt.get("heartRate"):
        lthr, src["lthr"] = int(lt["heartRate"]), "garmin:lactate_threshold_latest"
        if lt.get("calendarDate"):
            src["lthr"] += f" ({str(lt['calendarDate'])[:10]})"
    elif zentry and zentry.get("lactateThresholdHeartRateUsed"):
        lthr, src["lthr"] = int(zentry["lactateThresholdHeartRateUsed"]), "garmin:heart_rate_zones"
    elif user.get("lactateThresholdHeartRate"):  # GUESS
        lthr, src["lthr"] = int(user["lactateThresholdHeartRate"]), "garmin:user_profile"
    if lthr and not (resting_hr < lthr < max_hr):
        notes.append(f"LTHR {lthr} is outside resting..max ({resting_hr}..{max_hr}); ignoring it.")
        lthr = None
        src.pop("lthr", None)

    lt_pace = None
    if o.get("lt_pace"):
        lt_pace, src["lt_pace_s_per_km"] = float(o["lt_pace"]), "override"
    else:
        lt_pace = lt_speed_to_pace(lt.get("speed")) or lt_speed_to_pace(
            user.get("lactateThresholdSpeed")  # GUESS
        )
        if lt_pace:
            src["lt_pace_s_per_km"] = "garmin:lactate_threshold (unit is a guess)"

    # ---- zones
    garmin_z = garmin_zones(zentry)
    method = o.get("zone_method") or ("lthr" if lthr else "hrr")
    if o.get("hr_zone_floors"):
        zones, method = zones_from_floors(list(o["hr_zone_floors"]), max_hr), "custom"
        src["hr_zones"] = "override"
    elif method == "garmin":
        if garmin_z:
            zones, src["hr_zones"] = garmin_z, "garmin:heart_rate_zones"
        else:
            notes.append("zone_method=garmin but no Garmin zones stored; using HR reserve.")
            method = "hrr"
    if "hr_zones" not in src:
        if method == "lthr" and not lthr:
            notes.append("zone_method=lthr but no LTHR known; using HR reserve.")
            method = "hrr"
        zones = zones_from(method, max_hr, resting_hr, lthr)
        src["hr_zones"] = {
            "lthr": "% of LTHR (Friel, 5 zones)",
            "hrr": "% of HR reserve (Karvonen)",
            "max": "% of max HR",
        }[method] + (" [override]" if "zone_method" in o else "")

    # ---- HRV baseline
    hrv_vals = sorted(v for _, v in inputs.get("hrv_daily") or [] if v)
    hrv_baseline = round(_median(hrv_vals), 1) if len(hrv_vals) >= RHR_MIN_DAYS else None
    if hrv_baseline:
        src["hrv_baseline_ms"] = f"history:median of last {RHR_WINDOW_DAYS} days"

    # ---- goal race
    goal = None
    if any(k.startswith("goal_") for k in o):
        goal = GoalRace(
            name=o.get("goal_race_name"),
            date=o.get("goal_race_date"),
            distance_km=o.get("goal_race_distance"),
            target_time=o.get("goal_target_time"),
        )
        src["goal"] = "override"
        if goal.date and goal.date < today.isoformat():
            notes.append(f"Goal race date {goal.date} is in the past.")
    else:
        notes.append("No goal race set. `python -m src.athlete set goal_race_date YYYY-MM-DD` etc.")

    vo2 = inputs.get("vo2max_running")
    if vo2:
        src["vo2max_running"] = "history:latest Garmin VO2max"

    return AthleteProfile(
        as_of=today.isoformat(),
        max_hr=max_hr,
        resting_hr=resting_hr,
        lthr=lthr,
        lt_pace_s_per_km=lt_pace,
        gender=gender,
        age=age,
        weight_kg=weight,
        vo2max_running=vo2,
        hrv_baseline_ms=hrv_baseline,
        zone_method=method,
        hr_zones=zones,
        garmin_hr_zones=garmin_z,
        goal=goal,
        training_days_per_week=o.get("training_days_per_week"),
        long_run_day=o.get("long_run_day"),
        sources=src,
        notes=notes,
        resting_hr_by_date=rhr_by_date,
    )


def load_profile(store=None, today: date | None = None, overrides_file: str | None = None) -> AthleteProfile:
    """Build the profile from the store (default: ``LocalStore()``) and overrides file."""
    if store is None:
        from src.storage.local_store import LocalStore

        store = LocalStore()
    return build_profile(gather_inputs(store, today), load_overrides(overrides_file))


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #

def _median(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    m = n // 2
    return float(sorted_vals[m]) if n % 2 else (sorted_vals[m - 1] + sorted_vals[m]) / 2


def _parse_duration(s: str) -> int:
    """'3:45:00' -> 13500, '4:50' -> 290 (M:SS)."""
    parts = [int(p) for p in str(s).strip().split(":")]
    if not 2 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise ValueError(f"bad duration {s!r}; use H:MM:SS or M:SS")
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _fmt_pace(s_per_km: float | None) -> str | None:
    if not s_per_km:
        return None
    m, s = divmod(int(round(s_per_km)), 60)
    return f"{m}:{s:02d}/km"
