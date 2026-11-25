"""Tests for the athlete profile and the features that consume it."""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.athlete.profile import (
    build_profile,
    gather_inputs,
    load_overrides,
    load_profile,
    observed_max_hr,
    parse_override,
    save_overrides,
    zones_from,
)
from src.features.intensity import intensity_distribution
from src.features.sleep_recovery import compute_recovery_score
from src.features.training_load import compute_acwr, compute_daily_load
from src.storage.local_store import LocalStore

TODAY = date(2026, 11, 25)

GARMIN_ZONES = [
    {"sport": "DEFAULT", "trainingMethod": "HR_MAX", "maxHeartRateUsed": 186,
     "restingHeartRateUsed": 55, "lactateThresholdHeartRateUsed": 168,
     "zone1Floor": 93, "zone2Floor": 111, "zone3Floor": 130, "zone4Floor": 148, "zone5Floor": 167},
]
LT = {"speed_and_heart_rate": {"heartRate": 170, "speed": 0.3333, "calendarDate": "2026-10-01"}}


def rhr_days(n: int, value: float = 48.0, end: date = TODAY):
    return [((end - timedelta(days=i)).isoformat(), value) for i in range(n)]


def base_inputs(**kw):
    d = {
        "today": TODAY,
        "activity_max_hr": [("2026-01-01", v) for v in (181, 183, 179, 175)],
        "resting_hr_daily": rhr_days(20),
        "hrv_daily": rhr_days(10, 62.0),
        "vo2max_running": 54.0,
        "weight_kg": 72.5,
        "heart_rate_zones": GARMIN_ZONES,
        "lactate_threshold": LT,
        "user_profile": {"userData": {"gender": "MALE", "birthDate": "1990-06-15"}},
    }
    d.update(kw)
    return d


# ------------------------------------------------------------------ derivation
class TestObservedMaxHR:
    def test_single_spike_is_ignored(self):
        assert observed_max_hr([224, 186, 184, 170]) == 186

    def test_implausible_values_dropped(self):
        assert observed_max_hr([250, 60, 180, 178]) == 180

    def test_nothing_confirmed_uses_second_highest(self):
        assert observed_max_hr([200, 180, 160]) == 180

    def test_empty(self):
        assert observed_max_hr([]) is None


class TestBuildProfile:
    def test_from_garmin_and_history(self):
        p = build_profile(base_inputs())
        assert p.max_hr == 186  # Garmin setting >= observed 183
        assert p.sources["max_hr"].startswith("garmin")
        assert p.resting_hr == 48
        assert p.sources["resting_hr"].startswith("history")
        assert p.lthr == 170  # latest LT test beats the zone setting
        assert p.zone_method == "lthr"
        assert p.gender == "male" and p.age == 36
        assert p.hrv_baseline_ms == 62.0
        assert p.lt_pace_s_per_km == pytest.approx(300.0, abs=0.5)  # 3.33 m/s = 5:00/km
        assert [z.low_bpm for z in p.hr_zones] == [117, 144, 153, 162, 170]
        assert p.hr_zones[-1].high_bpm > p.max_hr
        assert [z.low_bpm for z in p.garmin_hr_zones] == [93, 111, 130, 148, 167]

    def test_observed_max_above_garmin_wins_and_notes(self):
        p = build_profile(base_inputs(activity_max_hr=[("2026-01-01", 194), ("2026-02-01", 192)]))
        assert p.max_hr == 194
        assert p.sources["max_hr"].startswith("history")
        assert any("194" in n for n in p.notes)

    def test_overrides_win(self):
        p = build_profile(base_inputs(), {
            "max_hr": 190, "resting_hr": 45, "lthr": 172, "gender": "female",
            "zone_method": "hrr",
        })
        assert (p.max_hr, p.resting_hr, p.lthr, p.gender) == (190, 45, 172, "female")
        assert p.sources["max_hr"] == "override"
        assert p.zone_method == "hrr"
        assert p.hr_zones[0].low_bpm == round(45 + 0.5 * 145)

    def test_custom_floors(self):
        p = build_profile(base_inputs(), {"hr_zone_floors": [120, 135, 150, 162, 172]})
        assert p.zone_method == "custom"
        assert p.zone_for(155) == "Z3 Tempo"
        assert p.zone_for(110) == "below Z1"
        assert p.zone_for(186) == "Z5 VO2max"

    def test_garmin_method(self):
        p = build_profile(base_inputs(), {"zone_method": "garmin"})
        assert [z.low_bpm for z in p.hr_zones] == [93, 111, 130, 148, 167]

    def test_empty_store_falls_back_to_defaults(self):
        p = build_profile({"today": TODAY})
        assert (p.max_hr, p.resting_hr, p.lthr) == (190, 60, None)
        assert p.zone_method == "hrr"
        assert p.sources["max_hr"] == "default"
        assert p.notes

    def test_age_estimate_when_no_hr_data(self):
        p = build_profile({"today": TODAY}, {"birth_date": "1986-01-01"})
        assert p.max_hr == round(208 - 0.7 * 40)

    def test_implausible_lthr_dropped(self):
        p = build_profile(base_inputs(lactate_threshold={"speed_and_heart_rate": {"heartRate": 199}},
                                      heart_rate_zones=None))
        assert p.lthr is None and p.zone_method == "hrr"

    def test_resting_hr_varies_by_date(self):
        old = rhr_days(10, 60.0, end=TODAY - timedelta(days=400))
        p = build_profile(base_inputs(resting_hr_daily=old + rhr_days(10, 48.0)))
        assert p.resting_hr_on(TODAY - timedelta(days=402)) == 60.0
        assert p.resting_hr_on(TODAY) == 48.0

    def test_goal_race(self):
        p = build_profile(base_inputs(), {
            "goal_race_name": "Valencia", "goal_race_date": "2027-04-11",
            "goal_race_distance": 42.195, "goal_target_time": "3:15:00",
        })
        g = p.to_dict()["goal"]
        assert g["weeks_out"] == 19.6
        assert g["target_pace_s_per_km"] == pytest.approx(277.3, abs=0.1)
        assert p.to_prompt_dict()["goal_race"]["target_pace"] == "4:37/km"

    def test_zones_cover_range_without_gaps(self):
        for m in ("hrr", "max", "lthr"):
            zs = zones_from(m, 186, 48, 170)
            for a, b in zip(zs, zs[1:]):
                assert a.high_bpm == b.low_bpm


class TestOverrides:
    def test_parse(self):
        assert parse_override("goal_race_distance", "half") == pytest.approx(21.0975)
        assert parse_override("goal_race_distance", "30") == 30.0
        assert parse_override("lt_pace", "4:30") == 270
        assert parse_override("hr_zone_floors", "120, 135,150,162,172") == [120, 135, 150, 162, 172]
        for k, v in [("max_hr", "300"), ("gender", "x"), ("hr_zone_floors", "1,2,3"),
                     ("goal_race_date", "next week"), ("nope", "1")]:
            with pytest.raises(ValueError):
                parse_override(k, v)

    def test_roundtrip(self, tmp_path):
        f = str(tmp_path / "athlete.json")
        save_overrides({"max_hr": 188, "bogus": 1}, f)
        assert load_overrides(f) == {"max_hr": 188}


def test_load_profile_from_store(tmp_path):
    store = LocalStore(":memory:")
    for i, mx in enumerate((184, 182, 150)):
        store.upsert_activity({"activity_id": i, "date": "2026-11-0%d" % (i + 1), "max_hr": mx})
    for d, v in rhr_days(10, 50):
        store.upsert_wellness({"date": d, "resting_hr": v})
    store.put_raw("heart_rate_zones", "2026-11-20", [dict(GARMIN_ZONES[0], maxHeartRateUsed=180)])
    store.put_raw("heart_rate_zones", "2026-11-25", GARMIN_ZONES)
    store.commit()

    inputs = gather_inputs(store, TODAY)
    assert inputs["heart_rate_zones"][0]["maxHeartRateUsed"] == 186  # latest snapshot
    p = load_profile(store, today=TODAY, overrides_file=str(tmp_path / "none.json"))
    assert p.max_hr == 186 and p.resting_hr == 50


# ------------------------------------------------------------------ features
def _acts(rows):
    return pd.DataFrame(rows, columns=["startTimeLocal", "duration", "averageHR"])


class TestLoadWithProfile:
    def test_profile_changes_trimp(self):
        acts = _acts([("2026-11-20 07:00:00", 3600, 150)])
        generic = compute_daily_load(acts)["daily_trimp"].iloc[0]
        p = build_profile(base_inputs())  # max 186, rest 48
        personal = compute_daily_load(acts, profile=p)["daily_trimp"].iloc[0]
        assert personal > generic  # 150 bpm is a bigger share of a smaller reserve

    def test_missing_hr_is_zero_not_140(self):
        acts = _acts([("2026-11-20 07:00:00", 3600, None)])
        out = compute_daily_load(acts)
        assert out["daily_trimp"].iloc[0] == 0
        assert out["n_no_hr"].iloc[0] == 1

    def test_acwr_counts_rest_days(self):
        acts = _acts([("2026-11-01 07:00:00", 3600, 150), ("2026-11-10 07:00:00", 3600, 150)])
        acwr = compute_acwr(compute_daily_load(acts), end=date(2026, 11, 15))
        assert len(acwr) == 15
        assert acwr["daily_trimp"].iloc[1] == 0
        assert str(acwr["date"].iloc[-1]) == "2026-11-15"


class TestIntensity:
    def test_laps_assign_to_zones(self):
        p = build_profile(base_inputs(), {"hr_zone_floors": [120, 135, 150, 162, 172]})
        acts = [
            {"activityId": 1, "duration": 3600, "averageHR": 140},
            {"activityId": 2, "duration": 1200, "averageHR": 140},
            {"activityId": 3, "duration": 600},
        ]
        splits = {"1": {"lapDTOs": [
            {"duration": 2400, "averageHR": 140},  # Z2
            {"duration": 600, "averageHR": 165},   # Z4
            {"duration": 600, "averageHR": 175},   # Z5
        ]}}
        d = intensity_distribution(p, acts, splits)
        assert d["minutes"]["Z2 Endurance"] == 60.0  # 40 from laps + 20 from activity avg
        assert d["minutes"]["Z4 Threshold"] == 10.0
        assert d["hard_pct"] == pytest.approx(25.0)
        assert (d["activities_from_laps"], d["activities_from_avg_hr"], d["activities_without_hr"]) == (1, 1, 1)


def test_recovery_uses_personal_hrv_baseline():
    low_baseline = compute_recovery_score(None, hrv_status=45, hrv_baseline=45)
    generic = compute_recovery_score(None, hrv_status=45)
    assert low_baseline["recovery_score"] == 90.0
    assert generic["recovery_score"] == 75.0
