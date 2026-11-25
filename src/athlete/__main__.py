"""Show or edit the athlete profile.

    python -m src.athlete show            # derived profile + where each value came from
    python -m src.athlete show --json     # machine-readable
    python -m src.athlete set max_hr 188
    python -m src.athlete set goal_race_date 2027-04-11
    python -m src.athlete unset max_hr    # go back to the derived value
    python -m src.athlete keys            # list settable keys
"""

from __future__ import annotations

import argparse
import json
import sys

from src.athlete.profile import (
    OVERRIDE_KEYS,
    _fmt_pace,
    load_overrides,
    load_profile,
    overrides_path,
    parse_override,
    save_overrides,
)


def _print_profile(p) -> None:
    s = p.sources
    print(f"Athlete profile as of {p.as_of}\n")
    rows = [
        ("Max HR", p.max_hr, s.get("max_hr")),
        ("Resting HR", p.resting_hr, s.get("resting_hr")),
        ("LTHR", p.lthr, s.get("lthr")),
        ("LT pace", _fmt_pace(p.lt_pace_s_per_km), s.get("lt_pace_s_per_km")),
        ("Gender", p.gender, s.get("gender")),
        ("Age", p.age, s.get("age")),
        ("Weight kg", p.weight_kg, s.get("weight_kg")),
        ("VO2max (run)", p.vo2max_running, s.get("vo2max_running")),
        ("HRV baseline ms", p.hrv_baseline_ms, s.get("hrv_baseline_ms")),
    ]
    for label, val, source in rows:
        print(f"  {label:<16} {'-' if val is None else val!s:<10} {source or ''}")

    print(f"\nHR zones ({s.get('hr_zones')}):")
    garmin = {z.name: z for z in (p.garmin_hr_zones or [])}
    for z in p.hr_zones:
        g = garmin.get(z.name)
        gtxt = f"   (Garmin: {g.low_bpm}-{g.high_bpm - 1})" if g else ""
        print(f"  {z.name:<14} {z.low_bpm:>3}-{z.high_bpm - 1:<3} bpm{gtxt}")

    if p.goal:
        d = p.to_dict()["goal"]
        print("\nGoal race:")
        for k in ("name", "date", "distance_km", "target_time", "weeks_out"):
            if d.get(k) is not None:
                print(f"  {k:<16} {d[k]}")
        if d.get("target_pace_s_per_km"):
            print(f"  {'target_pace':<16} {_fmt_pace(d['target_pace_s_per_km'])}")
    if p.training_days_per_week or p.long_run_day:
        print(f"\nTraining days/week: {p.training_days_per_week or '-'}   long run day: {p.long_run_day or '-'}")
    if p.notes:
        print("\nNotes:")
        for n in p.notes:
            print(f"  - {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.athlete", description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="SQLite path (default: LOCAL_DB_PATH)")
    ap.add_argument("--file", help="Overrides JSON (default: ATHLETE_PROFILE_PATH)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("--json", action="store_true")
    st = sub.add_parser("set")
    st.add_argument("key")
    st.add_argument("value")
    us = sub.add_parser("unset")
    us.add_argument("key")
    sub.add_parser("keys")
    a = ap.parse_args(argv)

    if a.cmd == "keys":
        for k, desc in OVERRIDE_KEYS.items():
            print(f"  {k:<24} {desc}")
        return 0

    if a.cmd in ("set", "unset"):
        data = load_overrides(a.file)
        if a.cmd == "set":
            try:
                data[a.key] = parse_override(a.key, a.value)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
        else:
            data.pop(a.key, None)
        path = save_overrides(data, a.file)
        print(f"Saved {path}")
        return 0

    from src.storage.local_store import LocalStore

    profile = load_profile(LocalStore(a.db), overrides_file=a.file)
    if a.json:
        print(json.dumps(profile.to_dict(), indent=2, default=str))
    else:
        _print_profile(profile)
        print(f"\nOverrides file: {overrides_path(a.file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
