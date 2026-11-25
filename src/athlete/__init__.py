"""Athlete profile: personal HR values, zones and goal race.

Usage::

    from src.athlete import load_profile
    profile = load_profile(store)          # store = LocalStore()
    profile.max_hr, profile.resting_hr, profile.hr_zones, profile.goal
    profile.to_dict()                      # plain dict for prompts / planners
"""

from src.athlete.profile import (  # noqa: F401
    AthleteProfile,
    GoalRace,
    HRZone,
    build_profile,
    load_overrides,
    load_profile,
    save_overrides,
)
