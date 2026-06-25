"""
location_scorer.py — Location and relocation scoring.

The JD prefers Pune/Noida primarily, then Delhi NCR, then other metros.
Candidates willing to relocate get a significant boost even if currently
in a non-preferred city.
"""

from __future__ import annotations

from typing import Any


# ─── Location tier definitions ────────────────────────────────────────────────

_PREFERRED_LOCATIONS: tuple[str, ...] = (
    "noida", "pune",
)

_ACCEPTABLE_LOCATIONS: tuple[str, ...] = (
    "gurgaon", "gurugram", "delhi", "ncr", "new delhi", "faridabad",
    "ghaziabad",
)

_GOOD_LOCATIONS: tuple[str, ...] = (
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai",
    "kolkata", "ahmedabad", "jaipur", "indore",
)


def _normalize_location(location: str) -> str:
    """Lowercase and strip a location string for matching.

    Args:
        location: Raw location string.

    Returns:
        Normalized lowercase location.
    """
    return location.lower().strip()


def score_location(candidate: dict[str, Any]) -> float:
    """Compute the location compatibility score.

    Uses both the candidate's current location and their willingness
    to relocate.

    Args:
        candidate: Full candidate dict.

    Returns:
        Location score in [0.0, 1.0].
    """
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}

    raw_location = str(profile.get("location") or "")
    country = str(profile.get("country") or "").lower()
    willing_to_relocate = bool(signals.get("willing_to_relocate"))
    preferred_work_mode = str(signals.get("preferred_work_mode") or "").lower()

    location = _normalize_location(raw_location)

    # Check preferred tier
    if any(pref in location for pref in _PREFERRED_LOCATIONS):
        return 1.00

    # Check acceptable tier (Delhi NCR)
    if any(acc in location for acc in _ACCEPTABLE_LOCATIONS):
        return 0.90

    # Check good tier (other metros)
    if any(good in location for good in _GOOD_LOCATIONS):
        return 0.85

    # India but not a major tech hub
    if country == "india":
        if willing_to_relocate:
            return 0.75
        else:
            return 0.55

    # Outside India but willing to relocate
    if willing_to_relocate:
        return 0.50

    # Outside India, not willing to relocate
    return 0.20


def get_location_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract location metadata for reasoning generation.

    Args:
        candidate: Full candidate dict.

    Returns:
        Dict with location, country, willing_to_relocate, score.
    """
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}

    return {
        "location": str(profile.get("location") or "Unknown"),
        "country": str(profile.get("country") or "Unknown"),
        "willing_to_relocate": bool(signals.get("willing_to_relocate")),
        "preferred_work_mode": str(signals.get("preferred_work_mode") or "not specified"),
        "score": score_location(candidate),
    }
