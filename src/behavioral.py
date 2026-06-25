"""
behavioral.py — Behavioral multiplier calculation.

The behavioral multiplier is applied MULTIPLICATIVELY to the raw fit score.
It captures whether the candidate is actually hirable right now:

  behavioral_multiplier = recency x availability x responsiveness x notice_score x github_bonus

A perfect-fit candidate who has been inactive for 6 months is NOT actually
hirable and should rank lower than a slightly worse candidate who is active today.

Reference date: computed at runtime via date.today() so scores are always
relative to the actual submission date, never stale.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

# ─── Reference date for activity calculations ─────────────────────────────────

REFERENCE_DATE: date = date.today()


def _parse_date(date_str: Any) -> date | None:
    """Parse a date string into a Python date object.

    Supports ISO 8601 formats: YYYY-MM-DD, YYYY-MM, YYYY.

    Args:
        date_str: Raw date value (str, int, or None).

    Returns:
        Parsed date or None if unparseable.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    # Try YYYY-MM-DD (10 chars)
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    # Try YYYY-MM (7 chars)
    if len(s) >= 7:
        try:
            return datetime.strptime(s[:7], "%Y-%m").date()
        except ValueError:
            pass
    # Try YYYY (4 chars)
    if len(s) >= 4:
        try:
            return datetime.strptime(s[:4], "%Y").date()
        except ValueError:
            pass
    # Try just extracting a 4-digit year via regex
    import re
    m = re.search(r"(\d{4})", s)
    if m:
        try:
            return date(int(m.group(1)), 6, 1)  # mid-year estimate
        except ValueError:
            pass
    return None


def _days_since(date_str: Any) -> int | None:
    """Compute days between a date string and REFERENCE_DATE.

    Args:
        date_str: Raw date string.

    Returns:
        Integer days, or None if date is unparseable.
    """
    d = _parse_date(date_str)
    if d is None:
        return None
    delta = REFERENCE_DATE - d
    return max(0, delta.days)


def _recency_score(last_active_date: Any) -> float:
    """Score candidate recency based on last active date.

    Args:
        last_active_date: Raw date string.

    Returns:
        Recency score in [0.25, 1.00].
    """
    days = _days_since(last_active_date)
    if days is None:
        return 0.70  # neutral when date is missing

    if days < 14:
        return 1.00
    elif days < 30:
        return 0.95
    elif days < 90:
        return 0.80
    elif days < 180:
        return 0.55
    else:
        return 0.25


def _availability_score(open_to_work: Any) -> float:
    """Score availability based on open_to_work_flag.

    Args:
        open_to_work: Boolean or falsy value.

    Returns:
        1.00 if open to work, else 0.75.
    """
    return 1.00 if bool(open_to_work) else 0.75


def _responsiveness_score(recruiter_response_rate: Any) -> float:
    """Score responsiveness using recruiter_response_rate.

    Formula: floor(0.40) + 0.60 × rate

    Args:
        recruiter_response_rate: Float in [0.0, 1.0] or None.

    Returns:
        Score in [0.40, 1.00].
    """
    rate = recruiter_response_rate
    if rate is None or rate == -1:
        return 0.65  # neutral default
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return 0.65
    rate = max(0.0, min(rate, 1.0))
    return 0.40 + 0.60 * rate


def _notice_score(notice_period_days: Any) -> float:
    """Score based on notice period.

    Args:
        notice_period_days: Integer days or None.

    Returns:
        Notice score in [0.50, 1.00].
    """
    # IMPORTANT: use explicit None check, NOT `or 60`.
    # notice_period_days=0 (immediate) is falsy but means "best case" (1.00),
    # not the neutral default of 60 days.
    if notice_period_days is None:
        days = 60  # neutral default when field missing
    else:
        try:
            days = int(notice_period_days)
        except (TypeError, ValueError):
            days = 60

    if days <= 30:
        return 1.00
    elif days <= 60:
        return 0.88
    elif days <= 90:
        return 0.72
    else:
        return 0.50


def _github_bonus(github_activity_score: Any) -> float:
    """Small bonus/penalty based on GitHub activity score.

    Args:
        github_activity_score: Int in [-1, 100] (-1 = no GitHub).

    Returns:
        Multiplier in [0.92, 1.05].
    """
    try:
        score = int(github_activity_score if github_activity_score is not None else -1)
    except (TypeError, ValueError):
        score = -1

    if score == -1:
        return 0.95  # no GitHub linked
    elif score > 60:
        return 1.05
    elif score > 30:
        return 1.00
    else:
        return 0.92


def _interview_score(interview_completion_rate: Any) -> float:
    """Score based on interview completion rate.

    A high rate means the candidate follows through when contacted.

    Args:
        interview_completion_rate: Float in [0.0, 1.0] or -1 (no data).

    Returns:
        Score in [0.85, 1.00].
    """
    try:
        rate = float(interview_completion_rate)
    except (TypeError, ValueError):
        return 1.0  # neutral if missing
    if rate == -1:
        return 1.0  # no data = neutral
    rate = max(0.0, min(rate, 1.0))
    return 0.85 + 0.15 * rate


def _market_validation_score(saved_by_recruiters_30d: Any) -> float:
    """Score based on how many recruiters saved this profile recently.

    Other recruiters competing for this candidate is a signal of quality.

    Args:
        saved_by_recruiters_30d: Integer count or None.

    Returns:
        Multiplier in [1.00, 1.08] (small bonus, never a penalty).
    """
    try:
        saved = int(saved_by_recruiters_30d or 0)
    except (TypeError, ValueError):
        return 1.0
    return min(1.0 + saved * 0.01, 1.08)


def _completeness_score(profile_completeness_score: Any) -> float:
    """Score based on profile completeness.

    Incomplete profiles suggest a less serious candidate.

    Args:
        profile_completeness_score: Integer 0-100 or None.

    Returns:
        Score in [0.85, 1.00].
    """
    try:
        pct = float(profile_completeness_score or 70)
    except (TypeError, ValueError):
        pct = 70.0
    pct = max(0.0, min(pct, 100.0))
    return 0.85 + 0.15 * (pct / 100.0)


def _work_mode_score(preferred_work_mode: Any) -> float:
    """Score based on work mode preference vs JD requirement (hybrid).

    Args:
        preferred_work_mode: String like 'onsite', 'hybrid', 'remote', 'flexible'.

    Returns:
        Score in [0.85, 1.00].
    """
    mode = str(preferred_work_mode or "flexible").lower().strip()
    if mode in ("onsite", "hybrid", "flexible", ""):
        return 1.0
    elif mode == "remote":
        return 0.85  # slight penalty for remote-only preference
    return 1.0  # unknown = neutral


def _offer_acceptance_score(raw_offer: Any) -> float:
    """Score based on offer acceptance rate.

    A high rate means the candidate actually closes — not just interviews
    and then drops out.  -1 means no offers received (neutral).

    Args:
        raw_offer: Float in [0.0, 1.0] or -1 (no offers received).

    Returns:
        Multiplier in [0.90, 1.08].
    """
    try:
        offer_val = float(raw_offer) if raw_offer is not None else -1.0
    except (TypeError, ValueError):
        offer_val = -1.0

    if offer_val == -1.0:
        return 1.00   # no data — neutral
    elif offer_val >= 0.80:
        return 1.08   # highly likely to accept — strong positive
    elif offer_val >= 0.50:
        return 1.03   # moderate acceptance rate — slight positive
    elif offer_val >= 0.20:
        return 0.97   # low acceptance — slight negative
    else:
        return 0.90   # rarely accepts offers — likely picky or ghosting


def _activity_score(signals: dict[str, Any]) -> float:
    """Score based on applications submitted in the last 30 days.

    A candidate marked open_to_work who has applied to 0 jobs is passive
    (lower close probability).  One who applied to 5+ is actively searching.

    Args:
        signals: Full redrob_signals dict.

    Returns:
        Multiplier in [0.93, 1.05].
    """
    raw_apps = signals.get("applications_submitted_30d")
    try:
        apps = int(raw_apps) if raw_apps is not None else 0
    except (TypeError, ValueError):
        apps = 0

    open_flag = bool(signals.get("open_to_work_flag", False))
    if open_flag:
        if apps >= 5:
            return 1.05   # actively searching
        elif apps >= 1:
            return 1.02   # some activity
        else:
            return 0.93   # passive despite open flag
    return 1.00   # neutral when not open to work


def _salary_alignment_score(signals: dict[str, Any]) -> float:
    """Score based on expected salary vs Senior AI Engineer band (~25-60 LPA).

    Candidates expecting Director-level comp (100+ LPA) are over-priced for
    an IC Senior Engineer role at a Series A/B startup.

    Args:
        signals: Full redrob_signals dict.

    Returns:
        Multiplier in [0.80, 1.05].
    """
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    if not isinstance(salary, dict):
        return 1.0
    sal_min = salary.get("min") or 0
    sal_max = salary.get("max") or 0
    try:
        mid = (float(sal_min) + float(sal_max)) / 2
    except (TypeError, ValueError):
        return 1.0
    if mid == 0:
        return 1.0   # no data = neutral
    if 20 <= mid <= 70:
        return 1.05  # squarely aligned with Senior AI Engineer band
    if 70 < mid <= 100:
        return 0.92  # slightly high, manageable
    if mid > 100:
        return 0.80  # Director-level expectations, wrong level
    if mid < 15:
        return 0.88  # likely junior in expectation
    return 1.0


def _verification_bonus(signals: dict[str, Any]) -> float:
    """Small bonus for verified / connected candidates.

    A candidate with a connected LinkedIn profile and verified contact
    details is more serious and reachable than an anonymous listing.

    Args:
        signals: Full redrob_signals dict.

    Returns:
        Additive bonus in [0.0, 0.04].
    """
    bonus = 0.0
    if bool(signals.get("linkedin_connected", False)):
        bonus += 0.02
    if bool(signals.get("verified_email", False)) or bool(signals.get("verified_phone", False)):
        bonus += 0.02
    return bonus


def compute_behavioral_multiplier(candidate: dict[str, Any]) -> float:
    """Compute the full behavioral multiplier for a candidate.

    behavioral_multiplier = (
        recency x availability x responsiveness x notice x github
        x interview x work_mode x salary x offer_acceptance x activity
    )
    market_validation, completeness, and verification applied as additive bonuses.

    This is applied multiplicatively to the raw fit score to produce
    the final score.

    Args:
        candidate: Full candidate dict.

    Returns:
        Behavioral multiplier in approximately [0.0, 1.20].
    """
    signals = candidate.get("redrob_signals", {}) or {}

    recency = _recency_score(signals.get("last_active_date"))
    availability = _availability_score(signals.get("open_to_work_flag"))
    responsiveness = _responsiveness_score(signals.get("recruiter_response_rate"))
    notice = _notice_score(signals.get("notice_period_days"))
    github = _github_bonus(signals.get("github_activity_score"))
    interview = _interview_score(signals.get("interview_completion_rate"))
    work_mode = _work_mode_score(signals.get("preferred_work_mode"))
    market = _market_validation_score(signals.get("saved_by_recruiters_30d"))
    completeness = _completeness_score(signals.get("profile_completeness_score"))
    salary = _salary_alignment_score(signals)
    verification = _verification_bonus(signals)
    offer = _offer_acceptance_score(signals.get("offer_acceptance_rate"))
    activity = _activity_score(signals)

    # Core multiplicative factors
    multiplier = (
        recency * availability * responsiveness * notice * github
        * interview * work_mode * salary * offer * activity
    )
    # Additive boosts (market validation, completeness, verification)
    bonus = (market - 1.0) + (completeness - 0.85) * 0.5 + verification
    multiplier = multiplier + bonus
    # Compress range: raw multiplier can span ~0.30–0.80 which lets a very
    # available mediocre candidate outscore a better-skilled less-available one.
    # Shift range to [0.5 + 0.5×raw] so profile quality (raw_fit) plays a
    # bigger role while behavioral signals still meaningfully differentiate.
    multiplier = 0.5 + 0.5 * multiplier
    multiplier = min(multiplier, 1.20)  # hard cap
    return round(max(0.0, multiplier), 4)


def get_behavioral_breakdown(candidate: dict[str, Any]) -> dict[str, float]:
    """Return individual behavioral component scores for UI display.

    Args:
        candidate: Full candidate dict.

    Returns:
        Dict with behavioral component scores and metadata.
    """
    signals = candidate.get("redrob_signals", {}) or {}

    last_active = signals.get("last_active_date")
    days_inactive = _days_since(last_active) or 0

    recency = _recency_score(last_active)
    availability = _availability_score(signals.get("open_to_work_flag"))
    responsiveness = _responsiveness_score(signals.get("recruiter_response_rate"))

    # IMPORTANT: use explicit None check for notice_period_days.
    # 0 is falsy but means "immediate" (best case), not default 60 days.
    notice_raw = signals.get("notice_period_days")
    notice_days = int(notice_raw) if notice_raw is not None else 60
    notice = _notice_score(notice_raw)

    github = _github_bonus(signals.get("github_activity_score"))
    interview = _interview_score(signals.get("interview_completion_rate"))
    work_mode = _work_mode_score(signals.get("preferred_work_mode"))
    market = _market_validation_score(signals.get("saved_by_recruiters_30d"))
    completeness = _completeness_score(signals.get("profile_completeness_score"))
    salary = _salary_alignment_score(signals)
    verification = _verification_bonus(signals)
    offer = _offer_acceptance_score(signals.get("offer_acceptance_rate"))
    activity = _activity_score(signals)

    core_mult = (
        recency * availability * responsiveness * notice * github
        * interview * work_mode * salary * offer * activity
    )
    bonus = (market - 1.0) + (completeness - 0.85) * 0.5 + verification
    raw_mult = core_mult + bonus
    # Same compression as compute_behavioral_multiplier
    compressed = 0.5 + 0.5 * raw_mult
    full_mult = round(min(max(0.0, compressed), 1.20), 4)

    return {
        "recency": recency,
        "availability": availability,
        "responsiveness": responsiveness,
        "notice": notice,
        "github": github,
        "interview": interview,
        "work_mode": work_mode,
        "market_validation": market,
        "completeness": completeness,
        "salary_alignment": salary,
        "verification": verification,
        "offer_acceptance": offer,
        "activity": activity,
        "multiplier": full_mult,
        "days_inactive": days_inactive,
        "notice_days": notice_days,
        "open_to_work": bool(signals.get("open_to_work_flag")),
        "response_rate": float(signals.get("recruiter_response_rate") or 0.5),
    }
