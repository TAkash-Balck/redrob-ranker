"""
filters.py — Hard domain filters to exclude obviously irrelevant candidates.

These filters are applied BEFORE scoring to skip candidates whose profile
makes it clear they are not in-scope for a Senior AI/ML Engineer role.
Filtering ~85% of the 100K dataset early is critical for staying under
the 5-minute time budget.

Note: Filters are intentionally conservative — they only reject candidates
when there are multiple strong non-tech signals.  A civil engineer who has
since pivoted to ML should NOT be filtered.
"""

from __future__ import annotations

from typing import Any


# ─── Title-domain classification (shared with honeypot.py but lightweight) ───

_HARD_NEGATIVE_TITLE_WORDS = {
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "architect",           # building architect — not software
    "interior designer",
    "chartered accountant",
    "ca ",
    "cfa ",
    "hr manager",
    "human resources manager",
    "content writer",
    "seo specialist",
    "graphic designer",
    "motion designer",
    "ux designer",         # keep UX in — can be adjacent to tech
    "sales manager",
    "sales executive",
    "business development",
    "marketing manager",
    "brand manager",
    "operations manager",
    "supply chain",
    "procurement",
    "logistics",
    "telecaller",
    "bpo",
    "customer care",
    "receptionist",
    "medical officer",
    "doctor",
    "pharmacist",
    "nurse",
    "teacher",
    "lecturer",
    "chef",
    "cook",
}

_TECH_REDEMPTION_WORDS = {
    "machine learning",
    "artificial intelligence",
    "data scientist",
    "nlp",
    "search",
    "recommendation",
    "ml",
    "ai",
    "deep learning",
    "data engineer",
    "software engineer",
    "python",
    "analytics",
    "bi",
    "data analyst",
}

# Career titles that are hard non-tech — ALL career history having these
# titles means the profile is almost certainly not an ML/AI engineer,
# regardless of what the headline/summary says (keyword stuffing).
_HARD_CAREER_TITLE_WORDS = {
    "hr manager",
    "human resources",
    "operations manager",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "chartered accountant",
    "accounts manager",
    "sales manager",
    "sales executive",
    "marketing manager",
    "content writer",
    "graphic designer",
    "supply chain",
    "procurement",
    "logistics manager",
    "medical officer",
    "teacher",
    "lecturer",
}


def _career_title_is_nonttech(job_title: str) -> bool:
    """Return True if a single career job title is a hard non-tech role."""
    t = job_title.lower().strip()
    return any(neg in t for neg in _HARD_CAREER_TITLE_WORDS)


def is_hard_filtered(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Return True if the candidate should be filtered out before scoring.

    Applies three checks:
    1. Hard-negative current title (non-tech profession) with no tech redemption
       in headline, summary, OR career history titles.
    2. Entire career history is all non-tech roles (not just current title).
    3. Zero years of experience + no tech signal at all.

    We never filter purely on company — a TCS employee who pivoted to ML
    is still worth scoring (the consulting penalty handles them later).

    Args:
        candidate: Parsed candidate dict.

    Returns:
        Tuple of (should_filter: bool, reason: str).
    """
    profile = candidate.get("profile", {}) or {}
    current_title = str(profile.get("current_title") or "").lower()
    headline = str(profile.get("headline") or "").lower()
    summary = str(profile.get("summary") or "").lower()
    years_exp = float(profile.get("years_of_experience") or 0)
    career_history = candidate.get("career_history", []) or []

    # Build a combined redemption text that includes career history titles.
    # This prevents keyword-stuffed headlines/summaries from saving non-tech profiles.
    career_titles_text = " ".join(
        str(j.get("title", "")).lower()
        for j in career_history
        if isinstance(j, dict)
    )
    redemption_text = headline + " " + summary + " " + career_titles_text

    # ── Check 1: Hard-negative title with no tech redemption ──────────────
    for neg_word in _HARD_NEGATIVE_TITLE_WORDS:
        if neg_word in current_title:
            has_tech = any(tw in redemption_text for tw in _TECH_REDEMPTION_WORDS)
            if not has_tech:
                return True, f"non_tech_title: '{current_title}'"

    # ── Check 2: Entire career history is non-tech roles ──────────────────
    # If every single job in their career was a hard non-tech role, filter out.
    # This catches candidates whose career reality contradicts their current title.
    if career_history:
        non_tech_count = sum(
            1 for j in career_history
            if isinstance(j, dict) and _career_title_is_nonttech(str(j.get("title", "")))
        )
        if non_tech_count == len(career_history):
            # All roles are non-tech — even if current title looks OK, this is a mismatch
            return True, "all_career_titles_non_tech"

    # ── Check 3: No experience and no tech signal at all ─────────────────
    if years_exp == 0:
        all_text = current_title + " " + headline + " " + summary
        has_any_tech = any(tw in all_text for tw in _TECH_REDEMPTION_WORDS)
        if not has_any_tech:
            return True, "zero_exp_no_tech_signal"

    return False, ""


def filter_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into (to_score, filtered_out) lists.

    Args:
        candidates: Full list of candidate dicts.

    Returns:
        Tuple of (to_score, filtered_out).  filtered_out candidates
        will NOT appear in the final submission (they are ranked last
        with score=0 if we need padding, but for 100K datasets with
        ~85% off-role candidates, we never need to pad).
    """
    to_score = []
    filtered_out = []
    for c in candidates:
        flag, _ = is_hard_filtered(c)
        if flag:
            filtered_out.append(c)
        else:
            to_score.append(c)
    return to_score, filtered_out

