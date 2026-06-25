"""
reasoner.py — Per-candidate reasoning generator.

Produces rank-consistent, specific, non-hallucinated 1–2 sentence
reasoning strings for each candidate in the final Top 100.

Rules:
- Only reference actual values from the candidate's profile
- Be honest about concerns (consulting, inactivity, location)
- Top-ranked candidates get positive framing
- Mid-ranked get balanced framing (one + and one -)
- Low-ranked (51–100) acknowledge the marginal fit explicitly
"""

from __future__ import annotations

from typing import Any

from src.behavioral import get_behavioral_breakdown, _days_since
from src.career_scorer import get_career_metadata, _classify_title
from src.filters import _HARD_NEGATIVE_TITLE_WORDS
from src.location_scorer import get_location_metadata
from src.product_scorer import get_top_product_companies
from src.skill_scorer import get_top_matched_skills, _classify_skill_tier


def _add_pivot_note(reasoning: str, candidate: dict) -> str:
    """Append a pivot-back risk clause when profile.current_title is non-tech.

    The ranker applies a 35% score penalty to candidates whose current role
    is a hard-negative title (e.g. Civil Engineer) even though their career
    history has legitimate ML/AI experience.  Without this note the score
    penalty is unexplained, making the ranking look like an anomaly.

    Only reads profile.current_title (the stale field) -- deliberately so,
    because the penalty is triggered by the CURRENT job, not the best career
    title used in the rest of the reasoning.
    """
    profile = candidate.get("profile", {}) or {}
    current_title_raw = str(profile.get("current_title") or "").strip()
    current_title_lower = current_title_raw.lower()
    is_pivot_back = any(
        neg in current_title_lower for neg in _HARD_NEGATIVE_TITLE_WORDS
    )
    if not is_pivot_back:
        return reasoning
    base = reasoning.rstrip(".")
    return (
        f"{base}; currently working as {current_title_raw} "
        f"(pivot-back risk factored into score)."
    )


def _get_best_title(candidate: dict) -> str:
    """Return the best title for reasoning from career history.

    Always prefers career_history over profile.current_title because
    profile.current_title is a stale denormalized field — mismatches in
    179/200 sample candidates confirm career_history is the source of truth.

    Priority:
    1. Most recent HIGH-tier career title
    2. Most recent MEDIUM-tier career title
    3. career_history[0].title (most recent job, any tier)
    4. profile.current_title (last resort only)
    """
    career = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}
    fallback = str(profile.get("current_title", "Engineer")).strip()

    if not career:
        return fallback

    # Pass 1: best HIGH-tier title (most recent first)
    for job in reversed(career):
        title = str(job.get("title", ""))
        if _classify_title(title) == "high":
            return title

    # Pass 2: best MEDIUM-tier title
    for job in reversed(career):
        title = str(job.get("title", ""))
        if _classify_title(title) == "medium":
            return title

    # Pass 3: most recent career job title regardless of tier
    latest = str(career[0].get("title", "")).strip()
    return latest if latest else fallback


def _format_notice(notice_days: int) -> str:
    """Human-readable notice period string."""
    if notice_days == 0:
        return "immediate availability"
    elif notice_days <= 30:
        return f"{notice_days}d notice"
    elif notice_days <= 90:
        return f"{notice_days}d notice"
    else:
        return f"long notice ({notice_days}d)"


def _primary_gap(candidate: dict, scores: dict) -> str:
    """Return the most accurate gap phrase for mid/low-rank candidates.

    Priority: experience band mismatch > skill gaps > behavioral gaps > fallback.
    Never use a junior-framing phrase for an overqualified candidate or vice versa.
    """
    profile = candidate.get("profile", {}) or {}
    sig = candidate.get("redrob_signals", {}) or {}
    yoe = float(profile.get("years_of_experience") or 0)
    exp_band = float(scores.get("experience_band", 1.0))
    skill_depth = float(scores.get("skill_depth", 1.0))

    # -- Experience band gaps (most specific, checked first) ---------------
    if yoe > 13.0:
        return "significantly above the 5-9y target range"
    if exp_band < 0.45 and yoe > 10.0:
        return "above the 5-9y experience target for this role"
    if exp_band < 0.60 and yoe < 4.0:
        return f"under-experienced ({yoe:.0f}y vs 5-9y target)"
    if exp_band < 0.80 and yoe > 9.0:
        return "slightly senior for this role's experience target"
    if exp_band < 0.60 and yoe < 5.0:
        return "below the 5y minimum experience threshold"

    # -- Skill gaps --------------------------------------------------------
    if skill_depth < 0.20:
        return "limited documented AI/ML skills"
    if skill_depth < 0.35:
        return "sparse AI/ML skill profile"

    # -- Behavioral gaps ---------------------------------------------------
    from datetime import date, datetime
    today = date.today()
    last_active = sig.get("last_active_date")
    if last_active:
        try:
            days = (today - datetime.strptime(
                str(last_active), "%Y-%m-%d").date()).days
            if days > 150:
                return f"inactive for {days} days"
            if days > 90:
                return f"low recent activity ({days}d since last login)"
        except Exception:
            pass

    notice = int(sig.get("notice_period_days") or 0)
    if notice > 90:
        return f"long notice period ({notice}d)"

    response = float(sig.get("recruiter_response_rate") or 0)
    if response < 0.30:
        return "low recruiter response rate"

    # -- Fallback ----------------------------------------------------------
    return "limited production ML/AI experience"



def generate_reasoning(
    candidate: dict[str, Any],
    rank: int,
    scores: dict[str, float],
    final_score: float,
) -> str:
    """Generate a 1–2 sentence reasoning string for a candidate.

    Args:
        candidate: Full candidate dict.
        rank: Final rank (1-indexed).
        scores: Dict with keys: skill_depth, career_trajectory, product_company,
                experience_band, location, behavioral_multiplier.
        final_score: Final normalized score.

    Returns:
        Reasoning string (1–2 sentences, ≤280 characters).
    """
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}

    years_exp = float(profile.get("years_of_experience") or 0)
    current_title = _get_best_title(candidate)   # use best career title, not stale current_title
    headline = str(profile.get("headline") or "")

    # Gather sub-components
    top_skills = get_top_matched_skills(candidate, n=3)
    top_companies = get_top_product_companies(candidate, n=2)
    behavioral = get_behavioral_breakdown(candidate)
    career_meta = get_career_metadata(candidate)
    location_meta = get_location_metadata(candidate)

    skill1 = top_skills[0] if top_skills else "ML/AI skills"
    skill2 = top_skills[1] if len(top_skills) > 1 else None
    company_str = " & ".join(top_companies) if top_companies else "product companies"
    notice_str = _format_notice(behavioral["notice_days"])
    location_str = location_meta["location"]
    response_rate = behavioral["response_rate"]

    # ── Rank 1–15: Top tier (3 rotating templates for variety) ────────────
    if rank <= 15:
        skill_clause = f"{skill1} and {skill2}" if skill2 else skill1
        template_idx = (rank - 1) % 3

        # Shared overqualified flag used by all three templates
        exp_band_score = float(scores.get("experience_band", 1.0))
        is_overqualified = exp_band_score < 0.80 and years_exp >= 5.0

        if template_idx == 0:
            if is_overqualified:
                # Experienced framing — no target-band claim
                if top_companies:
                    s1 = (f"Experienced {current_title} ({years_exp:.0f}y) "
                          f"from {company_str} with {skill_clause} background")
                else:
                    s1 = (f"{years_exp:.0f}y {current_title} "
                          f"with {skill_clause} background")
            else:
                if years_exp >= 7:
                    label = "applied ML/AI engineer"
                elif years_exp >= 5:
                    label = "ML/AI engineer"
                else:
                    label = str(current_title)
                if top_companies:
                    s1 = (f"{years_exp:.0f}y {label} from {company_str}; "
                          f"demonstrated {skill_clause} in production")
                else:
                    s1 = (f"{years_exp:.0f}y {label} with hands-on "
                          f"{skill_clause} experience")

            # Sparse-skill qualifier — recruiter signal when documented skills are thin
            work_mode_note = (
                "; prefers remote (JD is hybrid)"
                if str(signals.get("preferred_work_mode") or "").lower() == "remote"
                else ""
            )
            if scores.get("skill_depth", 1.0) < 0.35:
                s2 = (f"{location_str}, {notice_str}, "
                      f"{response_rate:.0%} recruiter response rate; "
                      f"limited documented AI/ML skills{work_mode_note}.")
            else:
                s2 = (f"{location_str}, {notice_str}, "
                      f"{response_rate:.0%} recruiter response rate{work_mode_note}.")
            return _add_pivot_note(f"{s1}; {s2}", candidate)

        elif template_idx == 1:
            # Template B: lead with skills + availability
            # Avoid "Strong fit" framing for candidates outside the 5-9y target
            # band — the score already penalises them via exp_band, so the
            # reasoning must not contradict that signal.
            open_flag = behavioral.get("open_to_work", False)
            avail_str = "actively open to work" if open_flag else "open to opportunities"

            if is_overqualified:
                # Neutral framing for 10y+ candidates
                if top_companies:
                    s1 = (f"Experienced {current_title} ({years_exp:.0f}y) "
                          f"from {company_str} with {skill_clause} background")
                else:
                    s1 = (f"{years_exp:.0f}y {current_title} "
                          f"with {skill_clause} background")
            else:
                if top_companies:
                    s1 = (f"Strong fit -- {years_exp:.0f}y at {company_str} "
                          f"with hands-on {skill_clause}")
                else:
                    s1 = (f"Strong fit -- {years_exp:.0f}y {current_title} "
                          f"with {skill_clause} background")

            work_mode_note = (
                "; prefers remote (JD is hybrid)"
                if str(signals.get("preferred_work_mode") or "").lower() == "remote"
                else ""
            )
            s2 = (f"{avail_str} from {location_str}; "
                  f"{notice_str}, {response_rate:.0%} response rate{work_mode_note}.")
            return _add_pivot_note(f"{s1}; {s2}", candidate)

        else:
            # Template C: lead with skill depth + company context
            # Three cases: junior (<5y), overqualified (exp_band<0.80), senior (target band)
            days_inactive = behavioral.get("days_inactive", 0)
            activity_str = (
                "recently active" if days_inactive < 14
                else f"active {days_inactive}d ago"
            )
            is_junior = years_exp < 5.0
            is_overqualified_c = is_overqualified and not is_junior

            if is_junior:
                # Junior-safe framing — no production claim
                if top_companies:
                    s1 = (f"{years_exp:.0f}y {current_title} with "
                          f"{skill1} experience at {company_str}")
                else:
                    s1 = (f"{years_exp:.0f}y {current_title} with "
                          f"{skill1} background")
            elif is_overqualified_c:
                # Overqualified framing — acknowledge seniority without false endorsement
                if top_companies:
                    s1 = (f"Experienced {current_title} ({years_exp:.0f}y) "
                          f"from {company_str} with {skill1} background")
                else:
                    s1 = (f"{years_exp:.0f}y {current_title} "
                          f"with {skill1} background")
            else:
                # Target-band senior framing — production is appropriate
                if top_companies:
                    s1 = (f"Production {skill1} background across "
                          f"{company_str} ({years_exp:.0f}y total)")
                else:
                    s1 = (f"{years_exp:.0f}y of {skill1} work "
                          f"as {current_title}")

            work_mode_note = (
                "; prefers remote (JD is hybrid)"
                if str(signals.get("preferred_work_mode") or "").lower() == "remote"
                else ""
            )
            s2 = (f"Based in {location_str}, {notice_str}; "
                  f"{activity_str}, {response_rate:.0%} response rate{work_mode_note}.")
            return _add_pivot_note(f"{s1}; {s2}", candidate)


    # ── Rank 16–50: Solid fits ─────────────────────────────────────────────
    if rank <= 50:
        concern = _primary_gap(candidate, scores)

        # Vary the "slightly senior" phrase so 18 consecutive ranks don't
        # all read identically — same factual meaning, 3 different wordings.
        if "slightly senior" in concern:
            senior_variants = [
                "slightly senior for this role's experience target",
                f"above the 5-9y sweet spot at {years_exp:.0f}y",
                "experience level exceeds the target band for this role",
            ]
            concern = senior_variants[rank % 3]

        positive = ""
        if top_companies:
            positive = f"product company background ({company_str})"
        elif scores.get("skill_depth", 0) > 0.4:
            positive = f"solid {skill1} skills"
        elif scores.get("experience_band", 0) >= 0.9:
            positive = "good experience band (5-9y target)"
        else:
            positive = "relevant ML/AI exposure"

        sentence1 = (
            f"{current_title} with {years_exp:.0f}y exp and {skill1} background"
        )
        sentence2 = f"{concern} limits ranking, but {positive} is a positive."
        return _add_pivot_note(f"{sentence1}; {sentence2}", candidate)

    # ── Rank 51–100: Marginal fits ─────────────────────────────────────────
    gap = _primary_gap(candidate, scores)

    # 3 rotating templates so 50 consecutive rows don't share identical structure.
    low_templates = [
        # Template 0 — original
        (f"Adjacent profile — {current_title} with some {skill1} exposure; "
         f"Ranked {rank} due to {gap}; included based on partial skill overlap."),
        # Template 1 — concern-forward
        (f"{current_title} ({years_exp:.0f}y) included for marginal {skill1} relevance; "
         f"primary concern: {gap}."),
        # Template 2 — borderline framing
        (f"Borderline inclusion — {current_title} with {skill1} background; "
         f"{gap} is the key limiting factor."),
    ]
    return _add_pivot_note(low_templates[rank % 3], candidate)


def generate_bulk_reasoning(
    ranked_candidates: list[tuple[int, dict[str, Any], dict[str, float], float]]
) -> list[str]:
    """Generate reasoning for all ranked candidates.

    Args:
        ranked_candidates: List of (rank, candidate, scores, final_score) tuples.

    Returns:
        List of reasoning strings in rank order.
    """
    return [
        generate_reasoning(candidate, rank, scores, score)
        for rank, candidate, scores, score in ranked_candidates
    ]
