"""
career_scorer.py — Career trajectory + consulting penalty scoring.

Holistic career analysis:
1. Title progression quality (ML/AI roles vs generic SWE vs non-tech)
2. Consulting firm penalty (reduces score proportionally to consulting ratio)
3. Job-hopping detection (title-chaser pattern)
4. Career direction bonus (pivoting toward ML/AI = positive signal)

The score reflects the QUALITY of career choices and environments,
not just years of experience.
"""

from __future__ import annotations

import re
from typing import Any

# ─── Title classification sets ────────────────────────────────────────────────

_HIGH_VALUE_TITLE_PATTERNS: tuple[str, ...] = (
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "artificial intelligence engineer",
    "nlp engineer",
    "search engineer",
    "applied scientist",
    "research engineer",
    "recommendation systems engineer",
    "ranking engineer",
    "relevance engineer",
    "applied ml",
    "applied ai",
    "information retrieval",
    "data scientist",        # scored as HIGH only if IR/NLP present in desc
)

_HIGH_VALUE_ALWAYS: tuple[str, ...] = (
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "nlp engineer",
    "search engineer",
    "applied scientist",
    "research engineer",
    "recommendation systems engineer",
    "ranking engineer",
    "relevance engineer",
)

_MEDIUM_VALUE_TITLE_PATTERNS: tuple[str, ...] = (
    "software engineer",
    "backend engineer",
    "data engineer",
    "full stack",
    "platform engineer",
    "cloud engineer",
    "devops engineer",
    "sre",
    "site reliability",
    "data scientist",
    "analytics engineer",
    "staff engineer",
    "principal engineer",
    "tech lead",
    "senior engineer",
)

_LOW_VALUE_TITLE_PATTERNS: tuple[str, ...] = (
    "project manager",
    "business analyst",
    "operations",
    "marketing",
    "hr manager",
    "human resources",
    "sales",
    "accountant",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "content writer",
    "seo",
    "graphic designer",
    "recruiter",
    "scrum master",
    "agile coach",
    "delivery manager",
    "program manager",
)

# ─── Consulting firms (must match lowercased company names) ──────────────────

_CONSULTING_FIRMS: frozenset[str] = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "techmahindra", "mindtree",
    "hexaware", "mphasis", "l&t infotech", "larsen & toubro infotech",
    "cyient", "niit technologies", "patni", "mastech", "kpit",
    "persistent systems", "birlasoft", "zensar", "firstsource",
    "sonata software", "sasken", "infosonics",
})

# ─── Career description signals that redeem a consulting firm stint ───────────

_REDEMPTION_SIGNALS: tuple[str, ...] = (
    "vector", "embedding", "ranking", "recommendation", "retrieval",
    "search", "nlp", "machine learning", "deep learning", "production ml",
    "model serving", "deployed", "a/b test", "ndcg",
)


# Minimum IR/ML signals that distinguish ML Data Scientists from analytics DS
_DATA_SCIENTIST_IR_SIGNALS: tuple[str, ...] = (
    "machine learning", "model", "nlp", "embedding", "prediction",
    "classification", "neural", "deep learning", "recommendation",
    "retrieval", "ranking", "vector", "deploy", "training",
)

# ── ML signals for promoting medium-tier titles (SWE/Backend doing real ML) ──
_MEDIUM_TITLE_ML_SIGNALS: tuple[str, ...] = (
    "embedding", "vector", "ranking", "retrieval", "recommendation",
    "nlp", "machine learning", "search quality", "deployed model",
    "a/b test", "ndcg", "semantic", "faiss", "transformer",
    "model training", "model serving", "fine-tun", "llm",
)

# ── Strong production signals (for description quality bonus) ────────────────
_STRONG_PRODUCTION_SIGNALS: tuple[str, ...] = (
    "shipped", "deployed to production", "serving", "production traffic",
    "millions of", "latency", "throughput", "a/b test", "real users",
    "retrieval system", "ranking pipeline", "search quality metric",
)

# ── Domain signals (for description quality bonus) ────────────────────────────
_DESC_DOMAIN_SIGNALS: tuple[str, ...] = (
    "embedding", "vector search", "semantic", "dense retrieval",
    "faiss", "elasticsearch", "recommendation", "information retrieval",
    "reranking", "hybrid search", "ndcg", "mrr",
)


def _is_consulting_company(company_name: str) -> bool:
    """Check if a company name is a known consulting/services firm.

    Args:
        company_name: Raw company name string.

    Returns:
        True if the company is a known consulting firm.
    """
    lower = company_name.lower().strip()
    for firm in _CONSULTING_FIRMS:
        if firm in lower:
            return True
    return False


def _classify_title(title: str) -> str:
    """Classify a job title into 'high', 'medium', or 'low' value.

    Args:
        title: Raw title string.

    Returns:
        One of "high", "medium", "low".
    """
    lower = title.lower()

    # Always-high ML/AI roles
    for pat in _HIGH_VALUE_ALWAYS:
        if pat in lower:
            return "high"

    # Data Scientist is treated as HIGH unconditionally at the title level.
    # The old approach (medium always, high only with IR description) double-penalized
    # genuine ML Data Scientists. Context-based nuance is handled separately
    # in _title_progression_score() via the description check.
    if "data scientist" in lower:
        return "high"

    # Check low before medium to avoid over-generous medium classification
    for pat in _LOW_VALUE_TITLE_PATTERNS:
        if pat in lower:
            return "low"
    for pat in _MEDIUM_VALUE_TITLE_PATTERNS:
        if pat in lower:
            return "medium"
    # Default to medium — gives benefit of the doubt
    return "medium"


def _title_progression_score(career_history: list[dict[str, Any]]) -> float:
    """Score the quality and trajectory of job titles across career history.

    Args:
        career_history: List of job dicts ordered earliest→latest.

    Returns:
        Score in [0.0, 1.0].
    """
    if not career_history:
        return 0.1

    title_scores: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.05}
    weighted_sum = 0.0
    total_months = 0.0

    # Give extra weight to recent jobs using quadratic recency weighting.
    # Linear (1.0→1.5) was too flat for 5+ job careers on the real 100K dataset.
    # Quadratic (1.0→2.0) strongly emphasises the most recent role.
    n = len(career_history)
    for i, job in enumerate(career_history):
        if not isinstance(job, dict):
            continue
        title = str(job.get("title") or "")
        duration = float(job.get("duration_months") or 1)
        tier = _classify_title(title)
        recency_weight = 1.0 + (i / max(n - 1, 1)) ** 2 * 1.0  # 1.0→2.0 quadratic
        desc = str(job.get("description") or "").lower()

        # Analytics DS downgrade: if title is Data Scientist but description has
        # ZERO ML/IR signals, this is a BI/analytics role, not an ML role.
        # Only downgrade to medium; do not further penalize.
        if tier == "high" and "data scientist" in title.lower():
            has_ml = any(sig in desc for sig in _DATA_SCIENTIST_IR_SIGNALS)
            if not has_ml and desc.strip():  # only downgrade if desc is non-empty
                tier = "medium"

        # Medium title ML promotion: SWE/Backend/Data Engineers who describe real
        # ML/IR work in their job description are doing the same work as an
        # ML Engineer — title alone is a poor signal at product companies.
        # Require ≥2 independent signal hits to reduce false positives.
        if tier == "medium":
            ml_hit_count = sum(
                1 for sig in _MEDIUM_TITLE_ML_SIGNALS if sig in desc
            )
            if ml_hit_count >= 2:
                tier = "high"  # SWE/Backend with real ML work = high value

        # Consulting firm ML redemption: medium title at consulting + IR work = high
        # (catches cases not already promoted above, e.g. single-signal descriptions)
        if tier == "medium" and _is_consulting_company(str(job.get("company") or "")):
            if any(sig in desc for sig in _REDEMPTION_SIGNALS):
                tier = "high"  # promote to high if doing real ML work

        weighted_sum += title_scores[tier] * duration * recency_weight
        total_months += duration * recency_weight

    if total_months == 0:
        return 0.1

    return min(weighted_sum / total_months, 1.0)


def _consulting_penalty(career_history: list[dict[str, Any]]) -> float:
    """Compute the consulting firm penalty multiplier.

    Args:
        career_history: List of job dicts.

    Returns:
        Multiplier in [0.4, 1.0].  A 100% consulting career returns 0.4.
        Exception: if consulting descriptions show IR/ML product work,
        penalty is reduced.
    """
    if not career_history:
        return 1.0

    consulting_months = 0.0
    consulting_with_ir = 0.0
    total_months = 0.0

    for job in career_history:
        if not isinstance(job, dict):
            continue
        duration = float(job.get("duration_months") or 1)
        total_months += duration
        company = str(job.get("company") or "")
        if _is_consulting_company(company):
            consulting_months += duration
            # Check if the consulting role had real ML/IR work
            desc = str(job.get("description") or "").lower()
            if any(sig in desc for sig in _REDEMPTION_SIGNALS):
                consulting_with_ir += duration

    if total_months == 0:
        return 1.0

    # Reduce consulting_months by half for those with real ML work
    effective_consulting = consulting_months - (consulting_with_ir * 0.5)
    consulting_ratio = max(0.0, effective_consulting) / total_months
    multiplier = 1.0 - (0.6 * consulting_ratio)
    return max(0.4, multiplier)


def _job_hopper_penalty(career_history: list[dict[str, Any]]) -> float:
    """Detect title-chasers who switch companies frequently for title bumps.

    Args:
        career_history: List of job dicts.

    Returns:
        Multiplier in [0.8, 1.0].  Returns 0.8 if 3+ short stints detected.
    """
    if len(career_history) < 4:
        return 1.0

    short_stints = sum(
        1 for j in career_history
        if isinstance(j, dict) and float(j.get("duration_months") or 24) < 18
    )
    return 0.8 if short_stints >= 3 else 1.0


def _career_direction_bonus(career_history: list[dict[str, Any]]) -> float:
    """Award a bonus for careers pivoting toward ML/AI in recent roles.

    Detects transitions like: Marketing → Data Analyst → ML Engineer.

    Args:
        career_history: List of job dicts ordered earliest→latest.

    Returns:
        Bonus in [0.0, 0.1].
    """
    if len(career_history) < 2:
        return 0.0

    # Look at the last 2 roles
    recent = career_history[-2:] if len(career_history) >= 2 else career_history
    tiers = [_classify_title(str(j.get("title") or "")) for j in recent if isinstance(j, dict)]

    # Upward trajectory: ends in "high" after a non-high role
    if len(tiers) >= 2 and tiers[-1] == "high" and tiers[-2] != "high":
        return 0.10
    if len(tiers) >= 2 and tiers[-1] == "medium" and tiers[-2] == "low":
        return 0.05

    return 0.0


def _career_depth_bonus(career_history: list[dict[str, Any]]) -> float:
    """Reward candidates who have held multiple senior ML/AI roles.

    A candidate with 3+ high-value ML titles across their career has
    proven themselves repeatedly — this is strong signal for a Senior role.

    Args:
        career_history: List of job dicts.

    Returns:
        Bonus in [0.0, 0.08].
    """
    if len(career_history) < 2:
        return 0.0
    high_count = sum(
        1 for j in career_history
        if isinstance(j, dict) and _classify_title(str(j.get("title") or "")) == "high"
    )
    if high_count >= 3:
        return 0.08   # 3+ distinct senior ML roles = excellent breadth
    if high_count >= 2:
        return 0.05   # 2 senior ML roles = good breadth
    return 0.0


def _description_quality_bonus(career_history: list[dict[str, Any]]) -> float:
    """Standalone bonus for production-quality ML work described in career history.

    This signal is independent of title tier: a candidate who describes
    'ranking pipelines serving millions of users in production' deserves
    a bonus regardless of their job title.

    Args:
        career_history: List of job dicts.

    Returns:
        Bonus in [0.0, 0.20].
    """
    if not career_history:
        return 0.0
    all_desc = " ".join(
        str(j.get("description") or "") for j in career_history if isinstance(j, dict)
    ).lower()
    if not all_desc.strip():
        return 0.0
    strong_hits = sum(1 for s in _STRONG_PRODUCTION_SIGNALS if s in all_desc)
    domain_hits = sum(1 for s in _DESC_DOMAIN_SIGNALS if s in all_desc)
    # Strong production signals = 0.05 each, domain signals = 0.03 each
    return min(strong_hits * 0.05 + domain_hits * 0.03, 0.20)


def score_career(candidate: dict[str, Any]) -> float:
    """Compute the holistic career trajectory score for a candidate.

    Combines title progression, consulting penalty, job-hopper detection,
    career direction trajectory, career depth, and description quality
    into a single [0, 1] score.

    Args:
        candidate: Full candidate dict.

    Returns:
        Career trajectory score in [0.0, 1.0].
    """
    career_history = candidate.get("career_history", []) or []

    if not career_history:
        # Fall back to title-only scoring
        profile = candidate.get("profile", {}) or {}
        title = str(profile.get("current_title") or "")
        tier = _classify_title(title)
        return {"high": 0.7, "medium": 0.4, "low": 0.1}.get(tier, 0.3)

    base_score = _title_progression_score(career_history)
    consulting_mult = _consulting_penalty(career_history)
    hopper_mult = _job_hopper_penalty(career_history)
    direction_bonus = _career_direction_bonus(career_history)
    depth_bonus = _career_depth_bonus(career_history)
    desc_bonus = _description_quality_bonus(career_history)

    raw = (base_score * consulting_mult * hopper_mult) + direction_bonus + depth_bonus + desc_bonus
    return max(0.0, min(raw, 1.0))


def get_career_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata useful for reasoning generation.

    Args:
        candidate: Full candidate dict.

    Returns:
        Dict with keys: consulting_ratio, short_stints, title_tier,
        has_redemption_signals, dominant_company_type.
    """
    career_history = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}

    consulting_months = 0.0
    total_months = 0.0
    short_stints = 0
    has_redemption = False

    for job in career_history:
        if not isinstance(job, dict):
            continue
        duration = float(job.get("duration_months") or 0)
        total_months += duration
        company = str(job.get("company") or "")
        if _is_consulting_company(company):
            consulting_months += duration
            desc = str(job.get("description") or "").lower()
            if any(sig in desc for sig in _REDEMPTION_SIGNALS):
                has_redemption = True
        if duration < 18 and duration > 0:
            short_stints += 1

    consulting_ratio = consulting_months / total_months if total_months else 0.0
    title = str(profile.get("current_title") or "")
    title_tier = _classify_title(title)

    return {
        "consulting_ratio": consulting_ratio,
        "consulting_pct": round(consulting_ratio * 100),
        "short_stints": short_stints,
        "title_tier": title_tier,
        "has_redemption_signals": has_redemption,
        "dominant_company_type": "consulting" if consulting_ratio > 0.5 else "product",
    }
