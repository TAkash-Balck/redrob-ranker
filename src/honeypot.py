"""
honeypot.py — Detect synthetic / impossible candidate profiles.

The dataset contains ~80 honeypot candidates designed to fool naive
keyword-based rankers.  We detect them using five heuristic checks
based on internal consistency of dates, skill proficiency, and
title-vs-description domain mismatch.

All detected honeypots receive a final score of 0.0.
"""

from __future__ import annotations

import re
from typing import Any

from src.loader import safe_get

# ─── Domain keyword sets ──────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "ml_ai": {
        "machine learning", "artificial intelligence", "deep learning", "neural",
        "nlp", "natural language", "embedding", "vector", "retrieval", "ranking",
        "recommendation", "pytorch", "tensorflow", "scikit", "xgboost", "transformer",
        "bert", "gpt", "llm", "model", "training", "inference", "prediction",
        "classification", "regression", "clustering", "feature", "dataset",
    },
    "software": {
        "software", "engineer", "backend", "frontend", "full stack", "api",
        "microservice", "database", "sql", "rest", "grpc", "service", "application",
        "java", "python", "javascript", "typescript", "golang", "rust", "c++",
        "system design", "architecture", "cloud", "aws", "gcp", "azure",
    },
    "cv_robotics": {
        "computer vision", "image", "video", "object detection", "segmentation",
        "yolo", "opencv", "slam", "robotics", "autonomous", "lidar", "camera",
        "pose estimation", "tracking", "face recognition",
    },
    "data": {
        "data engineer", "etl", "pipeline", "spark", "hadoop", "kafka", "airflow",
        "data warehouse", "dbt", "bigquery", "snowflake", "databricks", "analytics",
        "tableau", "power bi", "looker",
    },
    "non_tech": {
        "marketing", "sales", "human resources", "hr", "recruiter", "finance",
        "accounting", "civil", "mechanical", "electrical", "operations manager",
        "project manager", "business analyst", "content writer", "seo",
        "graphic design", "customer success", "customer support",
    },
}


def _classify_text_domain(text: str) -> str:
    """Return the most likely domain of a text blob based on keyword frequency.

    Args:
        text: Free-form text (title, description, summary).

    Returns:
        Domain label: one of the keys in ``_DOMAIN_KEYWORDS``, or ``"unknown"``.
    """
    if not text:
        return "unknown"
    lower = text.lower()
    scores = {
        domain: sum(1 for kw in keywords if kw in lower)
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }
    best_domain = max(scores, key=lambda d: scores[d])
    return best_domain if scores[best_domain] > 0 else "unknown"


def is_honeypot(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Run all honeypot detection checks against a candidate profile.

    Args:
        candidate: Parsed candidate dict from the JSONL file.

    Returns:
        Tuple of (is_honeypot: bool, reason: str).
        When is_honeypot is True, reason explains which check triggered.
    """
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    years_of_experience = float(profile.get("years_of_experience") or 0)

    # ── Check 1: Timeline impossibility ────────────────────────────────────
    valid_careers = [
        j for j in career_history
        if isinstance(j, dict) and isinstance(j.get("duration_months"), (int, float))
    ]
    total_career_months = sum(float(j["duration_months"]) for j in valid_careers)
    expected_max = years_of_experience * 12 * 1.25 + 12  # +12 grace months
    if total_career_months > max(expected_max, 12) and years_of_experience > 0:
        return True, (
            f"timeline_impossible: career={total_career_months:.0f}m "
            f"vs yoe={years_of_experience:.1f}y"
        )

    # ── Check 2: Expert skill with 0 months duration ────────────────────────
    expert_zero = [
        s for s in skills
        if isinstance(s, dict)
        and str(s.get("proficiency", "")).lower() == "expert"
        and int(s.get("duration_months") or 0) == 0
    ]
    if len(expert_zero) >= 2:
        names = [s.get("name", "?") for s in expert_zero[:3]]
        return True, f"expert_zero_duration: {names}"

    # ── Check 3: Too many expert skills (keyword stuffer) ───────────────────
    all_experts = [
        s for s in skills
        if isinstance(s, dict)
        and str(s.get("proficiency", "")).lower() == "expert"
    ]
    if len(all_experts) > 8:
        return True, f"too_many_experts: {len(all_experts)} expert skills"

    # ── Check 4: Company founding date impossibility ─────────────────────────
    for job in career_history:
        if not isinstance(job, dict):
            continue
        start_date = job.get("start_date") or ""
        duration = float(job.get("duration_months") or 0)
        match = re.match(r"(\d{4})", str(start_date))
        if match:
            start_year = int(match.group(1))
            if start_year < 2010 and duration > 200:
                return True, (
                    f"impossible_tenure: {duration:.0f}m at "
                    f"{job.get('company', '?')} starting {start_year}"
                )

    # ── Check 5: Title vs all descriptions domain mismatch ──────────────────
    current_title = str(profile.get("current_title") or "")
    title_domain = _classify_text_domain(current_title)

    if title_domain != "unknown" and title_domain != "software" and career_history:
        desc_domains = []
        for job in career_history:
            if isinstance(job, dict):
                desc = str(job.get("description") or "")
                if desc:
                    desc_domains.append(_classify_text_domain(desc))

        if desc_domains:
            # If ALL descriptions belong to a different (non-unknown, non-software) domain
            # and none match the title domain → likely a honeypot
            matching = [d for d in desc_domains if d == title_domain or d == "unknown"]
            non_matching_non_unknown = [
                d for d in desc_domains
                if d not in {title_domain, "unknown", "software"}
            ]
            if len(matching) == 0 and len(non_matching_non_unknown) >= max(1, len(desc_domains) - 1):
                dominant = max(set(desc_domains), key=desc_domains.count)
                if dominant != title_domain and dominant != "unknown":
                    return True, (
                        f"domain_mismatch: title='{title_domain}' "
                        f"but descriptions='{dominant}'"
                    )

    return False, ""


def bulk_honeypot_check(candidates: list[dict[str, Any]]) -> dict[str, bool]:
    """Check a list of candidates and return a mapping of candidate_id → is_honeypot.

    Args:
        candidates: List of parsed candidate dicts.

    Returns:
        Dict mapping candidate_id to bool (True = honeypot).
    """
    result = {}
    for c in candidates:
        cid = c.get("candidate_id", "unknown")
        flag, _ = is_honeypot(c)
        result[cid] = flag
    return result
