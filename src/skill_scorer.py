"""
skill_scorer.py — Weighted skill scoring with tier system.

Scoring approach:
- Skills are weighted by tier (Tier1=2.0 → Tier4=0.5, Negative=-1.0)
- Within each skill, weight = proficiency × (duration + endorsements) blend
- Career description text is scanned for production IR/ML signal keywords
- Final score is normalized to [0, 1]
"""

from __future__ import annotations

import re
from typing import Any

# ─── Proficiency weights ──────────────────────────────────────────────────────

_PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner":     0.3,
    "intermediate": 0.7,
    "advanced":     1.0,
    "expert":       1.2,
}

# ─── Tier skill sets (lowercase for fast membership testing) ─────────────────
# These are loaded from config at module level but also defined here as fallback.

_TIER1_SKILLS: frozenset[str] = frozenset({
    "embeddings", "embedding", "sentence-transformers", "sentence_transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "vector database", "vector search", "vector store",
    "semantic search", "hybrid search", "bm25", "information retrieval",
    "dense retrieval", "rag", "retrieval augmented generation",
    "ndcg", "mrr", "learning-to-rank", "learning to rank", "ltr",
    "reranking", "re-ranking", "ranking systems", "ann",
    "approximate nearest neighbor", "hnsw", "e5", "bge",
    "bi-encoder", "cross-encoder",
})

_TIER2_SKILLS: frozenset[str] = frozenset({
    "nlp", "natural language processing", "llm", "large language model",
    "fine-tuning", "fine_tuning", "lora", "qlora", "peft",
    "transformers", "hugging face", "huggingface",
    "recommendation systems", "recommender systems",
    "ranking", "a/b testing", "ab testing", "mlflow",
    "evaluation frameworks", "text classification",
    "named entity recognition", "ner", "question answering",
    "haystack", "langchain", "llamaindex", "llama index",
    "spacy", "gensim", "bert", "roberta", "gpt", "t5",
    "search quality", "relevance engineering",
})

_TIER3_SKILLS: frozenset[str] = frozenset({
    "python", "machine learning", "deep learning",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "feature engineering",
    "model deployment", "mlops", "model serving", "fastapi",
    "flask", "triton", "onnx", "neural networks",
    "data science", "statistics", "probability",
})

_TIER4_SKILLS: frozenset[str] = frozenset({
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s",
    "sql", "nosql", "spark", "hadoop", "kafka", "airflow",
    "distributed systems", "rest api", "microservices", "redis",
    "mongodb", "postgresql", "linux", "git", "ci/cd",
})

_NEGATIVE_SKILLS: frozenset[str] = frozenset({
    "computer vision", "cv", "object detection", "image segmentation",
    "image classification", "yolo", "opencv", "robotics",
    "speech recognition", "automatic speech recognition", "asr",
})

_TIER_WEIGHTS: dict[str, float] = {
    "tier1":    2.0,
    "tier2":    1.5,
    "tier3":    1.0,
    "tier4":    0.5,
    "negative": -1.0,
}

# Normalization ceiling.
# At 8.0: a 3-skill elite profile (FAISS expert + 2 advanced Tier1 = raw ~6.1) scores ~0.77.
# A 6-skill elite profile scores ~1.0 (capped). This correctly positions the realistic
# ceiling for a 7-8y Senior AI Engineer as reachable, not theoretical.
_NORMALIZATION_DIVISOR: float = 8.0

# ─── Career description signal keywords ──────────────────────────────────────

_CAREER_SIGNALS: tuple[str, ...] = (
    "shipped ranking", "production retrieval", "vector search", "embedding",
    "recommendation system", "search quality", "relevance", "ranking model",
    "a/b test", "online evaluation", "ndcg", "product company", "real users",
    "latency", "throughput", "indexed", "personalization", "query understanding",
    "candidate generation", "two-stage", "re-rank", "retrieval system",
    "search pipeline", "served", "deployed to production", "production traffic",
    "millions of",
)

# NLP-positive terms found in descriptions (broader catch)
_NLP_IR_TERMS: tuple[str, ...] = (
    "information retrieval", "text ranking", "document ranking",
    "sentence embedding", "semantic similarity", "passage retrieval",
    "dense passage", "sparse retrieval", "bm25", "faiss", "pinecone",
    "qdrant", "weaviate", "vector store", "ann index",
)

# ─── Certification value set ──────────────────────────────────────────────────

_VALUABLE_CERTS: frozenset[str] = frozenset({
    "aws certified machine learning",
    "gcp professional machine learning",
    "google cloud professional data engineer",
    "tensorflow developer",
    "pytorch",
    "hugging face",
    "deep learning specialization",
    "nlp specialization",
    "natural language processing specialization",
    "information retrieval",
    "elastic certified engineer",
    "databricks certified",
    "mlops",
})

# ─── Profile summary power phrases ───────────────────────────────────────────

_SUMMARY_POWER_PHRASES: tuple[str, ...] = (
    "production", "scale", "million", "shipped", "led", "built",
    "vector search", "ranking system", "retrieval", "recommendation",
    "embedding", "deployed", "search quality", "relevance",
    "real users", "a/b test", "latency", "throughput",
    "served", "inference", "model serving",
)


def _classify_skill_tier(skill_name: str) -> str:
    """Return the tier for a skill name string.

    Args:
        skill_name: Raw skill name (case-insensitive match).

    Returns:
        One of "tier1", "tier2", "tier3", "tier4", "negative", or "unknown".
    """
    lower = skill_name.lower().strip()
    # Exact match first
    if lower in _TIER1_SKILLS:
        return "tier1"
    if lower in _TIER2_SKILLS:
        return "tier2"
    if lower in _TIER3_SKILLS:
        return "tier3"
    if lower in _TIER4_SKILLS:
        return "tier4"
    if lower in _NEGATIVE_SKILLS:
        return "negative"

    # Substring match for multi-word skills
    for kw in _TIER1_SKILLS:
        if kw in lower or lower in kw:
            return "tier1"
    for kw in _TIER2_SKILLS:
        if kw in lower or lower in kw:
            return "tier2"
    for kw in _NEGATIVE_SKILLS:
        if kw in lower:
            return "negative"
    for kw in _TIER3_SKILLS:
        if kw in lower or lower in kw:
            return "tier3"
    for kw in _TIER4_SKILLS:
        if kw in lower or lower in kw:
            return "tier4"

    return "unknown"


def _score_single_skill(skill: dict[str, Any]) -> float:
    """Compute weighted score for a single skill entry.

    Args:
        skill: Skill dict with keys: name, proficiency, endorsements, duration_months.

    Returns:
        Weighted float score (may be negative for negative-tier skills).
    """
    name = str(skill.get("name") or "")
    proficiency = str(skill.get("proficiency") or "intermediate").lower()
    endorsements = int(skill.get("endorsements") or 0)
    duration_months = int(skill.get("duration_months") or 0)

    tier = _classify_skill_tier(name)
    if tier == "unknown":
        return 0.0

    tier_weight = _TIER_WEIGHTS[tier]
    prof_weight = _PROFICIENCY_WEIGHTS.get(proficiency, 0.5)
    duration_weight = min(duration_months / 24.0, 1.0)
    endorsement_weight = min(endorsements / 20.0, 1.0)

    # Credibility check: a skill claimed as advanced/expert with zero
    # endorsements AND zero usage duration is almost certainly padding.
    # Apply a discount to prevent keyword-stuffers from gaming skill scores.
    if endorsements == 0 and duration_months == 0 and proficiency in ("advanced", "expert"):
        credibility = 0.50   # strong claim with zero evidence
    elif endorsements == 0 and duration_months == 0:
        credibility = 0.75   # intermediate/beginner claim, no evidence
    else:
        credibility = 1.00   # has either duration or endorsements — credible

    skill_weight = prof_weight * (0.6 + 0.2 * duration_weight + 0.2 * endorsement_weight) * credibility
    return tier_weight * skill_weight


def _score_career_descriptions(
    career_history: list[dict[str, Any]],
    skills: list[dict[str, Any]] | None = None,
) -> float:
    """Scan career job descriptions for production IR/ML signal keywords.

    Applies a sparsity discount when the candidate's skill list is thin.
    A sparse profile (1-2 skills) with a keyword-rich description is more
    likely to be keyword-stuffing than genuine experience.

    Args:
        career_history: List of job dicts, each with a 'description' field.
        skills: Candidate's skills list (used for sparsity check).

    Returns:
        Bonus score in [0, 2.0] — added to raw skill score before normalization.
    """
    if not career_history:
        return 0.0

    all_desc = " ".join(
        str(j.get("description") or "").lower()
        for j in career_history
        if isinstance(j, dict)
    )

    if not all_desc.strip():
        return 0.0

    hits = sum(1 for kw in _CAREER_SIGNALS if kw in all_desc)
    nlp_hits = sum(1 for kw in _NLP_IR_TERMS if kw in all_desc)

    # Each signal hit = 0.15 bonus, cap at 2.0
    bonus = min((hits * 0.15) + (nlp_hits * 0.2), 2.0)

    # Sparsity discount: thin skill lists can't be redeemed by description alone.
    # A candidate with 1 skill (e.g. SQL) and a glowing description is suspicious.
    if skills is not None:
        num_skills = sum(1 for s in skills if isinstance(s, dict))
        if num_skills <= 2:
            bonus *= 0.4   # Severe discount for very sparse profiles
        elif num_skills <= 4:
            bonus *= 0.7   # Moderate discount for sparse profiles

    return bonus


def _cv_to_nlp_penalty(skills: list[dict[str, Any]]) -> float:
    """Apply penalty when CV/robotics skills strongly dominate NLP/IR skills.

    Args:
        skills: List of skill dicts.

    Returns:
        Penalty value in [0, 1.5] (subtracted from raw score).
    """
    cv_score = 0.0
    nlp_score = 0.0

    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = str(skill.get("name") or "").lower()
        prof_w = _PROFICIENCY_WEIGHTS.get(
            str(skill.get("proficiency") or "intermediate").lower(), 0.5
        )
        if any(kw in name for kw in _NEGATIVE_SKILLS):
            cv_score += prof_w
        if any(kw in name for kw in list(_TIER1_SKILLS)[:10]):  # IR/NLP core
            nlp_score += prof_w

    if cv_score > nlp_score + 1.0:
        # Pure CV background with no IR work
        return min(cv_score - nlp_score, 1.5)
    return 0.0


def score_certifications(candidate: dict[str, Any]) -> float:
    """Bonus score for holding valuable ML/IR certifications.

    AWS ML Specialty, GCP ML, TF Developer, Hugging Face, etc.
    Recent certs (2023+) get full value; older ones get 70%.

    Args:
        candidate: Full candidate dict.

    Returns:
        Bonus in [0.0, 0.10].
    """
    certs = candidate.get("certifications", []) or []
    bonus = 0.0
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        name = str(cert.get("name") or cert.get("title") or "").lower()
        if any(v in name for v in _VALUABLE_CERTS):
            year = cert.get("year") or cert.get("issued_year") or 2020
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = 2020
            recency = 1.0 if year >= 2023 else 0.7
            bonus += 0.03 * recency
    return min(bonus, 0.10)


def _score_profile_summary(candidate: dict[str, Any], skills: list[dict[str, Any]]) -> float:
    """Bonus from scanning profile.summary for production ML signal phrases.

    Phrases like 'shipped ranking systems serving 50M users' are stronger
    evidence than skills alone. Applies same sparsity discount as description bonus.

    Args:
        candidate: Full candidate dict.
        skills: Candidate's skills list (for sparsity discount).

    Returns:
        Bonus in [0.0, 0.15].
    """
    profile = candidate.get("profile", {}) or {}
    summary = str(profile.get("summary") or "").lower()
    if not summary:
        return 0.0

    hits = sum(1 for p in _SUMMARY_POWER_PHRASES if p in summary)
    bonus = min(hits * 0.03, 0.15)

    # Same sparsity discount as career description bonus
    num_skills = sum(1 for s in skills if isinstance(s, dict))
    if num_skills <= 2:
        bonus *= 0.4
    elif num_skills <= 4:
        bonus *= 0.7

    return bonus


def score_skills(candidate: dict[str, Any]) -> float:
    """Compute the overall skill depth score for a candidate.

    Combines:
    - Weighted skill tier scores
    - Career description keyword bonuses (sparsity-discounted)
    - Profile summary power phrase bonus (sparsity-discounted)
    - Certification bonus
    - CV-vs-NLP penalty

    Args:
        candidate: Full candidate dict.

    Returns:
        Normalized score in [0.0, 1.0].
    """
    skills = candidate.get("skills", []) or []
    career_history = candidate.get("career_history", []) or []

    profile = candidate.get("profile", {}) or {}
    yoe_months = float(profile.get("years_of_experience") or 0) * 12

    # Sanity-clamp skills whose claimed duration exceeds the candidate's total career.
    # Over-claimed durations (e.g. "FAISS expert 120m" on a 5y engineer) are suspicious.
    # Downgrade only proficiency to 'intermediate' — duration itself may be stale data.
    def _sanitize_skill(s: dict) -> dict:
        if not isinstance(s, dict):
            return s
        dur = float(s.get("duration_months") or 0)
        if yoe_months > 0 and dur > yoe_months * 1.1:
            s = dict(s)  # copy — do not mutate original
            s["proficiency"] = "intermediate"
        return s

    sanitized_skills = [_sanitize_skill(s) for s in skills if isinstance(s, dict)]

    # Deduplicate: keep the highest-proficiency entry per skill name.
    # When proficiency ties, keep the one with more months of usage.
    # Some profiles on the full 100K dataset list the same skill twice
    # (e.g. Python:beginner + Python:expert). Without deduplication, both
    # get scored and the skill total is artificially inflated.
    _PROF_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    _seen: dict[str, dict] = {}
    for s in sanitized_skills:
        name_key = str(s.get("name") or "").lower().strip()
        if not name_key:
            continue
        existing = _seen.get(name_key)
        if existing is None:
            _seen[name_key] = s
        else:
            existing_rank = _PROF_RANK.get(
                str(existing.get("proficiency") or "").lower(), 0
            )
            new_rank = _PROF_RANK.get(
                str(s.get("proficiency") or "").lower(), 0
            )
            if new_rank > existing_rank:
                _seen[name_key] = s
            elif new_rank == existing_rank:
                # Tie-break by duration: prefer longer usage
                existing_dur = int(existing.get("duration_months") or 0)
                new_dur = int(s.get("duration_months") or 0)
                if new_dur > existing_dur:
                    _seen[name_key] = s
    sanitized_skills = list(_seen.values())

    raw_skill_score = sum(
        _score_single_skill(s)
        for s in sanitized_skills
    )
    description_bonus = _score_career_descriptions(career_history, skills=sanitized_skills)
    summary_bonus = _score_profile_summary(candidate, sanitized_skills)
    cert_bonus = score_certifications(candidate)
    cv_penalty = _cv_to_nlp_penalty(sanitized_skills)

    # cert_bonus and summary_bonus are additive post-normalization bonuses
    # (kept separate so they don't interact with the normalization divisor)
    total = raw_skill_score + description_bonus - cv_penalty
    normalized = max(0.0, min(total / _NORMALIZATION_DIVISOR, 1.0))
    # Apply post-norm bonuses, capped at 1.0
    normalized = min(normalized + summary_bonus + cert_bonus, 1.0)
    return normalized


def get_top_matched_skills(
    candidate: dict[str, Any],
    n: int = 3,
) -> list[str]:
    """Return the top N highest-tier skills by score for reasoning generation.

    Prioritizes Tier1 > Tier2 > Tier3 > Tier4 skills so that reasoning
    mentions meaningful IR/NLP/ML skills rather than generic ones like SQL.
    Only falls back to Tier4 if nothing better exists.

    Args:
        candidate: Full candidate dict.
        n: Number of top skills to return.

    Returns:
        List of skill name strings (highest-tier, highest-scoring first).
    """
    skills = candidate.get("skills", []) or []

    _TIER_PRIORITY: dict[str, int] = {
        "tier1": 4,
        "tier2": 3,
        "tier3": 2,
        "tier4": 1,
        "unknown": 0,
        "negative": -1,
    }

    scored = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        sc = _score_single_skill(s)
        if sc <= 0:
            continue
        tier = _classify_skill_tier(str(s.get("name") or ""))
        tier_priority = _TIER_PRIORITY.get(tier, 0)
        # Boost score by tier priority so tier1 always outranks tier4
        adjusted_sc = sc + tier_priority * 10.0
        scored.append((adjusted_sc, str(s.get("name") or "")))

    scored.sort(reverse=True)
    return [name for _, name in scored[:n]]
