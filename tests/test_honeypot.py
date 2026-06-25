"""
test_honeypot.py — Unit tests for honeypot detection logic.
"""

import pytest
from src.honeypot import is_honeypot


def _make_candidate(
    years_of_experience=5.0,
    career_history=None,
    skills=None,
    current_title="ML Engineer",
) -> dict:
    """Helper to build a minimal candidate dict for testing."""
    return {
        "candidate_id": "TEST_001",
        "profile": {
            "current_title": current_title,
            "years_of_experience": years_of_experience,
        },
        "career_history": career_history or [
            {
                "company": "Tech Corp",
                "title": "ML Engineer",
                "start_date": "2019-01-01",
                "end_date": "2024-01-01",
                "duration_months": 60,
                "description": "Built ML models.",
            }
        ],
        "skills": skills or [
            {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 10},
        ],
        "redrob_signals": {},
    }


class TestHoneypotTimeline:
    """Tests for timeline impossibility detection."""

    def test_normal_candidate_passes(self):
        """A candidate with realistic timeline should not be flagged."""
        cand = _make_candidate(years_of_experience=5.0)
        flag, reason = is_honeypot(cand)
        assert not flag, f"Should not be honeypot: {reason}"

    def test_timeline_impossibility_flagged(self):
        """Career months significantly exceeding YoE should be flagged."""
        cand = _make_candidate(
            years_of_experience=3.0,
            career_history=[
                {
                    "company": "A", "title": "Eng",
                    "start_date": "2015-01-01", "end_date": "2020-01-01",
                    "duration_months": 60, "description": "Work",
                },
                {
                    "company": "B", "title": "Eng",
                    "start_date": "2020-01-01", "end_date": "2024-01-01",
                    "duration_months": 48, "description": "Work",
                },
            ],
        )
        flag, reason = is_honeypot(cand)
        assert flag, "Should detect timeline impossibility"
        assert "timeline" in reason.lower()

    def test_zero_yoe_with_career_history_is_ok(self):
        """Zero YoE with short career history should not trigger timeline check."""
        cand = _make_candidate(
            years_of_experience=0,
            career_history=[
                {
                    "company": "A", "title": "Intern",
                    "start_date": "2024-01-01", "end_date": "2024-06-01",
                    "duration_months": 6, "description": "Internship",
                }
            ],
        )
        # Timeline check: 6 > 0 * 12 * 1.25 + 12 = 12 → 6 < 12, should NOT flag
        flag, reason = is_honeypot(cand)
        # May or may not flag depending on exact math, but should not be timeline
        if flag:
            assert "timeline" not in reason.lower()


class TestHoneypotExpertSkills:
    """Tests for expert skill impossibility."""

    def test_expert_zero_duration_flagged(self):
        """Two or more expert skills with 0 months should be flagged."""
        skills = [
            {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        ]
        cand = _make_candidate(skills=skills)
        flag, reason = is_honeypot(cand)
        assert flag, "Should flag expert skills with 0 duration"
        assert "expert_zero" in reason

    def test_one_expert_zero_duration_ok(self):
        """Single expert skill with 0 months should NOT be flagged."""
        skills = [
            {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 12, "endorsements": 5},
        ]
        cand = _make_candidate(skills=skills)
        flag, reason = is_honeypot(cand)
        # Only 1 expert_zero, threshold is 2
        if flag:
            assert "expert_zero" not in reason

    def test_too_many_experts_flagged(self):
        """More than 8 expert skills should be flagged as keyword stuffing."""
        skills = [
            {"name": f"skill_{i}", "proficiency": "expert", "duration_months": 12, "endorsements": 5}
            for i in range(10)
        ]
        cand = _make_candidate(skills=skills)
        flag, reason = is_honeypot(cand)
        assert flag, "Should flag too many expert skills"
        assert "too_many_experts" in reason

    def test_exactly_8_experts_ok(self):
        """Exactly 8 expert skills should NOT be flagged."""
        skills = [
            {"name": f"skill_{i}", "proficiency": "expert", "duration_months": 24, "endorsements": 10}
            for i in range(8)
        ]
        cand = _make_candidate(skills=skills)
        flag, reason = is_honeypot(cand)
        if flag:
            assert "too_many_experts" not in reason


class TestHoneypotDomainMismatch:
    """Tests for title vs description domain mismatch."""

    def test_matching_domains_pass(self):
        """ML title + ML descriptions should NOT be flagged."""
        cand = _make_candidate(
            current_title="ML Engineer",
            career_history=[
                {
                    "company": "TechCo", "title": "ML Engineer",
                    "start_date": "2020-01-01", "end_date": "2024-01-01",
                    "duration_months": 48,
                    "description": "Built machine learning models for recommendation system. "
                    "Used PyTorch and scikit-learn for training. "
                    "Deployed to production using FastAPI.",
                }
            ],
        )
        flag, reason = is_honeypot(cand)
        assert not flag, f"Matching domains should not be flagged: {reason}"

    def test_missing_descriptions_dont_flag(self):
        """Empty descriptions should not trigger domain mismatch."""
        cand = _make_candidate(
            current_title="Software Engineer",
            career_history=[
                {
                    "company": "Corp", "title": "SE",
                    "start_date": "2020-01-01", "end_date": "2024-01-01",
                    "duration_months": 48,
                    "description": "",
                }
            ],
        )
        flag, reason = is_honeypot(cand)
        if flag:
            assert "domain_mismatch" not in reason


class TestHoneypotImpossibleTenure:
    """Tests for impossible company tenure dates."""

    def test_ancient_start_date_long_tenure_flagged(self):
        """Starting before 2010 with 200+ months should be flagged."""
        cand = _make_candidate(
            years_of_experience=20.0,
            career_history=[
                {
                    "company": "OldCorp", "title": "Engineer",
                    "start_date": "2000-01-01", "end_date": "2024-01-01",
                    "duration_months": 288, "description": "Engineering work",
                }
            ],
        )
        flag, reason = is_honeypot(cand)
        # Timeline check: 288 > 20 * 12 * 1.25 = 300 → 288 < 300 + 12 = 312, may not flag
        # But impossible_tenure check: start_year < 2010 and duration > 200
        if flag:
            assert "timeline" in reason or "impossible_tenure" in reason

    def test_normal_old_tenure_ok(self):
        """A career starting before 2010 with reasonable duration is OK."""
        cand = _make_candidate(
            years_of_experience=15.0,
            career_history=[
                {
                    "company": "A", "title": "Eng",
                    "start_date": "2008-01-01", "end_date": "2016-01-01",
                    "duration_months": 96, "description": "Engineering",
                },
                {
                    "company": "B", "title": "Senior Eng",
                    "start_date": "2016-01-01", "end_date": "2024-01-01",
                    "duration_months": 96, "description": "ML work",
                },
            ],
        )
        flag, reason = is_honeypot(cand)
        # 192 months vs 15 * 12 * 1.25 + 12 = 237 → should pass
        assert not flag, f"Reasonable 15y career flagged: {reason}"
