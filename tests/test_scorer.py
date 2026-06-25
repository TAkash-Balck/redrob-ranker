"""
test_scorer.py — Unit tests for the individual scoring components.

Tests calibrated against CAND_0000031 (the "gold standard" candidate):
  Recommendation Systems Engineer, 6y, Hyderabad, Swiggy/Uber/Zomato
  Should score near the top of all components.
"""

import pytest
from src.skill_scorer import score_skills, get_top_matched_skills
from src.career_scorer import score_career, get_career_metadata
from src.product_scorer import score_product_company, get_top_product_companies
from src.behavioral import compute_behavioral_multiplier, get_behavioral_breakdown
from src.location_scorer import score_location


# ─── Fixture: Gold-standard candidate ────────────────────────────────────────

def _gold_standard_candidate() -> dict:
    """Build a candidate resembling CAND_0000031 (best expected candidate)."""
    return {
        "candidate_id": "CAND_GOLD_001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "headline": "Recommendation Systems & Vector Search @ Swiggy",
            "summary": "6 years building production recommendation systems using embeddings and FAISS",
            "location": "Hyderabad",
            "country": "India",
            "years_of_experience": 6.0,
            "current_company": "Swiggy",
            "current_company_size": "1001-5000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Swiggy",
                "title": "Senior ML Engineer",
                "start_date": "2021-01-01",
                "end_date": "2024-01-01",
                "duration_months": 36,
                "is_current": True,
                "industry": "Technology",
                "company_size": "1001-5000",
                "description": (
                    "Built production vector search system using FAISS and sentence-transformers. "
                    "Shipped ranking model (NDCG@10 improved by 8%). "
                    "Deployed recommendation system serving 5M real users. "
                    "A/B testing framework for online evaluation."
                ),
            },
            {
                "company": "Uber",
                "title": "ML Engineer",
                "start_date": "2019-01-01",
                "end_date": "2021-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "Technology",
                "company_size": "10001+",
                "description": (
                    "Semantic search and information retrieval for Uber Eats. "
                    "Embedding-based candidate generation pipeline. "
                    "Elasticsearch integration for hybrid search."
                ),
            },
            {
                "company": "Zomato",
                "title": "Data Scientist",
                "start_date": "2018-01-01",
                "end_date": "2019-01-01",
                "duration_months": 12,
                "is_current": False,
                "industry": "Technology",
                "company_size": "1001-5000",
                "description": "Built recommendation features for restaurant discovery. NLP for query understanding.",
            },
        ],
        "skills": [
            {"name": "FAISS", "proficiency": "expert", "duration_months": 36, "endorsements": 25},
            {"name": "sentence-transformers", "proficiency": "advanced", "duration_months": 36, "endorsements": 20},
            {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 24, "endorsements": 15},
            {"name": "Information Retrieval", "proficiency": "advanced", "duration_months": 48, "endorsements": 18},
            {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 30},
            {"name": "NLP", "proficiency": "advanced", "duration_months": 48, "endorsements": 12},
            {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36, "endorsements": 10},
            {"name": "Learning-to-Rank", "proficiency": "advanced", "duration_months": 24, "endorsements": 8},
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.9,
            "notice_period_days": 30,
            "github_activity_score": 75,
            "willing_to_relocate": True,
        },
        "education": [],
        "certifications": [],
    }


def _poor_candidate() -> dict:
    """Build a candidate representing a poor fit (HR Manager at TCS)."""
    return {
        "candidate_id": "CAND_POOR_001",
        "profile": {
            "current_title": "HR Manager",
            "headline": "HR Professional at TCS",
            "summary": "8 years in human resources management",
            "location": "Mumbai",
            "country": "India",
            "years_of_experience": 8.0,
            "current_company": "TCS",
            "current_company_size": "10001+",
            "current_industry": "IT Services",
        },
        "career_history": [
            {
                "company": "TCS",
                "title": "HR Manager",
                "start_date": "2016-01-01",
                "end_date": "2024-01-01",
                "duration_months": 96,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Managed recruitment, onboarding, and employee relations. Conducted performance reviews.",
            }
        ],
        "skills": [
            {"name": "Human Resources", "proficiency": "expert", "duration_months": 96, "endorsements": 50},
            {"name": "Recruitment", "proficiency": "advanced", "duration_months": 96, "endorsements": 30},
            {"name": "Excel", "proficiency": "intermediate", "duration_months": 96, "endorsements": 10},
        ],
        "redrob_signals": {
            "last_active_date": "2025-01-01",
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.3,
            "notice_period_days": 90,
            "github_activity_score": -1,
            "willing_to_relocate": False,
        },
        "education": [],
        "certifications": [],
    }


# ─── Skill Scorer Tests ───────────────────────────────────────────────────────

class TestSkillScorer:
    def test_gold_standard_high_skill_score(self):
        """Gold standard candidate should have high skill score."""
        cand = _gold_standard_candidate()
        score = score_skills(cand)
        assert score > 0.6, f"Gold standard should have skill score > 0.6, got {score:.3f}"

    def test_poor_candidate_low_skill_score(self):
        """HR Manager should have very low skill score."""
        cand = _poor_candidate()
        score = score_skills(cand)
        assert score < 0.2, f"HR Manager should have skill score < 0.2, got {score:.3f}"

    def test_tier1_skills_weighted_highest(self):
        """Tier1 skills (FAISS, embeddings) should outweigh Tier4 (Docker)."""
        tier1_cand = {
            "candidate_id": "T1",
            "profile": {"years_of_experience": 5.0},
            "career_history": [],
            "skills": [
                {"name": "FAISS", "proficiency": "advanced", "duration_months": 24, "endorsements": 10},
            ],
            "redrob_signals": {},
        }
        tier4_cand = {
            "candidate_id": "T4",
            "profile": {"years_of_experience": 5.0},
            "career_history": [],
            "skills": [
                {"name": "Docker", "proficiency": "advanced", "duration_months": 24, "endorsements": 10},
            ],
            "redrob_signals": {},
        }
        assert score_skills(tier1_cand) > score_skills(tier4_cand)

    def test_empty_skills_zero_score(self):
        """Candidate with no skills should get near-zero skill score."""
        cand = {
            "candidate_id": "EMPTY",
            "profile": {"years_of_experience": 5.0},
            "career_history": [],
            "skills": [],
            "redrob_signals": {},
        }
        score = score_skills(cand)
        assert score < 0.1

    def test_top_matched_skills_returns_tier1_first(self):
        """Top matched skills should prioritize Tier1 skills."""
        cand = _gold_standard_candidate()
        top = get_top_matched_skills(cand, n=3)
        assert len(top) >= 2
        # FAISS or sentence-transformers should appear in top 3
        top_lower = [s.lower() for s in top]
        has_tier1 = any(
            any(t in s for t in ["faiss", "sentence", "retrieval", "elasticsearch", "rank"])
            for s in top_lower
        )
        assert has_tier1, f"Expected Tier1 skill in top 3, got: {top}"

    def test_career_description_bonus_applied(self):
        """Career descriptions with IR keywords should boost score."""
        no_desc_cand = {
            "candidate_id": "NO_DESC",
            "profile": {"years_of_experience": 5.0},
            "career_history": [
                {
                    "company": "Corp",
                    "title": "Engineer",
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 48,
                    "description": "",
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 24, "endorsements": 10},
            ],
            "redrob_signals": {},
        }
        with_desc_cand = {
            **no_desc_cand,
            "candidate_id": "WITH_DESC",
            "career_history": [
                {
                    "company": "Corp",
                    "title": "Engineer",
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 48,
                    "description": "Shipped ranking system with vector search, embedding-based retrieval, NDCG evaluation",
                }
            ],
        }
        assert score_skills(with_desc_cand) > score_skills(no_desc_cand)


# ─── Career Scorer Tests ──────────────────────────────────────────────────────

class TestCareerScorer:
    def test_gold_standard_high_career_score(self):
        """Product company + ML titles should score high."""
        cand = _gold_standard_candidate()
        score = score_career(cand)
        assert score > 0.7, f"Gold standard should have career score > 0.7, got {score:.3f}"

    def test_consulting_heavy_penalized(self):
        """100% consulting career should be penalized."""
        consulting_cand = {
            "candidate_id": "CONSULT",
            "profile": {"years_of_experience": 8.0, "current_title": "ML Engineer"},
            "career_history": [
                {
                    "company": "Infosys",
                    "title": "Senior Engineer",
                    "start_date": "2016-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 96,
                    "description": "Worked on various client projects. Maintained legacy systems.",
                }
            ],
            "skills": [],
            "redrob_signals": {},
        }
        score = score_career(consulting_cand)
        assert score < 0.6, f"Heavy consulting career should score < 0.6, got {score:.3f}"

    def test_consulting_with_ml_work_less_penalized(self):
        """Consulting with clear ML/IR work should get reduced penalty."""
        with_ml = {
            "candidate_id": "CONSULT_ML",
            "profile": {"years_of_experience": 5.0, "current_title": "ML Engineer"},
            "career_history": [
                {
                    "company": "TCS",
                    "title": "ML Engineer",
                    "start_date": "2019-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 60,
                    "description": "Built embedding-based retrieval system. Deployed recommendation model. A/B testing.",
                }
            ],
            "skills": [],
            "redrob_signals": {},
        }
        without_ml = {
            "candidate_id": "CONSULT_NO_ML",
            "profile": {"years_of_experience": 5.0, "current_title": "System Analyst"},
            "career_history": [
                {
                    "company": "TCS",
                    "title": "System Analyst",
                    "start_date": "2019-01-01",
                    "end_date": "2024-01-01",
                    "duration_months": 60,
                    "description": "Maintained legacy Java applications. Client support and documentation.",
                }
            ],
            "skills": [],
            "redrob_signals": {},
        }
        assert score_career(with_ml) > score_career(without_ml)

    def test_job_hopper_penalized(self):
        """3+ short stints (< 18 months) should apply 0.8x penalty."""
        hopper = {
            "candidate_id": "HOPPER",
            "profile": {"years_of_experience": 5.0, "current_title": "ML Engineer"},
            "career_history": [
                {"company": f"Co{i}", "title": "ML Engineer",
                 "start_date": f"202{i}-01-01", "end_date": f"202{i}-10-01",
                 "duration_months": 9, "description": "ML work."}
                for i in range(5)
            ],
            "skills": [],
            "redrob_signals": {},
        }
        stable = {
            "candidate_id": "STABLE",
            "profile": {"years_of_experience": 5.0, "current_title": "ML Engineer"},
            "career_history": [
                {"company": "Swiggy", "title": "ML Engineer",
                 "start_date": "2019-01-01", "end_date": "2024-01-01",
                 "duration_months": 60, "description": "ML work."}
            ],
            "skills": [],
            "redrob_signals": {},
        }
        assert score_career(stable) > score_career(hopper)

    def test_career_metadata_consulting_ratio(self):
        """get_career_metadata should correctly compute consulting_ratio."""
        cand = {
            "candidate_id": "META",
            "profile": {"years_of_experience": 8.0, "current_title": "Analyst"},
            "career_history": [
                {"company": "Wipro", "title": "Analyst",
                 "start_date": "2016-01-01", "end_date": "2020-01-01",
                 "duration_months": 48, "description": "Analytics work"},
                {"company": "Google", "title": "Data Scientist",
                 "start_date": "2020-01-01", "end_date": "2024-01-01",
                 "duration_months": 48, "description": "ML work"},
            ],
            "skills": [],
            "redrob_signals": {},
        }
        meta = get_career_metadata(cand)
        assert abs(meta["consulting_ratio"] - 0.5) < 0.01
        assert meta["consulting_pct"] == 50


# ─── Product Company Scorer Tests ─────────────────────────────────────────────

class TestProductScorer:
    def test_swiggy_uber_zomato_score_high(self):
        """Swiggy/Uber/Zomato career should score close to 1.0."""
        cand = _gold_standard_candidate()
        score = score_product_company(cand)
        assert score > 0.8, f"Swiggy+Uber+Zomato should score > 0.8, got {score:.3f}"

    def test_pure_tcs_scores_zero(self):
        """TCS-only career should get 0 product company score."""
        cand = _poor_candidate()
        cand["profile"]["current_company"] = "TCS"
        score = score_product_company(cand)
        assert score == 0.0, f"TCS should score 0 for product company"

    def test_score_caps_at_one(self):
        """Product score should cap at 1.0 regardless of total months."""
        cand = {
            "candidate_id": "LONG",
            "profile": {"years_of_experience": 15.0, "current_company": "Google"},
            "career_history": [
                {"company": "Google", "title": "Engineer",
                 "start_date": "2010-01-01", "end_date": "2024-01-01",
                 "duration_months": 168, "description": "Engineering"}
            ],
            "skills": [],
            "redrob_signals": {},
        }
        assert score_product_company(cand) == 1.0

    def test_four_years_equals_full_score(self):
        """48+ months at product companies = score 1.0."""
        cand = {
            "candidate_id": "FOUR_YR",
            "profile": {"years_of_experience": 5.0, "current_company": "Flipkart"},
            "career_history": [
                {"company": "Flipkart", "title": "ML Engineer",
                 "start_date": "2020-01-01", "end_date": "2024-01-01",
                 "duration_months": 48, "description": "ML work"}
            ],
            "skills": [],
            "redrob_signals": {},
        }
        assert score_product_company(cand) == 1.0

    def test_get_top_companies_returns_known_names(self):
        """get_top_product_companies should return actual company names."""
        cand = _gold_standard_candidate()
        companies = get_top_product_companies(cand, n=2)
        assert len(companies) > 0
        # Should include at least one of Swiggy, Uber, Zomato
        known = {"swiggy", "uber", "zomato"}
        found = any(c.lower() in known for c in companies)
        assert found, f"Expected known product company, got: {companies}"


# ─── Behavioral Multiplier Tests ──────────────────────────────────────────────

class TestBehavioral:
    def test_active_open_responsive_scores_near_one(self):
        """Highly active, open to work, responsive candidate should score ~1.0."""
        cand = {
            "candidate_id": "ACTIVE",
            "profile": {},
            "redrob_signals": {
                "last_active_date": "2026-06-06",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.95,
                "notice_period_days": 0,
                "github_activity_score": 80,
            },
        }
        mult = compute_behavioral_multiplier(cand)
        # 1.0 (recency) * 1.0 (available) * 0.97 (0.4+0.6*0.95) * 1.0 (0d notice) * 1.05 (github>60)
        # = ~0.8963 — good active candidate
        assert mult > 0.85, f"Very active candidate should have multiplier > 0.85, got {mult}"

    def test_inactive_long_notice_scores_low(self):
        """Inactive 200 days + 180 day notice should have a relatively low multiplier.

        The behavioral multiplier is compressed via 0.5 + 0.5 * raw_mult so the
        absolute floor rises (~0.55 for worst case), but the relative ordering is
        preserved: a bad candidate still scores well below a good candidate.
        The threshold here is < 0.65 (compressed worst-case) vs > 0.85 for the
        active candidate tested above.
        """
        cand = {
            "candidate_id": "INACTIVE",
            "profile": {},
            "redrob_signals": {
                "last_active_date": "2025-11-01",  # 218 days before 2026-06-07
                "open_to_work_flag": False,
                "recruiter_response_rate": 0.1,
                "notice_period_days": 180,
                "github_activity_score": -1,
            },
        }
        mult = compute_behavioral_multiplier(cand)
        assert mult < 0.65, f"Inactive+long notice should have multiplier < 0.65, got {mult}"

    def test_multiplier_capped_above_zero(self):
        """Multiplier should never be exactly 0 for valid candidates."""
        cand = {
            "candidate_id": "WORST",
            "profile": {},
            "redrob_signals": {
                "last_active_date": "2020-01-01",
                "open_to_work_flag": False,
                "recruiter_response_rate": 0.0,
                "notice_period_days": 180,
                "github_activity_score": -1,
            },
        }
        mult = compute_behavioral_multiplier(cand)
        assert mult > 0.0, "Multiplier should be positive even for worst case"

    def test_github_bonus_applied(self):
        """High GitHub activity should give small bonus."""
        base = {
            "candidate_id": "GH_BASE",
            "profile": {},
            "redrob_signals": {
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.8,
                "notice_period_days": 30,
                "github_activity_score": 0,  # low
            },
        }
        high_gh = {
            **base,
            "candidate_id": "GH_HIGH",
            "redrob_signals": {
                **base["redrob_signals"],
                "github_activity_score": 80,  # high
            },
        }
        assert compute_behavioral_multiplier(high_gh) > compute_behavioral_multiplier(base)

    def test_breakdown_keys_present(self):
        """get_behavioral_breakdown should return all expected keys."""
        cand = _gold_standard_candidate()
        breakdown = get_behavioral_breakdown(cand)
        required_keys = {"recency", "availability", "responsiveness", "notice", "github", "multiplier"}
        assert required_keys.issubset(set(breakdown.keys()))


# ─── Location Scorer Tests ────────────────────────────────────────────────────

class TestLocationScorer:
    def test_noida_scores_one(self):
        """Noida location should score 1.0."""
        cand = {
            "profile": {"location": "Noida, UP", "country": "India"},
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert score_location(cand) == 1.0

    def test_pune_scores_one(self):
        """Pune should score 1.0."""
        cand = {
            "profile": {"location": "Pune, Maharashtra", "country": "India"},
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert score_location(cand) == 1.0

    def test_gurgaon_scores_0_90(self):
        """Gurgaon (NCR) should score 0.90."""
        cand = {
            "profile": {"location": "Gurgaon", "country": "India"},
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert score_location(cand) == 0.90

    def test_hyderabad_scores_0_85(self):
        """Hyderabad should score 0.85."""
        cand = {
            "profile": {"location": "Hyderabad", "country": "India"},
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert score_location(cand) == 0.85

    def test_india_relocate_scores_0_75(self):
        """India + willing to relocate should score 0.75."""
        cand = {
            "profile": {"location": "Mysore", "country": "India"},
            "redrob_signals": {"willing_to_relocate": True},
        }
        assert score_location(cand) == 0.75

    def test_india_no_relocate_scores_0_55(self):
        """India + not willing to relocate + non-metro should score 0.55."""
        cand = {
            "profile": {"location": "Bhilai", "country": "India"},
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert score_location(cand) == 0.55

    def test_foreign_no_relocate_scores_0_20(self):
        """Non-India + not willing to relocate should score 0.20."""
        cand = {
            "profile": {"location": "San Francisco, CA", "country": "USA"},
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert score_location(cand) == 0.20

    def test_gold_standard_hyderabad_scores_high(self):
        """CAND_0000031 (Hyderabad) should score 0.85."""
        cand = _gold_standard_candidate()
        score = score_location(cand)
        # Hyderabad + willing to relocate → should get 0.85 or higher
        assert score >= 0.85


# ─── Integration: Gold vs Poor ────────────────────────────────────────────────

class TestIntegration:
    def test_gold_beats_poor_on_all_components(self):
        """Gold standard should score higher than poor candidate on every component."""
        gold = _gold_standard_candidate()
        poor = _poor_candidate()

        gold_skill = score_skills(gold)
        poor_skill = score_skills(poor)
        assert gold_skill > poor_skill, f"Gold skill {gold_skill:.3f} should beat poor {poor_skill:.3f}"

        gold_career = score_career(gold)
        poor_career = score_career(poor)
        assert gold_career > poor_career, f"Gold career {gold_career:.3f} should beat poor {poor_career:.3f}"

        gold_product = score_product_company(gold)
        poor_product = score_product_company(poor)
        assert gold_product > poor_product, f"Gold product {gold_product:.3f} should beat poor {poor_product:.3f}"

    def test_final_score_ordering(self):
        """Full scoring should rank gold significantly above poor."""
        from src.ranker import score_candidate, WEIGHTS

        gold = _gold_standard_candidate()
        poor = _poor_candidate()

        gold_result = score_candidate(gold)
        poor_result = score_candidate(poor)

        assert gold_result is not None, "Gold standard should not be filtered"
        # Poor candidate (HR Manager) might be filtered — that's OK
        if poor_result is not None:
            assert gold_result["final_score"] > poor_result["final_score"], (
                f"Gold ({gold_result['final_score']:.4f}) should beat poor ({poor_result['final_score']:.4f})"
            )
