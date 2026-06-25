"""
product_scorer.py — Product company detection and scoring.

A candidate who has spent significant time at product companies
(Swiggy, Zomato, Uber, Google, etc.) demonstrates exposure to
real-scale engineering problems that matter for this role.

This is one of the most predictive signals: consulting firm engineers
rarely encounter the search/ranking/retrieval systems that this role requires.
"""

from __future__ import annotations

from typing import Any

# ─── Known product companies (lowercase for matching) ────────────────────────

_PRODUCT_COMPANIES: frozenset[str] = frozenset({
    # India unicorns / product companies
    "swiggy", "zomato", "uber", "ola", "flipkart", "meesho", "razorpay",
    "cred", "zepto", "dunzo", "groww", "upstox", "smallcase", "freshworks",
    "zoho", "chargebee", "browserstack", "postman", "hasura", "setu",
    "sarvam", "krutrim", "turing", "clevertap", "moengage", "sharechat",
    "dailyhunt", "dream11", "unacademy", "byju", "lenskart", "pharmeasy",
    "urban company", "nykaa", "boat", "mamaearth", "vedantu", "cars24",
    "spinny", "rapido", "yulu", "bounce", "ather", "ola electric",
    "slice", "jupiter", "fi money", "niyo", "open financial",
    "cashfree", "juspay", "paytm", "phonepe", "gpay", "google pay",
    "myntra", "ajio", "navi", "khatabook", "vyapar", "dukaan",
    "apna", "foundit", "naukri", "info edge", "just dial", "indiamart",
    "policybazaar", "acko", "digit insurance", "coverfox",
    "practo", "tata 1mg", "netmeds", "medlife", "cult.fit",
    "ola cabs", "meru", "blablacar", "shuttl", "tummoc",
    "porter", "shiprocket", "delhivery", "shadowfax", "xpressbees",
    "ecom express", "ekart",
    "hotstar", "disney+ hotstar", "zee5", "sonyliv", "jio cinema",
    "netflix", "amazon prime", "mxplayer",
    "quora", "reddit", "twitter", "x.com",
    "linkedin", "glassdoor", "indeed",
    # Global tech
    "google", "microsoft", "amazon", "meta", "facebook", "apple",
    "netflix", "spotify", "airbnb", "stripe", "twilio", "datadog",
    "snowflake", "databricks", "hugging face", "huggingface",
    "cohere", "openai", "anthropic", "deepmind", "nvidia",
    "salesforce", "atlassian", "slack", "shopify", "figma",
    "notion", "linear", "vercel", "cloudflare", "hashicorp",
    "elastic", "confluent", "dremio", "pinecone", "weaviate", "qdrant",
    "mad street den", "fractal analytics", "tiger analytics",
    "thoughtworks", "walmart labs", "walmart global tech",
    "adobe", "oracle", "sap", "workday", "servicenow", "zendesk",
    "hubspot", "segment", "mixpanel", "amplitude", "grafana",
    "mongodb", "redis", "cockroachdb", "planetscale",
    "instacart", "doordash", "lyft", "robinhood", "coinbase",
    "brex", "plaid", "rippling", "gusto", "lattice",
    "pagerduty", "sumo logic", "splunk", "new relic", "dynatrace",
    "sentry", "sauce labs",
    "qualcomm", "intel", "amd", "arm", "broadcom",
    "samsung r&d", "samsung research",
    "ibm research", "ibm",
    "yahoo", "yahoo labs",
    "booking.com", "expedia", "airbnb", "trivago",
    "grab", "gojek", "sea group", "shopee", "tokopedia",
    "bytebyance", "bytedance", "tiktok", "baidu", "alibaba", "tencent",
    "jd.com", "meituan", "didi",
    # Borderline but better than pure consulting
    "mu sigma", "sigmoid",
})

# ─── Company size signals ─────────────────────────────────────────────────────

# Company sizes that suggest product/startup culture when combined with ML industry
_STARTUP_SIZES: frozenset[str] = frozenset({
    "1-10", "11-50", "51-200", "201-500",
})

_IT_SERVICES_SIZES: frozenset[str] = frozenset({
    "10001+",
})

_ML_AI_INDUSTRIES: frozenset[str] = frozenset({
    "artificial intelligence", "machine learning", "technology",
    "software", "internet", "e-commerce", "fintech", "saas",
    "computer software", "information technology",
})


def _is_product_company(
    company_name: str,
    company_size: str = "",
    industry: str = "",
) -> bool:
    """Determine if a company is a known product/tech company.

    Args:
        company_name: Company name string.
        company_size: Size enum string (e.g., "11-50").
        industry: Industry string.

    Returns:
        True if the company is a product/tech company.
    """
    lower = company_name.lower().strip()

    # Exact match first
    if lower in _PRODUCT_COMPANIES:
        return True

    # Substring match for company name
    for known in _PRODUCT_COMPANIES:
        if known in lower:
            return True

    # Heuristic: small company in ML/AI industry = startup signal
    if company_size in _STARTUP_SIZES:
        industry_lower = industry.lower()
        if any(ml in industry_lower for ml in _ML_AI_INDUSTRIES):
            return True

    return False


def score_product_company(candidate: dict[str, Any]) -> float:
    """Compute the product company exposure score.

    Calculates the total months spent at known product/tech companies,
    normalized so 4+ years (48 months) = score 1.0.

    Args:
        candidate: Full candidate dict.

    Returns:
        Product company score in [0.0, 1.0].
    """
    career_history = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}

    product_months = 0.0
    # Also check current company even if not in career_history
    current_company = str(profile.get("current_company") or "").lower()
    current_industry = str(profile.get("current_industry") or "")
    current_company_size = str(profile.get("current_company_size") or "")

    if not career_history:
        # Score based only on current company
        if _is_product_company(current_company, current_company_size, current_industry):
            years = float(profile.get("years_of_experience") or 0)
            product_months = min(years * 12, 48.0)
        return min(product_months / 48.0, 1.0)

    for job in career_history:
        if not isinstance(job, dict):
            continue
        company = str(job.get("company") or "")
        size = str(job.get("company_size") or "")
        industry = str(job.get("industry") or "")
        duration = float(job.get("duration_months") or 0)

        if _is_product_company(company, size, industry):
            product_months += duration

    return min(product_months / 48.0, 1.0)


def get_top_product_companies(
    candidate: dict[str, Any],
    n: int = 2,
) -> list[str]:
    """Return the top N most notable product companies from career history.

    Uses only career_history entries — NOT profile.current_company.
    The profile.current_company field is frequently inconsistent with
    actual career_history data (it can be stale or wrong), and using it
    causes phantom company pairings in reasoning output.

    Prioritizes globally recognized companies by order of appearance in
    career history (most recent first, as career_history is typically ordered).

    Args:
        candidate: Full candidate dict.
        n: Number of company names to return.

    Returns:
        List of company name strings from actual career_history only.
    """
    career_history = candidate.get("career_history", []) or []

    results = []
    for job in career_history:
        if not isinstance(job, dict):
            continue
        company = str(job.get("company") or "")
        size = str(job.get("company_size") or "")
        industry = str(job.get("industry") or "")
        if company and _is_product_company(company.lower(), size, industry):
            if company not in results:
                results.append(company)
        if len(results) >= n:
            break

    return results[:n]

