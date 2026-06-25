"""
ranker.py — Combines all component scores into a final ranking.

Architecture:
  final_score = raw_fit_score × behavioral_multiplier

  raw_fit_score = (
      0.30 × skill_depth_score
    + 0.25 × career_trajectory_score
    + 0.20 × product_company_score
    + 0.15 × experience_band_score
    + 0.10 × location_score
  )

Processing pipeline:
  1. Stream candidates from JSONL (memory-efficient)
  2. Apply hard domain filter (skip ~85% non-tech candidates)
  3. Detect honeypots (score = 0)
  4. Score remaining candidates
  5. Keep top-K in a min-heap (heap of size 300 max)
  6. Sort top-100 and generate reasoning
  7. Normalize scores to [0, 1] with monotonically non-increasing guarantee
"""

from __future__ import annotations

import heapq
import time
from multiprocessing import Pool, cpu_count
from typing import Any

from src.behavioral import compute_behavioral_multiplier, get_behavioral_breakdown
from src.career_scorer import score_career
from src.filters import is_hard_filtered, _HARD_NEGATIVE_TITLE_WORDS
from src.honeypot import is_honeypot
from src.location_scorer import score_location
from src.product_scorer import score_product_company
from src.reasoner import generate_reasoning
from src.skill_scorer import score_skills

# ─── Score weights (must sum to 1.0) ─────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "skill_depth":         0.30,
    "career_trajectory":   0.25,
    "product_company":     0.20,
    "experience_band":     0.15,
    "location":            0.10,
}

# ─── Experience band scoring ──────────────────────────────────────────────────


def _experience_band_score(years: float) -> float:
    """Convert years of experience to a score based on JD target band.

    Target: 5-9 years (sweet spot: 6-8).
    Hard penalties for both under- and over-experienced candidates.

    Args:
        years: Years of experience (float).

    Returns:
        Experience band score in [0.0, 1.0].
    """
    if years <= 3.0:      # <= catches exactly 3.0y
        return 0.08    # Hard floor -- too junior
    elif years < 5.0:
        return 0.50
    elif years <= 9.0:
        return 1.00
    elif years <= 12.0:
        return 0.75
    elif years <= 14.0:
        return 0.35    # Probably senior/staff IC, outside JD target
    elif years <= 15.0:
        return 0.25    # Likely management track
    else:
        return 0.20    # 15+ years = wrong level for this role


def _education_bonus(candidate: dict[str, Any]) -> float:
    """Small bonus for top-tier education in a relevant field.

    IIT/IISc/BITS (tier_1) with CS/ML degree = 0.05
    IIT/IISc/BITS (tier_1) any field       = 0.03
    Tier-2 with relevant field             = 0.02

    Args:
        candidate: Full candidate dict.

    Returns:
        Bonus in [0.0, 0.05] (additive, applied to raw_fit before behavioral).
    """
    education = candidate.get("education", []) or []
    for edu in education:
        if not isinstance(edu, dict):
            continue
        tier = str(edu.get("tier") or "").lower().strip()
        field = str(edu.get("field_of_study") or "").lower()
        is_relevant = any(
            kw in field
            for kw in ("computer", "software", "machine learning",
                       "artificial intelligence", "data", "information")
        )
        if tier == "tier_1" and is_relevant:
            return 0.05
        if tier == "tier_1":
            return 0.03
        if tier == "tier_2" and is_relevant:
            return 0.02
    return 0.0


# ─── Single-candidate scoring (picklable for multiprocessing) ─────────────────


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Score a single candidate and return a result dict.

    This function is designed to be called via multiprocessing.Pool.map(),
    so it must be picklable (no closures over non-picklable objects).

    Args:
        candidate: Parsed candidate dict from JSONL.

    Returns:
        Result dict with keys: candidate_id, raw_score, final_score, scores,
        behavioral_multiplier.  Returns None if candidate should be skipped.
    """
    # ── Hard domain filter ────────────────────────────────────────────────
    filtered, filter_reason = is_hard_filtered(candidate)
    if filtered:
        return None

    # ── Honeypot detection ────────────────────────────────────────────────
    honeypot, hp_reason = is_honeypot(candidate)
    candidate_id = candidate.get("candidate_id", "UNKNOWN")
    profile = candidate.get("profile", {}) or {}
    years_exp = float(profile.get("years_of_experience") or 0)

    if honeypot:
        return {
            "candidate_id": candidate_id,
            "raw_score": 0.0,
            "final_score": 0.0,
            "scores": {k: 0.0 for k in WEIGHTS},
            "behavioral_multiplier": 0.0,
            "candidate": candidate,
            "is_honeypot": True,
        }

    # ── Component scoring ─────────────────────────────────────────────────
    skill_depth = score_skills(candidate)
    career_traj = score_career(candidate)
    product_co = score_product_company(candidate)
    exp_band = _experience_band_score(years_exp)
    location = score_location(candidate)

    raw_fit = (
        WEIGHTS["skill_depth"]       * skill_depth
        + WEIGHTS["career_trajectory"] * career_traj
        + WEIGHTS["product_company"]   * product_co
        + WEIGHTS["experience_band"]   * exp_band
        + WEIGHTS["location"]          * location
    )

    # ── Hard experience caps ──────────────────────────────────────────────
    # Junior: too junior regardless of other signals
    if years_exp <= 3.0:      # <= catches exactly 3.0y
        raw_fit = min(raw_fit, 0.30)
    elif years_exp < 4.0:
        raw_fit = min(raw_fit, 0.52)
    elif years_exp < 5.0:
        raw_fit = min(raw_fit, 0.65)   # borderline junior — not top-10 material
    # Overqualified: 13y+ is well outside the 5-9y IC target band
    if years_exp > 13.0:
        raw_fit = min(raw_fit, 0.42)   # was 0.55 — prevents 14y from ranking top 10
    elif years_exp > 11.0:
        raw_fit = min(raw_fit, 0.58)   # 11-13y: slightly overqualified, softer cap

    # ── Current title mismatch penalty ───────────────────────────────────
    # Pivot-back candidates: currently titled as non-tech (Civil Engineer etc.)
    # passed the career-history filter (they have past ML roles) but their
    # *current* role is non-tech — they are higher-risk than active ML candidates.
    current_title_lower = str(profile.get("current_title") or "").lower()
    is_currently_nontech = any(
        neg in current_title_lower for neg in _HARD_NEGATIVE_TITLE_WORDS
    )
    if is_currently_nontech:
        raw_fit *= 0.65   # 35% penalty — risky pivot-back hire

    # ── Skill floor gate ─────────────────────────────────────────────────
    # A Senior AI Engineer must have meaningful AI/ML skills.
    # A near-zero skill_depth must not rank high via company pedigree alone.
    # Three tiers of protection:
    #   <0.15 → severe cap   (SQL-only, no ML skills at all)
    #   <0.25 → moderate cap (1 Tier1 skill, very sparse)
    #   <0.35 → soft cap     (2 sparse skills, not enough for senior role)
    if skill_depth < 0.15:
        raw_fit = min(raw_fit, 0.40)
    elif skill_depth < 0.25:
        raw_fit = min(raw_fit, 0.60)
    elif skill_depth < 0.35:
        raw_fit = min(raw_fit, 0.70)   # NEW: sparse profile, soft cap

    # ── Education bonus (additive, before behavioral) ─────────────────────
    edu_bonus = _education_bonus(candidate)
    raw_fit = min(raw_fit + edu_bonus, 1.0)

    # ── Behavioral multiplier ─────────────────────────────────────────────
    beh_mult = compute_behavioral_multiplier(candidate)
    final_score = raw_fit * beh_mult

    return {
        "candidate_id": candidate_id,
        "raw_score": round(raw_fit, 6),
        "final_score": round(final_score, 6),
        "scores": {
            "skill_depth":       round(skill_depth, 4),
            "career_trajectory": round(career_traj, 4),
            "product_company":   round(product_co, 4),
            "experience_band":   round(exp_band, 4),
            "location":          round(location, 4),
        },
        "behavioral_multiplier": round(beh_mult, 4),
        "candidate": candidate,
        "is_honeypot": False,
    }


class CandidateRanker:
    """Main ranker that processes a JSONL file and produces a Top-100 ranking.

    Usage::

        ranker = CandidateRanker(top_k=100, keep_buffer=300)
        results = ranker.rank_from_file("candidates.jsonl")
        ranker.save_csv(results, "submission.csv")
    """

    def __init__(
        self,
        top_k: int = 100,
        keep_buffer: int = 300,
        use_multiprocessing: bool = True,
        show_progress: bool = True,
        max_candidates: int | None = None,
    ) -> None:
        """Initialize the ranker.

        Args:
            top_k: Number of candidates in final output (default 100).
            keep_buffer: Max in-memory heap size during processing.
                         Larger = more memory but safer for tie-breaking.
            use_multiprocessing: Whether to use multiprocessing for scoring.
            show_progress: Show tqdm progress bar.
            max_candidates: Cap on total candidates (for testing).
        """
        self.top_k = top_k
        self.keep_buffer = keep_buffer
        self.use_multiprocessing = use_multiprocessing
        self.show_progress = show_progress
        self.max_candidates = max_candidates

    def rank_from_file(self, path: str) -> list[dict[str, Any]]:
        """Full pipeline: load → score → rank → generate reasoning.

        Args:
            path: Path to JSONL file.

        Returns:
            Sorted list of top-100 result dicts with reasoning added.
        """
        from src.loader import stream_candidates

        t0 = time.perf_counter()

        # ── Streaming + scoring ───────────────────────────────────────────
        # We batch candidates into chunks for multiprocessing, but stream
        # from disk to avoid loading all 487 MB at once.

        top_heap: list[tuple[float, int, dict]] = []  # (score, tiebreak_idx, result)
        total_processed = 0
        total_skipped = 0
        tiebreak = 0

        from tqdm import tqdm

        if self.use_multiprocessing:
            n_workers = cpu_count()
            chunksize = 500
            batch: list[dict] = []

            def _process_batch(b: list[dict]) -> list[dict | None]:
                with Pool(processes=n_workers) as pool:
                    return pool.map(score_candidate, b, chunksize=min(chunksize, len(b)))

            stream = stream_candidates(
                path,
                max_candidates=self.max_candidates,
                show_progress=self.show_progress,
            )

            pbar = tqdm(
                desc="Scoring candidates",
                unit=" cands",
                disable=not self.show_progress,
            )

            for candidate in stream:
                batch.append(candidate)
                if len(batch) >= 5000:  # process in 5K chunks
                    results = _process_batch(batch)
                    for r in results:
                        pbar.update(1)
                        total_processed += 1
                        if r is None:
                            total_skipped += 1
                            continue
                        self._push_to_heap(top_heap, r, tiebreak)
                        tiebreak += 1
                    batch = []

            # Process remaining
            if batch:
                results = _process_batch(batch)
                for r in results:
                    pbar.update(1)
                    total_processed += 1
                    if r is None:
                        total_skipped += 1
                        continue
                    self._push_to_heap(top_heap, r, tiebreak)
                    tiebreak += 1

            pbar.close()

        else:
            # Single-process mode (useful for debugging / small datasets)
            stream = stream_candidates(
                path,
                max_candidates=self.max_candidates,
                show_progress=self.show_progress,
            )
            pbar = tqdm(
                stream,
                desc="Scoring candidates",
                unit=" cands",
                disable=not self.show_progress,
            )
            for candidate in pbar:
                total_processed += 1
                r = score_candidate(candidate)
                if r is None:
                    total_skipped += 1
                    continue
                self._push_to_heap(top_heap, r, tiebreak)
                tiebreak += 1

        elapsed = time.perf_counter() - t0

        print(
            f"\n[Done] Scored {total_processed:,} candidates "
            f"({total_skipped:,} filtered) in {elapsed:.1f}s"
        )

        # ── Extract top-K and finalize ─────────────────────────────────────
        top_results = heapq.nlargest(
            self.top_k,
            top_heap,
            key=lambda x: x[0],  # score
        )

        # ── Normalize scores to [0, 1] using ratio to max ─────────────────
        # CRITICAL: do NOT use min-max (rank-100 as zero) — that inflates weak
        # scores. A raw 0.37 must not become 0.98 just because rank-100 is 0.14.
        # Instead divide by max, preserving real score relationships.
        if top_results:
            max_score = top_results[0][0]
            if max_score > 0:
                normalized = [
                    (min(score / max_score, 1.0), idx, result)
                    for score, idx, result in top_results
                ]
            else:
                normalized = [(0.0, idx, result) for score, idx, result in top_results]
        else:
            normalized = top_results

        # ── Build final output with reasoning ─────────────────────────────
        final_output = []
        for rank_idx, (norm_score, _, result) in enumerate(normalized, start=1):
            candidate = result["candidate"]
            scores = result["scores"]

            reasoning = generate_reasoning(
                candidate=candidate,
                rank=rank_idx,
                scores=scores,
                final_score=norm_score,
            )

            final_output.append({
                "candidate_id": result["candidate_id"],
                "rank": rank_idx,
                "score": round(norm_score, 6),
                "raw_score": result["raw_score"],
                "reasoning": reasoning,
                "scores": scores,
                "behavioral_multiplier": result["behavioral_multiplier"],
                "candidate": candidate,
            })

        # ── Guarantee monotonically non-increasing scores ─────────────────
        final_output = _enforce_monotonic_scores(final_output)

        t_total = time.perf_counter() - t0
        print(f"[Done] Ranking complete in {t_total:.1f}s total")

        return final_output

    def _push_to_heap(
        self,
        heap: list,
        result: dict[str, Any],
        tiebreak: int,
    ) -> None:
        """Push a result onto the min-heap, maintaining max size = keep_buffer.

        Args:
            heap: The min-heap (list of (score, tiebreak, result) tuples).
            result: Scored candidate result dict.
            tiebreak: Unique integer for stable ordering on score ties.
        """
        score = result["final_score"]
        entry = (score, tiebreak, result)

        if len(heap) < self.keep_buffer:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:  # heap[0] is the smallest score
            heapq.heapreplace(heap, entry)

    def save_csv(
        self,
        ranked_results: list[dict[str, Any]],
        output_path: str,
    ) -> None:
        """Save the ranked results to a CSV file in submission format.

        Output columns: candidate_id, rank, score, reasoning.

        Args:
            ranked_results: Output of rank_from_file().
            output_path: Path to write the CSV file.
        """
        import csv
        import os

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["candidate_id", "rank", "score", "reasoning"],
            )
            writer.writeheader()
            for row in ranked_results:
                writer.writerow({
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": f"{row['score']:.6f}",
                    "reasoning": row["reasoning"],
                })

        print(f"[Done] Saved {len(ranked_results)} rows -> {output_path}")


def _enforce_monotonic_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure scores are monotonically non-increasing across ranks.

    If a tiny floating-point issue causes rank N to have a higher score
    than rank N-1, clamp it down.

    Args:
        results: List of result dicts sorted by rank ascending.

    Returns:
        Results with scores guaranteed non-increasing.
    """
    if not results:
        return results
    prev_score = results[0]["score"]
    for r in results[1:]:
        if r["score"] > prev_score:
            r["score"] = prev_score
        prev_score = r["score"]
    return results


def rank_candidates_from_list(
    candidates: list[dict[str, Any]],
    top_k: int = 100,
) -> list[dict[str, Any]]:
    """Score and rank a list of candidates (for Streamlit demo / small datasets).

    Uses single-process scoring for compatibility with Streamlit's environment.

    Args:
        candidates: List of parsed candidate dicts.
        top_k: Number of candidates to return.

    Returns:
        Sorted list of top-k result dicts with reasoning.
    """
    scored = []
    for candidate in candidates:
        r = score_candidate(candidate)
        if r is not None:
            scored.append(r)

    # Sort by final_score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Take top-k
    top = scored[:top_k]

    # Normalize scores: ratio to max preserves real score differences
    # (min-max against rank-100 would inflate weak scores falsely)
    if top:
        max_s = top[0]["final_score"]
        if max_s > 0:
            for r in top:
                r["score"] = round(min(r["final_score"] / max_s, 1.0), 6)
        else:
            for r in top:
                r["score"] = 0.0
    else:
        for r in top:
            r["score"] = r["final_score"]

    # Add ranking and reasoning
    final_output = []
    for rank_idx, result in enumerate(top, start=1):
        candidate = result["candidate"]
        reasoning = generate_reasoning(
            candidate=candidate,
            rank=rank_idx,
            scores=result["scores"],
            final_score=result.get("score", result["final_score"]),
        )
        final_output.append({
            "candidate_id": result["candidate_id"],
            "rank": rank_idx,
            "score": result.get("score", result["final_score"]),
            "raw_score": result["raw_score"],
            "reasoning": reasoning,
            "scores": result["scores"],
            "behavioral_multiplier": result["behavioral_multiplier"],
            "candidate": candidate,
        })

    return _enforce_monotonic_scores(final_output)
