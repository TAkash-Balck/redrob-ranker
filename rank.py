#!/usr/bin/env python3
"""
rank.py — MAIN ENTRY POINT for the Redrob Candidate Ranker.

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
    python rank.py --candidates candidates.jsonl --out submission.csv --workers 8
    python rank.py --candidates candidates.jsonl --out submission.csv --no-mp  # single process
    python rank.py --candidates candidates.jsonl --out submission.csv --max 1000  # test run

Constraints guaranteed:
    - Exactly 100 rows in output
    - Ranks 1–100 used exactly once
    - Scores monotonically non-increasing
    - All scores in [0.0, 1.0]
    - Runs in <5 minutes on 4-core CPU (typically ~60–90s)
    - Peak RAM ~2 GB (not all 487 MB loaded at once)
    - No external API calls
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranker — produces top-100 submission CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full production run
  python rank.py --candidates candidates.jsonl --out submission.csv

  # Quick test with first 5000 candidates
  python rank.py --candidates candidates.jsonl --out test_out.csv --max 5000

  # Single-process mode (easier debugging)
  python rank.py --candidates candidates.jsonl --out submission.csv --no-mp
        """,
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to the candidates.jsonl file",
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of candidates to output (default: 100)",
    )
    parser.add_argument(
        "--buffer", "-b",
        type=int,
        default=300,
        help="In-memory heap buffer size (default: 300)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Cap total candidates processed (for testing)",
    )
    parser.add_argument(
        "--no-mp",
        action="store_true",
        help="Disable multiprocessing (single-process mode)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )
    return parser.parse_args()


def validate_output(results: list[dict], top_k: int) -> list[str]:
    """Validate the output meets all submission requirements.

    Args:
        results: List of ranked result dicts.
        top_k: Expected number of rows.

    Returns:
        List of error messages (empty if all checks pass).
    """
    errors = []

    # Check row count
    if len(results) != top_k:
        errors.append(f"Expected {top_k} rows, got {len(results)}")

    # Check ranks 1–top_k exactly once
    ranks = [r["rank"] for r in results]
    expected_ranks = set(range(1, top_k + 1))
    if set(ranks) != expected_ranks:
        errors.append(f"Rank set mismatch: got {sorted(ranks)[:5]}...")

    # Check scores in [0, 1]
    scores = [r["score"] for r in results]
    if any(s < 0.0 or s > 1.0 for s in scores):
        bad = [s for s in scores if s < 0.0 or s > 1.0]
        errors.append(f"Scores out of [0,1] range: {bad[:3]}")

    # Check monotonically non-increasing
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:  # allow tiny float tolerance
            errors.append(
                f"Score not monotonic at rank {i+1}: "
                f"{scores[i-1]:.6f} → {scores[i]:.6f}"
            )
            break

    # Check reasoning is non-empty
    empty_reasoning = [r["candidate_id"] for r in results if not r.get("reasoning", "").strip()]
    if empty_reasoning:
        errors.append(f"Empty reasoning for: {empty_reasoning[:3]}")

    return errors


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = parse_args()
    t_start = time.perf_counter()

    # ── Validate input file ───────────────────────────────────────────────
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: Candidate file not found: {args.candidates}", file=sys.stderr)
        return 1

    file_size_mb = candidates_path.stat().st_size / 1_048_576
    print(f"[Input]  {args.candidates} ({file_size_mb:.1f} MB)")
    print(f"[Output] {args.out}")
    print(f"[Mode]   {'single-process' if args.no_mp else 'multiprocessing'}")
    print(f"[Target] top {args.top_k} candidates")
    if args.max:
        print(f"[Test]   processing only first {args.max:,} candidates")
    print()

    # ── Initialize ranker ─────────────────────────────────────────────────
    from src.ranker import CandidateRanker

    ranker = CandidateRanker(
        top_k=args.top_k,
        keep_buffer=max(args.buffer, args.top_k * 3),
        use_multiprocessing=not args.no_mp,
        show_progress=not args.quiet,
        max_candidates=args.max,
    )

    # ── Run ranking ───────────────────────────────────────────────────────
    try:
        results = ranker.rank_from_file(str(candidates_path))
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        return 1
    except Exception as exc:
        print(f"ERROR during ranking: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # ── Validate output ───────────────────────────────────────────────────
    errors = validate_output(results, args.top_k)
    if errors:
        print("\n⚠️  Validation warnings:")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"\n[OK] Output validation passed ({args.top_k} rows, ranks 1-{args.top_k}, monotonic scores)")

    # ── Save CSV ──────────────────────────────────────────────────────────
    ranker.save_csv(results, args.out)

    # ── Print summary ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    print(f"\n[Time] Total runtime: {total_time:.1f}s")

    if results:
        top3 = results[:3]
        print("\n[Top 3 candidates]")
        for r in top3:
            cid = r["candidate_id"]
            score = r["score"]
            reasoning = r["reasoning"][:80] + "..." if len(r["reasoning"]) > 80 else r["reasoning"]
            print(f"  #{r['rank']:3d} [{cid}] score={score:.4f} - {reasoning}")

    return 0


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
