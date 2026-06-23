"""
output/report_generator.py
--------------------------
Rewritten for the Redrob hackathon submission format.

Output: submission CSV with columns: candidate_id, rank, score, reasoning
  - Exactly 100 rows (ranks 1–100)
  - score monotonically non-increasing
  - reasoning: 1-2 sentence, specific facts, no hallucination, varies per candidate

Reasoning generation is deterministic from the scored features — no LLM needed
at this stage. Each reasoning string cites specific facts from the candidate
profile so it passes Stage 4 manual review checks.
"""

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

OUTPUT_DIR = Path(__file__).parent
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


def _days_label(days: int) -> str:
    """Human-readable recency label."""
    if days < 7:
        return "active this week"
    elif days < 30:
        return "active this month"
    elif days < 90:
        return f"last active {days} days ago"
    elif days < 180:
        return f"last active ~{days//30} months ago"
    else:
        return f"last active {days//30} months ago"


def _notice_label(days: int) -> str:
    if days == 0:
        return "immediate joiner"
    elif days <= 15:
        return f"{days}-day notice"
    elif days <= 30:
        return f"{days}-day notice (ideal)"
    elif days <= 60:
        return f"{days}-day notice"
    else:
        return f"{days}-day notice (long)"


def generate_reasoning(score_dict: dict, candidate: Optional[dict] = None) -> str:
    """
    Generate a 1-2 sentence reasoning string for Stage 4 review.

    Rules:
      - Must cite specific facts (title, years, skills, signals)
      - Must acknowledge concerns where they exist
      - Must match tone to rank (top candidates → positive; bottom → honest gaps)
      - No templating — each reasoning must differ substantively

    Parameters
    ----------
    score_dict : output from hybrid_scorer.score_candidate()
    candidate  : optional full candidate dict for richer detail

    Returns
    -------
    str — 1-2 sentences
    """
    name = score_dict.get("name", "Candidate")
    title = score_dict.get("current_title", "Unknown")
    yoe = score_dict.get("years_of_experience", 0)
    title_tier = score_dict.get("title_tier", 0)
    skill_score = score_dict.get("skill_depth_score", 0)
    avail = score_dict.get("availability_score", 0)
    sem = score_dict.get("semantic_score", 0)
    days_active = score_dict.get("days_since_active", 999)
    notice = score_dict.get("notice_period_days", 60)
    open_to_work = score_dict.get("open_to_work", False)
    response_rate = score_dict.get("recruiter_response_rate", 0.5)
    disq_flags = score_dict.get("disqualifier_flags", [])
    honeypot_flags = score_dict.get("honeypot_flags", [])
    final_score = score_dict.get("final_score", 0)
    github = score_dict.get("github_score", 0)
    willing_relocate = score_dict.get("willing_to_relocate", False)
    location = score_dict.get("location", "")

    # Build a targeted skill mention from bucket scores
    skill_buckets = score_dict.get("skill_bucket_scores", {})
    strong_buckets = [
        k.replace("req_", "").replace("pref_", "").replace("_", " ")
        for k, v in skill_buckets.items()
        if v >= 0.60
    ]

    parts = []

    # ── Sentence 1: Core qualification statement ──────────────────────────────
    if title_tier >= 3 and skill_score >= 0.50:
        skill_mention = (
            f" with hands-on {strong_buckets[0]}" if strong_buckets else ""
        )
        parts.append(
            f"{title} with {yoe:.1f}y experience{skill_mention}; "
            f"semantic alignment with JD is {sem:.0%}."
        )
    elif title_tier >= 2:
        parts.append(
            f"{title} ({yoe:.1f}y) in adjacent technical domain; "
            f"skills partially overlap with retrieval/ML requirements."
        )
    else:
        parts.append(
            f"{title} ({yoe:.1f}y) — title is outside the ML/AI domain; "
            f"included based on semantic and skill signal only."
        )

    # ── Sentence 2: Availability / concern / strength ────────────────────────
    concern_parts = []
    strength_parts = []

    if disq_flags:
        concern_parts.append(disq_flags[0].lower()[:80])
    if honeypot_flags:
        concern_parts.append(f"profile anomaly detected: {honeypot_flags[0][:60]}")
    if days_active > 180:
        concern_parts.append(f"not active on platform for {days_active//30} months")
    if response_rate < 0.20:
        concern_parts.append(f"low recruiter response rate ({response_rate:.0%})")
    if notice > 90:
        concern_parts.append(f"{notice}-day notice period")

    if github >= 0.50:
        strength_parts.append(f"strong GitHub activity (score {github*100:.0f}/100)")
    if open_to_work:
        strength_parts.append("marked open-to-work")
    if willing_relocate and location:
        strength_parts.append(f"willing to relocate from {location}")
    if notice < 30 and final_score >= 0.5:
        strength_parts.append(f"{_notice_label(notice)}")

    if concern_parts:
        parts.append(
            "Concern: " + "; ".join(concern_parts[:2]) + "."
        )
    elif strength_parts:
        parts.append(
            "; ".join(strength_parts[:2]).capitalize() + "."
        )
    else:
        avail_label = (
            "strong availability signals"
            if avail >= 0.7 else
            "moderate availability" if avail >= 0.5 else
            "weak availability signals"
        )
        parts.append(f"{avail_label.capitalize()} ({_days_label(days_active)}).")

    return " ".join(parts)


def write_submission_csv(
    ranked_candidates: List[dict],
    output_path: Optional[str] = None,
    candidates_map: Optional[Dict[str, dict]] = None,
) -> str:
    """
    Write the submission CSV from the ranked candidate list.

    Parameters
    ----------
    ranked_candidates : list of score dicts sorted by final_score descending
    output_path       : override default path (optional)
    candidates_map    : optional {candidate_id: full_candidate_dict} for richer reasoning

    Returns
    -------
    str — absolute path to the written CSV file
    """
    out_path = Path(output_path) if output_path else SUBMISSION_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Take exactly top 100
    top_100 = ranked_candidates[:100]

    # Ensure scores are monotonically non-increasing (required by validator)
    for i in range(1, len(top_100)):
        if top_100[i]["final_score"] > top_100[i-1]["final_score"]:
            top_100[i]["final_score"] = top_100[i-1]["final_score"]

    rows = []
    for rank, score_dict in enumerate(top_100, 1):
        cid = score_dict["candidate_id"]
        score = score_dict["final_score"]
        full_candidate = candidates_map.get(cid) if candidates_map else None
        reasoning = generate_reasoning(score_dict, full_candidate)

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        f"{score:.6f}",
            "reasoning":    reasoning,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  [Output] Submission CSV written → {out_path}")
    print(f"  [Output] {len(rows)} candidates, ranks 1–{len(rows)}")
    print(f"  [Output] Score range: [{rows[-1]['score']} – {rows[0]['score']}]")

    return str(out_path)


def print_summary_table(ranked_candidates: List[dict], top_n: int = 20) -> None:
    """Print a readable summary table to the console."""
    print("\n" + "=" * 90)
    print("  TOP CANDIDATES")
    print("=" * 90)
    header = f"  {'Rank':<5} {'ID':<14} {'Name':<22} {'Title':<28} {'Score':>7} {'Avail':>6}"
    print(header)
    print("  " + "-" * 86)

    for rank, r in enumerate(ranked_candidates[:top_n], 1):
        score = r["final_score"] * 100
        avail = r.get("availability_score", 0) * 100
        disq = "[!] " if r.get("is_disqualified") else ""
        hp = "[HONEYPOT]" if r.get("is_likely_honeypot") else ""
        print(
            f"  #{rank:<4} {r['candidate_id']:<14} "
            f"{r['name'][:20]:<22} "
            f"{disq}{r['current_title'][:26]:<28} "
            f"{score:>6.1f}% "
            f"{avail:>5.1f}% "
            f"{hp}"
        )
    print("=" * 90)
