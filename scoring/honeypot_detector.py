"""
scoring/honeypot_detector.py
----------------------------
Detects candidates with impossible or highly suspicious profiles.

The hackathon dataset contains ~80 honeypot candidates with subtly
impossible profiles (per submission_spec.md Section 7). Ranking >10%
honeypots in the top-100 = Stage 3 disqualification.

Detection heuristics:
  1. Expert proficiency + 0 duration_months on the same skill
  2. Suspiciously high density of "expert" skills (10+ experts)
  3. years_of_experience inconsistent with career history sum
  4. Education graduation year is after their first career start
  5. Self-reported experience far exceeds career history total

Returns a honeypot_score in [0, 1]:
  0.0 = clean profile
  >0.7 = very likely honeypot → hard-penalise in hybrid scorer
"""

from datetime import date, datetime
from typing import List


def _parse_year(date_str: str | None) -> int | None:
    """Extract year from a date string 'YYYY-MM-DD', or return None."""
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


def _career_total_months(career_history: list) -> int:
    """Sum duration_months across all career entries."""
    return sum(int(r.get("duration_months", 0)) for r in career_history)


def _earliest_career_year(career_history: list) -> int | None:
    """Find the earliest start_year from career history."""
    years = []
    for r in career_history:
        y = _parse_year(r.get("start_date"))
        if y:
            years.append(y)
    return min(years) if years else None


def _education_end_year(education: list) -> int | None:
    """Find the latest graduation year in education records."""
    years = [e.get("end_year") for e in education if e.get("end_year")]
    return max(years) if years else None


def detect_honeypot(candidate: dict) -> dict:
    """
    Analyse a candidate record for impossible profile patterns.

    Parameters
    ----------
    candidate : full candidate dict from schema

    Returns
    -------
    dict with:
        honeypot_score : float 0–1 (higher = more suspicious)
        flags          : list of string descriptions of triggered rules
    """
    flags = []
    score_components = []

    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])

    yoe = float(profile.get("years_of_experience", 0) or 0)

    # ── Rule 1: Expert proficiency + 0 duration months ───────────────────────
    expert_zero_duration = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and int(s.get("duration_months", 0)) == 0
    ]
    if expert_zero_duration:
        flags.append(
            f"Expert proficiency with 0 months duration: {', '.join(expert_zero_duration[:3])}"
        )
        # Weight by count — 1 is suspicious, 3+ is very likely honeypot
        score_components.append(min(len(expert_zero_duration) / 3.0, 1.0))

    # ── Rule 2: Implausible expert skill density ──────────────────────────────
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8:
        flags.append(f"Implausibly high expert skill count: {expert_count}")
        score_components.append(min((expert_count - 7) / 5.0, 1.0))
    elif expert_count >= 5:
        # Just a mild flag — many real candidates can have 5 experts
        score_components.append(0.1)

    # ── Rule 3: years_of_experience inconsistent with career history ──────────
    career_months_total = _career_total_months(career_history)
    career_years_total = career_months_total / 12.0

    # Allow for a generous 2-year gap (studies, breaks, freelance not listed)
    if yoe > 0 and career_years_total > 0:
        gap = yoe - career_years_total
        if gap > 5.0:
            flags.append(
                f"years_of_experience ({yoe:.1f}) exceeds career history total "
                f"({career_years_total:.1f}y) by {gap:.1f}y"
            )
            score_components.append(min(gap / 8.0, 1.0))

    # ── Rule 4: Career started before graduation ──────────────────────────────
    earliest_work = _earliest_career_year(career_history)
    grad_year = _education_end_year(education)
    if earliest_work and grad_year and earliest_work < grad_year - 2:
        # Allow 2-year window for legitimate cases (dropped out, part-time, etc.)
        flags.append(
            f"Career started ({earliest_work}) before graduation ({grad_year})"
        )
        score_components.append(0.5)

    # ── Rule 5: Claimed total experience exceeds possible working life ─────────
    current_year = date.today().year
    if earliest_work:
        max_possible_years = current_year - earliest_work
        if yoe > max_possible_years + 2:  # 2-year tolerance
            flags.append(
                f"Claimed {yoe}y experience but earliest role starts {earliest_work} "
                f"(max possible: {max_possible_years}y)"
            )
            score_components.append(1.0)

    # ── Rule 6: Advanced/expert skill with 0 endorsements AND 0 duration ──────
    # Not definitive on its own, but combined with other flags it's suspicious
    advanced_empty = [
        s["name"] for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and int(s.get("duration_months", 0)) == 0
        and int(s.get("endorsements", 0)) == 0
    ]
    if len(advanced_empty) >= 5:
        flags.append(f"{len(advanced_empty)} advanced/expert skills with 0 duration AND 0 endorsements")
        score_components.append(min(len(advanced_empty) / 8.0, 0.7))

    # ── Aggregate score ───────────────────────────────────────────────────────
    if not score_components:
        honeypot_score = 0.0
    elif len(score_components) == 1:
        honeypot_score = score_components[0] * 0.5  # single rule = lower confidence
    else:
        # Multiple rules fired = higher confidence. Cap at 1.0.
        honeypot_score = min(sum(score_components) / len(score_components) * 1.5, 1.0)

    return {
        "honeypot_score": round(honeypot_score, 4),
        "flags": flags,
        "is_likely_honeypot": honeypot_score >= 0.6,
    }
