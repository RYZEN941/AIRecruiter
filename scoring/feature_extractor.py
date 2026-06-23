"""
scoring/feature_extractor.py
-----------------------------
Extracts all scoring features from a structured candidate JSON record.

This is the central aggregation layer — it calls all sub-scorers and
returns a flat feature dict that the hybrid scorer can consume.

Also responsible for:
  - Building the text representation used for embedding
  - Normalising skill depths (proficiency × duration × endorsements)
  - Computing location fit
  - Extracting assessment scores from redrob_signals
"""

import math
import re
from typing import Dict, List

from scoring.jd_profile import (
    REQUIRED_SKILL_BUCKETS,
    PREFERRED_SKILL_BUCKETS,
    LOCATION_TIER_1,
    LOCATION_TIER_2,
    LOCATION_TIER_3_COUNTRY,
    EXPERIENCE_TARGET_MIN,
    EXPERIENCE_TARGET_MAX,
    EXPERIENCE_TARGET_SOFT_MIN,
    EXPERIENCE_TARGET_SOFT_MAX,
)
from scoring.title_classifier import score_title_fit, classify_title, get_title_label
from scoring.honeypot_detector import detect_honeypot
from scoring.availability_scorer import compute_availability_score
from scoring.disqualifier import compute_disqualifier_multiplier


# ─── Proficiency weights ──────────────────────────────────────────────────────
# Used to weight skill depth score
PROFICIENCY_WEIGHTS = {
    "beginner":     0.25,
    "intermediate": 0.55,
    "advanced":     0.80,
    "expert":       1.00,
}

# Endorsement saturation point — beyond this, marginal value is low
ENDORSEMENT_SAT = 50

# Duration saturation — 36+ months ≈ deep experience
DURATION_SAT_MONTHS = 36


# ─── Text representation builder ──────────────────────────────────────────────

def build_candidate_text(candidate: dict, max_chars: int = 2000) -> str:
    """
    Build a rich text representation of a candidate for semantic embedding.

    Combines: headline + summary + top career descriptions + skills.
    Capped at max_chars to avoid exceeding model token limits.

    The order matters — most important signals come first so that truncation
    cuts less important content.
    """
    profile = candidate.get("profile", {})
    parts = []

    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)

    # Career descriptions — most recent first (is_current=True first, then by start_date desc)
    career = sorted(
        candidate.get("career_history", []),
        key=lambda r: (r.get("is_current", False), r.get("start_date", "")),
        reverse=True,
    )
    for role in career[:4]:  # Top 4 roles to control length
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")
        if title or desc:
            parts.append(f"{title} at {company}: {desc}")

    # Skills — join top skills by proficiency
    skills_sorted = sorted(
        candidate.get("skills", []),
        key=lambda s: (
            PROFICIENCY_WEIGHTS.get(s.get("proficiency", "beginner"), 0),
            s.get("endorsements", 0),
        ),
        reverse=True,
    )
    top_skills = [s["name"] for s in skills_sorted[:20]]
    if top_skills:
        parts.append("Skills: " + ", ".join(top_skills))

    text = " | ".join(parts)
    return text[:max_chars]


# ─── Skill depth scorer ───────────────────────────────────────────────────────

def _skill_depth_score_for_bucket(
    bucket_keywords: List[str],
    candidate_skills: list,
) -> float:
    """
    Score how deeply a candidate covers a skill bucket.

    Returns a depth score in [0, 1] based on the best-matching skill found.
    Considers: proficiency level, months of duration, and endorsements.
    """
    best_score = 0.0
    bucket_set = {kw.lower() for kw in bucket_keywords}

    for skill in candidate_skills:
        name = skill.get("name", "").lower().strip()
        # Fuzzy match: check if any bucket keyword is contained in the skill name or vice versa
        matched = any(kw in name or name in kw for kw in bucket_set)
        if not matched:
            continue

        proficiency = PROFICIENCY_WEIGHTS.get(skill.get("proficiency", "beginner"), 0.25)
        duration = min(int(skill.get("duration_months", 0)), DURATION_SAT_MONTHS) / DURATION_SAT_MONTHS
        endorsements = min(int(skill.get("endorsements", 0)), ENDORSEMENT_SAT) / ENDORSEMENT_SAT

        # Weighted combination: proficiency is most important, then duration, then endorsements
        depth = 0.55 * proficiency + 0.30 * duration + 0.15 * endorsements
        best_score = max(best_score, depth)

    return round(best_score, 4)


def compute_skill_scores(candidate_skills: list) -> Dict[str, float]:
    """
    Compute depth scores for each required and preferred skill bucket.

    Returns a dict of {bucket_name: depth_score}.
    Also returns aggregate required_score and preferred_score.
    """
    scores = {}

    # Required buckets
    required_scores = []
    for bucket, keywords in REQUIRED_SKILL_BUCKETS.items():
        s = _skill_depth_score_for_bucket(keywords, candidate_skills)
        scores[f"req_{bucket}"] = s
        required_scores.append(s)

    # Preferred buckets
    preferred_scores = []
    for bucket, keywords in PREFERRED_SKILL_BUCKETS.items():
        s = _skill_depth_score_for_bucket(keywords, candidate_skills)
        scores[f"pref_{bucket}"] = s
        preferred_scores.append(s)

    # Aggregate: mean of required + small bonus from preferred
    scores["required_skill_score"] = (
        sum(required_scores) / len(required_scores) if required_scores else 0.0
    )
    scores["preferred_skill_bonus"] = (
        sum(preferred_scores) / (len(preferred_scores) * 2)  # half-weight
        if preferred_scores else 0.0
    )
    scores["skill_depth_score"] = min(
        scores["required_skill_score"] + scores["preferred_skill_bonus"], 1.0
    )

    return scores


# ─── Experience proximity scorer ──────────────────────────────────────────────

def compute_experience_score(years: float) -> float:
    """
    Gaussian falloff score for experience years vs JD target (5–9 years).

    In-range: 1.0
    Outside range: decays with Gaussian centered on midpoint of range.
    """
    lo, hi = EXPERIENCE_TARGET_MIN, EXPERIENCE_TARGET_MAX
    soft_lo, soft_hi = EXPERIENCE_TARGET_SOFT_MIN, EXPERIENCE_TARGET_SOFT_MAX

    if lo <= years <= hi:
        return 1.0
    if soft_lo <= years < lo or hi < years <= soft_hi:
        # Soft range — partial score
        distance = max(lo - years, years - hi)
        half_width = max((hi - lo) / 2.0, 1.0)
        return round(math.exp(-(distance**2) / (2 * half_width**2)), 4)
    # Outside soft range — near zero
    distance = max(soft_lo - years, years - soft_hi)
    return round(max(0.05, math.exp(-(distance**2) / 4.0)), 4)


# ─── Location scorer ──────────────────────────────────────────────────────────

def compute_location_score(profile: dict, redrob_signals: dict) -> float:
    """
    Score location fit.

    Tier 1 cities (Pune/Noida):       1.0
    Tier 2 cities + willing_to_relocate: 0.85
    Tier 2 cities, not relocating:    0.70
    India + willing_to_relocate:      0.60
    India, not relocating:            0.45
    Outside India + will_relocate:    0.30
    Outside India, no:                0.10
    """
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = bool(redrob_signals.get("willing_to_relocate", False))

    for tier1 in LOCATION_TIER_1:
        if tier1 in location:
            return 1.0

    for tier2 in LOCATION_TIER_2:
        if tier2 in location:
            return 0.85 if willing else 0.70

    if LOCATION_TIER_3_COUNTRY in country:
        return 0.60 if willing else 0.45

    return 0.30 if willing else 0.10


# ─── Assessment score extractor ───────────────────────────────────────────────

def extract_assessment_bonus(skill_assessment_scores: dict) -> float:
    """
    Extract a bonus from platform-verified assessment scores.

    Platform assessments are more reliable than self-reported proficiency.
    Returns a bonus in [0, 0.10] that supplements the skill_depth_score.
    """
    if not skill_assessment_scores:
        return 0.0

    # Focus on ML-relevant assessments
    ml_keywords = [
        "python", "nlp", "machine learning", "deep learning", "sql",
        "tensorflow", "pytorch", "transformers", "data science",
    ]

    relevant_scores = []
    for skill_name, score in skill_assessment_scores.items():
        name_lower = skill_name.lower()
        if any(kw in name_lower for kw in ml_keywords):
            relevant_scores.append(float(score) / 100.0)

    if not relevant_scores:
        return 0.0

    avg = sum(relevant_scores) / len(relevant_scores)
    return round(avg * 0.10, 4)  # Scale to max 0.10 bonus


# ─── Master feature extractor ─────────────────────────────────────────────────

def extract_features(candidate: dict) -> dict:
    """
    Extract all scoring features from a structured candidate record.

    Parameters
    ----------
    candidate : full candidate dict from schema

    Returns
    -------
    Comprehensive feature dict consumed by hybrid_scorer.
    """
    cid = candidate.get("candidate_id", "UNKNOWN")
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # ── Basic profile fields ──────────────────────────────────────────────────
    name = profile.get("anonymized_name", "Unknown")
    yoe = float(profile.get("years_of_experience", 0) or 0)
    current_title = profile.get("current_title", "")
    current_industry = profile.get("current_industry", "")

    # ── Title fit ─────────────────────────────────────────────────────────────
    title_tier = classify_title(current_title)
    title_score = score_title_fit(current_title, career)

    # ── Skill depth ───────────────────────────────────────────────────────────
    skill_scores = compute_skill_scores(skills)

    # ── Assessment bonus ──────────────────────────────────────────────────────
    assessment_bonus = extract_assessment_bonus(
        signals.get("skill_assessment_scores", {})
    )

    # ── Experience ───────────────────────────────────────────────────────────
    experience_score = compute_experience_score(yoe)

    # ── Location ─────────────────────────────────────────────────────────────
    location_score = compute_location_score(profile, signals)

    # ── Availability ─────────────────────────────────────────────────────────
    availability = compute_availability_score(signals)

    # ── Disqualifiers ────────────────────────────────────────────────────────
    disq = compute_disqualifier_multiplier(candidate)

    # ── Honeypot detection ───────────────────────────────────────────────────
    honeypot = detect_honeypot(candidate)

    # ── GitHub activity ───────────────────────────────────────────────────────
    github_score_raw = float(signals.get("github_activity_score", -1))
    github_score = max(0.0, github_score_raw) / 100.0  # -1 → 0, else 0–1

    # ── Embed text ───────────────────────────────────────────────────────────
    embed_text = build_candidate_text(candidate)

    return {
        "candidate_id": cid,
        "name": name,
        "current_title": current_title,
        "current_industry": current_industry,
        "years_of_experience": yoe,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),

        # Title
        "title_tier": title_tier,
        "title_label": get_title_label(title_tier),
        "title_score": title_score,

        # Skills
        "skill_depth_score": skill_scores["skill_depth_score"],
        "required_skill_score": skill_scores["required_skill_score"],
        "preferred_skill_bonus": skill_scores["preferred_skill_bonus"],
        "assessment_bonus": assessment_bonus,
        "skill_bucket_scores": {
            k: v for k, v in skill_scores.items()
            if k.startswith("req_") or k.startswith("pref_")
        },

        # Experience
        "experience_score": experience_score,

        # Availability
        "availability_score": availability["availability_score"],
        "availability_components": availability["components"],
        "days_since_active": availability["days_since_active"],
        "notice_period_days": availability["notice_period_days"],
        "open_to_work": availability["open_to_work"],
        "recruiter_response_rate": availability["recruiter_response_rate"],

        # Location
        "location_score": location_score,

        # Disqualifiers
        "disqualifier_multiplier": disq["multiplier"],
        "disqualifier_flags": disq["flags"],
        "is_disqualified": disq["is_disqualified"],

        # Honeypot
        "honeypot_score": honeypot["honeypot_score"],
        "honeypot_flags": honeypot["flags"],
        "is_likely_honeypot": honeypot["is_likely_honeypot"],

        # GitHub
        "github_score": github_score,

        # Signals passthrough (for reasoning generation)
        "profile_completeness": float(signals.get("profile_completeness_score", 0)),
        "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d", 0)),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "expected_salary_lpa": signals.get("expected_salary_range_inr_lpa", {}),
        "preferred_work_mode": signals.get("preferred_work_mode", ""),

        # Embed text
        "embed_text": embed_text,
    }
