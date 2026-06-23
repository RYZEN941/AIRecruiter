"""
scoring/disqualifier.py
-----------------------
Applies hard negative multipliers from the JD's explicit disqualifiers.

These are multiplicative penalties (not additive) applied after the weighted
sum score so that genuinely disqualified candidates cannot sneak through on
strong semantic similarity alone.

Disqualifiers from the JD:
  1. Consulting-firm-only career  (TCS, Infosys, Wipro, Accenture, etc.)
  2. Computer vision / speech / robotics only — no NLP/IR
  3. No production deployment — pure academic/research background
  4. Very recent LLM-only experience (<12 months) with no pre-LLM ML history
  5. Title-chasing pattern (every 1.5-2 years, upward company hop, no depth)

Each disqualifier returns a multiplier in (0, 1].
A multiplier of 1.0 means no penalty.
Multiple disqualifiers compound multiplicatively.
"""

import re
from typing import List

from scoring.jd_profile import (
    CONSULTING_FIRM_KEYWORDS,
    CV_ONLY_SKILL_SIGNALS,
    SPEECH_ONLY_SKILL_SIGNALS,
)


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()


def _any_keyword_match(text: str, keywords: List[str]) -> bool:
    t = _normalise(text)
    return any(kw in t for kw in keywords)


def _skill_names(skills: list) -> List[str]:
    return [_normalise(s.get("name", "")) for s in skills]


def _all_companies_are_consulting(career_history: list) -> bool:
    """
    Return True if EVERY company in the candidate's career history is a
    well-known consulting / IT-services firm.

    We require all roles (not just current) to be consulting-only.
    A single product-company stint breaks this flag.
    """
    if not career_history:
        return False
    for role in career_history:
        company = _normalise(role.get("company", ""))
        is_consulting = any(kw in company for kw in CONSULTING_FIRM_KEYWORDS)
        if not is_consulting:
            return False  # Found a non-consulting company — not disqualified
    return True  # Every company matched consulting keywords


def _has_nlp_ir_signal(skills: list, career_history: list) -> bool:
    """
    Return True if the candidate has meaningful NLP / IR signals in either
    their skills or career descriptions.
    """
    nlp_keywords = [
        "nlp", "natural language", "text classification", "information retrieval",
        "search", "ranking", "recommendation", "transformer", "bert", "llm",
        "embeddings", "rag", "faiss", "vector", "language model",
    ]
    for skill_name in _skill_names(skills):
        if any(kw in skill_name for kw in nlp_keywords):
            return True
    for role in career_history:
        desc = _normalise(role.get("description", ""))
        if any(kw in desc for kw in nlp_keywords):
            return True
    return False


def _is_cv_or_speech_only(skills: list, career_history: list) -> bool:
    """
    Return True if candidate's primary signals are computer vision or speech
    WITHOUT any NLP/IR context.
    """
    skill_names = _skill_names(skills)

    cv_signals = sum(1 for s in skill_names if any(kw in s for kw in CV_ONLY_SKILL_SIGNALS))
    speech_signals = sum(1 for s in skill_names if any(kw in s for kw in SPEECH_ONLY_SKILL_SIGNALS))

    if (cv_signals + speech_signals) < 2:
        return False  # Not even primarily CV/speech

    return not _has_nlp_ir_signal(skills, career_history)


def _is_pure_research_no_production(career_history: list, profile_summary: str) -> bool:
    """
    Detect candidates whose entire career is in academic/research environments
    with no production deployment evidence.

    Signals for pure research:
    - All companies include "university", "lab", "research institute", "iit", etc.
    - No mention of production, deployment, users, scale in any description
    - Only "research scientist" / "research associate" titles
    """
    production_keywords = [
        "production", "deployed", "users", "scale", "millions",
        "latency", "serving", "api", "real-time", "real time",
        "customer", "client", "product", "shipped", "launched",
    ]
    research_keywords = [
        "university", "iit", "iisc", "nit", "college", "lab",
        "research institute", "phd", "doctoral", "postdoc",
    ]

    # Check if ANY description mentions production signals
    for role in career_history:
        desc = _normalise(role.get("description", ""))
        if any(kw in desc for kw in production_keywords):
            return False  # Has production experience

    # Check if all companies look like academic institutions
    all_academic = all(
        any(kw in _normalise(r.get("company", "")) for kw in research_keywords)
        for r in career_history
    ) if career_history else False

    return all_academic


def _is_title_chaser(career_history: list) -> bool:
    """
    Detect rapid company-hopping with title escalation and no depth.

    Pattern: 3+ jobs each <20 months, with upward title movement at each hop.
    This matches the JD's "optimizing for Senior → Staff → Principal titles
    by switching companies every 1.5 years" warning.
    """
    if len(career_history) < 3:
        return False

    short_stints = [
        r for r in career_history
        if int(r.get("duration_months", 999)) < 20
    ]
    if len(short_stints) < 3:
        return False

    # Check if titles escalate across the short stints
    seniority_keywords = ["junior", "mid", "senior", "lead", "staff", "principal", "director"]
    titles = [r.get("title", "").lower() for r in career_history]
    seniority_levels = []
    for t in titles:
        for i, kw in enumerate(seniority_keywords):
            if kw in t:
                seniority_levels.append(i)
                break

    if len(seniority_levels) >= 3:
        # Check for strictly increasing seniority across short stints
        # (going from staff back to junior would break the pattern)
        diffs = [seniority_levels[i+1] - seniority_levels[i]
                 for i in range(len(seniority_levels)-1)]
        if all(d >= 0 for d in diffs) and sum(d > 0 for d in diffs) >= 2:
            return True

    return False


# ─── Main function ────────────────────────────────────────────────────────────

def compute_disqualifier_multiplier(candidate: dict) -> dict:
    """
    Compute a compound penalty multiplier for a candidate.

    Parameters
    ----------
    candidate : full candidate dict

    Returns
    -------
    dict with:
        multiplier     : float (0, 1] — multiply final score by this
        flags          : list of triggered disqualifier descriptions
        is_disqualified: bool — True if multiplier < 0.5 (effectively excluded)
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])
    summary = _normalise(profile.get("summary", ""))

    flags = []
    multiplier = 1.0

    # ── Disqualifier 1: Consulting-only career ────────────────────────────────
    if _all_companies_are_consulting(career_history):
        flags.append("Entire career at consulting/IT-services firms (TCS/Wipro/Infosys etc.)")
        multiplier *= 0.30   # Strong penalty — JD is explicit about this

    # ── Disqualifier 2: CV/Speech only without NLP/IR ────────────────────────
    if _is_cv_or_speech_only(skills, career_history):
        flags.append("Primary domain is CV/Speech without NLP/IR exposure")
        multiplier *= 0.45

    # ── Disqualifier 3: Pure research, no production ──────────────────────────
    if _is_pure_research_no_production(career_history, summary):
        flags.append("Career appears to be pure academic/research — no production deployment evidence")
        multiplier *= 0.40

    # ── Disqualifier 4: Title-chaser pattern ──────────────────────────────────
    if _is_title_chaser(career_history):
        flags.append("Title-chasing pattern detected (rapid company hops with escalating titles)")
        multiplier *= 0.70   # Moderate penalty — could still have skills

    return {
        "multiplier": round(multiplier, 4),
        "flags": flags,
        "is_disqualified": multiplier < 0.50,
    }
