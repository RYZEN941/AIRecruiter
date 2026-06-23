"""
scoring/hybrid_scorer.py
------------------------
Complete rewrite for the Redrob hackathon.

Takes pre-extracted feature dicts + FAISS semantic scores and produces a
final combined score for each candidate.

Signal stack (weights from jd_profile.SIGNAL_WEIGHTS):
  semantic     0.25  — FAISS cosine similarity vs JD embedding
  title_fit    0.20  — current title / career trajectory tier
  skill_depth  0.20  — depth-weighted skill match (proficiency+duration+endorsements)
  experience   0.10  — years proximity to 5–9 range
  availability 0.15  — behavioral signals composite
  location     0.05  — location match
  llm_bonus    0.05  — optional cached LLM pre-score (redistributed if missing)

After weighted sum:
  × disqualifier_multiplier   (multiplicative hard penalties)
  × honeypot_penalty          (1.0 if clean, near-0 if likely honeypot)

Output: list of score dicts sorted by final_score descending.
"""

from typing import Dict, List, Optional, Tuple

from scoring.jd_profile import SIGNAL_WEIGHTS


def _honeypot_penalty(honeypot_score: float) -> float:
    """
    Convert honeypot score to a multiplicative penalty.
    honeypot_score >= 0.6 → hard near-zero (0.05)
    honeypot_score < 0.6  → mild penalty proportional to score
    """
    if honeypot_score >= 0.6:
        return 0.05   # Effectively excluded from top-100
    elif honeypot_score > 0.0:
        return 1.0 - (honeypot_score * 0.5)
    return 1.0


def score_candidate(
    features: dict,
    semantic_score: float,
    llm_cached_score: Optional[float] = None,
) -> dict:
    """
    Compute the final score for a single candidate.

    Parameters
    ----------
    features         : output of feature_extractor.extract_features()
    semantic_score   : cosine similarity from FAISS (0–1)
    llm_cached_score : optional pre-computed LLM score (0–1), None if unavailable

    Returns
    -------
    dict with all signal components + final_score
    """
    w = SIGNAL_WEIGHTS.copy()

    # If LLM score not available, redistribute its weight to skill_depth
    if llm_cached_score is None:
        llm_score = 0.0
        w["skill_depth"] += w["llm_bonus"]
        w["llm_bonus"] = 0.0
    else:
        llm_score = float(llm_cached_score)

    # ── Weighted sum of signals ───────────────────────────────────────────────
    raw_score = (
        w["semantic"]     * semantic_score                    +
        w["title_fit"]    * features["title_score"]           +
        w["skill_depth"]  * (
            features["skill_depth_score"] +
            features.get("assessment_bonus", 0.0)             # small bonus on top
        )                                                      +
        w["experience"]   * features["experience_score"]      +
        w["availability"] * features["availability_score"]    +
        w["location"]     * features["location_score"]        +
        w["llm_bonus"]    * llm_score
    )
    # Clamp — assessment_bonus can push above 1.0 slightly
    raw_score = min(raw_score, 1.0)

    # ── Multiplicative penalties ──────────────────────────────────────────────
    disq_mult = features.get("disqualifier_multiplier", 1.0)
    hp_mult = _honeypot_penalty(features.get("honeypot_score", 0.0))

    final_score = raw_score * disq_mult * hp_mult
    final_score = round(max(0.0, min(1.0, final_score)), 6)

    return {
        "candidate_id":            features["candidate_id"],
        "name":                    features["name"],
        "final_score":             final_score,
        "raw_score":               round(raw_score, 6),

        # Individual signal contributions (for reporting/debugging)
        "semantic_score":          round(semantic_score, 4),
        "title_score":             round(features["title_score"], 4),
        "skill_depth_score":       round(features["skill_depth_score"], 4),
        "experience_score":        round(features["experience_score"], 4),
        "availability_score":      round(features["availability_score"], 4),
        "location_score":          round(features["location_score"], 4),
        "llm_score":               round(llm_score, 4),
        "disqualifier_multiplier": round(disq_mult, 4),
        "honeypot_penalty":        round(hp_mult, 4),

        # Pass through features needed by output/report_generator
        "current_title":           features["current_title"],
        "years_of_experience":     features["years_of_experience"],
        "title_label":             features.get("title_label", ""),
        "title_tier":              features.get("title_tier", 0),
        "location":                features.get("location", ""),
        "days_since_active":       features.get("days_since_active", 999),
        "notice_period_days":      features.get("notice_period_days", 0),
        "open_to_work":            features.get("open_to_work", False),
        "recruiter_response_rate": features.get("recruiter_response_rate", 0),
        "disqualifier_flags":      features.get("disqualifier_flags", []),
        "honeypot_flags":          features.get("honeypot_flags", []),
        "is_likely_honeypot":      features.get("is_likely_honeypot", False),
        "is_disqualified":         features.get("is_disqualified", False),
        "github_score":            features.get("github_score", 0),
        "skill_bucket_scores":     features.get("skill_bucket_scores", {}),
        "willing_to_relocate":     features.get("willing_to_relocate", False),
        "preferred_work_mode":     features.get("preferred_work_mode", ""),
    }


def rank_candidates(
    features_map: Dict[str, dict],
    faiss_results: List[Tuple[str, float]],
    llm_cache: Optional[Dict[str, float]] = None,
    top_n: int = 100,
) -> List[dict]:
    """
    Score all FAISS-retrieved candidates and return the top-N.

    Parameters
    ----------
    features_map  : {candidate_id: features_dict}
    faiss_results : [(candidate_id, cosine_sim)] from FAISS retrieval
    llm_cache     : optional {candidate_id: llm_score_0_to_1}
    top_n         : number to return (default 100 for submission)

    Returns
    -------
    List of score dicts sorted by final_score descending.
    """
    scored = []

    for candidate_id, sem_score in faiss_results:
        if candidate_id not in features_map:
            continue  # Safety guard

        features = features_map[candidate_id]

        # Skip obvious honeypots early to save time
        if features.get("is_likely_honeypot", False):
            # Still score them (so we can exclude properly) but don't skip
            pass

        llm_score = None
        if llm_cache and candidate_id in llm_cache:
            llm_score = llm_cache[candidate_id]

        score_dict = score_candidate(
            features=features,
            semantic_score=sem_score,
            llm_cached_score=llm_score,
        )
        scored.append(score_dict)

    # Sort by final_score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    return scored[:top_n]
