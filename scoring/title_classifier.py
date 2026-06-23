"""
scoring/title_classifier.py
---------------------------
Maps a candidate's current_title (and career trajectory) to a fit-tier
for the Senior AI Engineer role.

Tiers:
  0 = Unrelated    (Marketing, HR, Accountant, Civil Eng, etc.)
  1 = Distant      (General SWE, DevOps, Frontend, QA)
  2 = Adjacent     (Data Engineer, Backend with ML, Data Scientist generic)
  3 = Relevant     (ML Engineer, AI Engineer, Data Scientist applied ML)
  4 = Strong Match (Senior ML/AI at product company, NLP/Search/Ranking)

Tier score = tier / 4  → float in [0, 1]

Career trajectory bonus:
  If current title is tier 2+ AND previous titles show upward ML progression
  → small bonus applied in feature_extractor.

This is rule-based on purpose — we want deterministic, reproducible,
explainable classification without LLM calls during ranking.
"""

import re
from typing import List

# ─── Keyword → tier mappings (ordered, higher tier = checked first) ───────────

TIER_4_PATTERNS = [
    # Exact senior AI/ML/NLP/Search/Ranking roles
    r"senior.*(?:machine learning|ml|ai|nlp|search|ranking|retrieval|recommend)",
    r"(?:machine learning|ml|ai|nlp|search|ranking|retrieval).*engineer.*(?:senior|lead|staff|principal)",
    r"staff.*(?:machine learning|ml|ai|nlp)",
    r"principal.*(?:machine learning|ml|ai|nlp)",
    r"lead.*(?:machine learning|ml|ai|nlp)",
    r"(?:nlp|nlu|information retrieval|ir|ranking|search|recommendation)\s*(?:engineer|scientist|researcher)",
    r"applied.*(?:ml|ai|machine learning).*engineer",
    r"(?:founding|early).*(engineer|scientist).*ai",
]

TIER_3_PATTERNS = [
    # Core ML/AI roles without strong seniority signal
    r"machine learning engineer",
    r"\bml engineer\b",
    r"ai engineer",
    r"(?:applied|research)\s*(?:ml|ai|machine learning|scientist)",
    r"data scientist.*(?:ml|machine learning|nlp|deep learning)",
    r"(?:deep learning|neural network)\s*engineer",
    r"(?:computer vision|cv)\s*engineer",  # CV is tier 3 — possible but weak
    r"junior.*(?:machine learning|ml|ai)\s*engineer",
    # Recommendation / search / ranking engineers (handles compound titles)
    r"recommendation\s*(?:systems?\s*)?engineer",
    r"recommender\s*(?:systems?\s*)?(?:engineer|scientist)",
    r"(?:search|ranking|retrieval)\s*(?:systems?\s*)?engineer",
    r"(?:search|ranking|retrieval)\s*engineer",
    r"llm\s*engineer",
    r"(?:nlp|nlu)\s*engineer",
]

TIER_2_PATTERNS = [
    # Adjacent technical roles
    r"data engineer",
    r"analytics engineer",
    r"(?:senior|lead|staff)\s*(?:software|backend|fullstack|full.stack)\s*engineer",
    r"backend.*engineer",
    r"(?:platform|infrastructure)\s*engineer",
    r"mlops\s*engineer",
    r"ml\s*(?:platform|infra)",
    r"data scientist",   # Without ML qualifier — generic DS
    r"(?:bi|business intelligence)\s*engineer",
    r"research\s*engineer",   # Could be ML, could be hardware
    r"research\s*scientist",
    r"cloud\s*(?:architect|engineer)",
    r"solutions\s*architect",
]

TIER_1_PATTERNS = [
    # General SWE, web, mobile, devops — not ML
    r"software\s*engineer",
    r"full.?stack\s*(?:developer|engineer)",
    r"frontend\s*(?:developer|engineer)",
    r"(?:ios|android|mobile)\s*(?:developer|engineer)",
    r"devops\s*engineer",
    r"(?:site\s*reliability|sre)\s*engineer",
    r"qa\s*(?:engineer|analyst)",
    r"test\s*(?:engineer|automation)",
    r"\.net\s*developer",
    r"java\s*developer",
    r"php\s*developer",
    r"(?:security|cyber)\s*engineer",
    r"network\s*engineer",
    r"database\s*administrator",
    r"dba",
]

TIER_0_PATTERNS = [
    # Clearly unrelated
    r"marketing\s*(?:manager|executive|lead)",
    r"hr\s*(?:manager|executive|specialist)",
    r"(?:human\s*resources|talent\s*acquisition)",
    r"accountant",
    r"(?:finance|financial)\s*(?:analyst|manager)",
    r"sales\s*(?:manager|executive|representative)",
    r"operations\s*manager",
    r"project\s*manager",
    r"product\s*manager",
    r"business\s*analyst",
    r"(?:civil|mechanical|electrical|chemical)\s*engineer",
    r"content\s*(?:writer|creator|manager)",
    r"graphic\s*(?:designer|artist)",
    r"ui\s*/?\s*ux",
    r"(?:customer\s*support|customer\s*success)",
    r"(?:supply\s*chain|logistics|procurement)",
]


def _match_tier(title: str, patterns: List[str]) -> bool:
    """Return True if title matches any pattern in the list."""
    t = title.lower().strip()
    for pat in patterns:
        if re.search(pat, t):
            return True
    return False


def classify_title(title: str) -> int:
    """
    Classify a job title into a fit tier (0–4).

    Parameters
    ----------
    title : str  — e.g. "Senior ML Engineer", "Marketing Manager"

    Returns
    -------
    int in [0, 4]
    """
    if not title:
        return 1  # Unknown → treat as distant

    if _match_tier(title, TIER_4_PATTERNS):
        return 4
    if _match_tier(title, TIER_3_PATTERNS):
        return 3
    if _match_tier(title, TIER_2_PATTERNS):
        return 2
    if _match_tier(title, TIER_1_PATTERNS):
        return 1
    return 0


def score_title_fit(current_title: str, career_history: list) -> float:
    """
    Compute a title fit score in [0, 1] that considers both the current
    title and the trajectory of past titles.

    Trajectory bonus: if past titles show ML progression, boost by up to 0.1.

    Parameters
    ----------
    current_title   : str
    career_history  : list of career dicts (from schema)

    Returns
    -------
    float in [0, 1]
    """
    current_tier = classify_title(current_title)
    base_score = current_tier / 4.0

    # Check trajectory — look at all historical titles
    all_tiers = [current_tier]
    for role in career_history:
        t = role.get("title", "")
        all_tiers.append(classify_title(t))

    # If there's any tier-3+ in history even if not current → small bonus
    # (Shows they had ML experience even if now in a different role)
    max_historical_tier = max(all_tiers)
    if max_historical_tier > current_tier:
        trajectory_bonus = (max_historical_tier - current_tier) * 0.05
        base_score = min(1.0, base_score + trajectory_bonus)

    # Strong positive signal: current title is tier 3+ at a product company
    # (not a consulting firm) — already handled by disqualifier.py

    return round(base_score, 4)


def get_title_label(tier: int) -> str:
    """Human-readable label for a tier, used in reasoning strings."""
    return {
        4: "strong ML/AI match",
        3: "ML/AI role",
        2: "adjacent technical",
        1: "general engineering",
        0: "unrelated domain",
    }.get(tier, "unknown")
