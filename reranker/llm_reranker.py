"""
reranker/llm_reranker.py
------------------------
Option B stub — LLM pre-scoring disabled.

No Anthropic API key is required. This module exists only so that rank.py
can call load_llm_cache() without error. It always returns an empty dict,
which causes hybrid_scorer to redistribute the 5% llm_bonus weight to
skill_depth automatically.

To re-enable Claude pre-scoring in the future, replace this file with the
full implementation from git history.
"""

from pathlib import Path
from typing import Dict

LLM_CACHE_PATH = Path(__file__).parent.parent / "embeddings" / "llm_scores_cache.json"


def load_llm_cache() -> Dict[str, float]:
    """
    Load cached LLM scores if they exist, otherwise return empty dict.

    In Option B (no API key), the cache file will not exist and this
    returns {} — the hybrid scorer redistributes the weight automatically.
    """
    if not LLM_CACHE_PATH.exists():
        print("  [LLM] No cache found — running without LLM bonus (Option B).")
        return {}
    import json
    with open(LLM_CACHE_PATH, "r") as f:
        raw = json.load(f)
    scores = {cid: v["llm_score"] for cid, v in raw.items() if "llm_score" in v}
    print(f"  [LLM] Loaded {len(scores)} cached scores.")
    return scores
