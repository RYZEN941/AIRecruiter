"""
main.py
-------
Legacy entry point — kept for reference only.

The hackathon submission uses a two-phase approach:

  Phase 1 (offline, unlimited time):
    python precompute.py --candidates ./data/candidates.jsonl

  Phase 2 (ranking, ≤5 min, no network):
    python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv

  Validate:
    python data/validate_submission.py ./output/submission.csv

See README.md for full instructions.
"""

print(
    "\n  [INFO] This project uses a two-phase pipeline.\n"
    "\n"
    "  STEP 1 — Pre-computation (run once, takes ~30-60 min for 100K candidates):\n"
    "    python precompute.py --candidates ./data/candidates.jsonl\n"
    "\n"
    "  STEP 2 — Ranking (<=5 min, no network):\n"
    "    python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv\n"
    "\n"
    "  STEP 3 — Validate:\n"
    "    python data/validate_submission.py ./output/submission.csv\n"
)
