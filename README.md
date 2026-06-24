# BeyondKeywords — AIRecruiter
### Redrob AI Hiring Hackathon | Team: BeyondKeywords | Solo submission

A fully offline, CPU-compliant candidate ranking pipeline for 100,000 structured candidate profiles. Built with sentence-transformers, FAISS, and multi-signal feature scoring. No API keys. No network calls during ranking.

---

## Problem Statement

Rank 100,000 candidate profiles against a Senior AI Engineer job description. Output: a CSV of the top-100 candidates with rank, score, and per-candidate reasoning. The ranking step must complete in ≤5 minutes on CPU with no network access.

---

## Architecture

The pipeline is split into two phases to satisfy the 5-minute constraint:

```
PHASE 1 — Pre-computation (offline, no time limit)
─────────────────────────────────────────────────
candidates.jsonl (100K records)
  → embeddings/embedder.py       Build text repr per candidate + embed with all-MiniLM-L6-v2
  → embeddings/faiss_index.py    Build FAISS flat inner-product index
  → embeddings/*.npy / *.bin     Artifacts saved to disk

PHASE 2 — Ranking (≤5 min, CPU only, no network)
─────────────────────────────────────────────────
Load pre-computed artifacts
  → FAISS retrieval              Top-2000 candidates by semantic similarity to JD
  → scoring/feature_extractor.py Extract structured signals per candidate
  → scoring/hybrid_scorer.py     Weighted multi-signal scoring
  → output/report_generator.py   Write submission.csv (top-100)
```

### Signal Stack

| Signal | Weight | Source |
|--------|--------|--------|
| Semantic similarity | 25% | FAISS cosine sim (JD embedding vs candidate text) |
| Title fit | 20% | Rule-based tier classifier (tiers 0–4) |
| Skill depth | 25% | Proficiency × duration × endorsements per bucket |
| Availability | 15% | 8-component behavioral signal (notice period, active days, open-to-work, etc.) |
| Experience fit | 10% | Gaussian proximity to 5–9 year target |
| Location fit | 5% | City/country tier match |

**Multiplicative penalties applied after weighted sum:**
- Disqualifier multiplier (consulting-only, academic-only, title chaser)
- Honeypot penalty (near-zero for profiles with impossible signal combinations)

---

## Dataset

```
data/
├── candidates.jsonl          # 100,000 structured candidate records (465 MB, not in git)
├── candidate_schema.json     # Full field schema
├── sample_candidates.json    # 50 sample records (in git, for sandbox/testing)
├── job_description.docx      # Raw JD
├── redrob_signals_doc.docx   # Behavioral signal definitions
├── submission_spec.docx      # Official submission spec
├── submission_metadata_template.yaml
├── sample_submission.csv     # Format reference only
└── validate_submission.py    # Local validator
```

Each candidate record contains:
- `profile` — headline, summary, title, location, years of experience
- `career_history` — roles with title, company, duration, description
- `skills` — name, proficiency level, duration_months, endorsements
- `redrob_signals` — behavioral signals: notice period, active days, open-to-work, GitHub score, etc.
- `skill_assessment_scores` — platform-verified assessment results

---

## Repository Structure

```
AIRecruiter/
├── precompute.py                 # Phase 1: embed + build FAISS index
├── rank.py                       # Phase 2: load artifacts + score + write CSV
├── main.py                       # Redirect notice (see precompute.py / rank.py)
├── requirements.txt
├── submission_metadata.yaml      # Filled submission metadata
├── README.md
│
├── embeddings/
│   ├── embedder.py               # Streaming JSONL embedder with cache validation
│   └── __init__.py
│
├── retrieval/
│   ├── faiss_index.py            # Build / save / load / query FAISS index
│   └── __init__.py
│
├── scoring/
│   ├── jd_profile.py             # Hardcoded JD requirements + signal weights
│   ├── feature_extractor.py      # Aggregates all signals per candidate
│   ├── title_classifier.py       # Rule-based title → tier 0–4
│   ├── hybrid_scorer.py          # Weighted scoring + penalty application
│   ├── honeypot_detector.py      # Impossible profile detection
│   ├── availability_scorer.py    # Behavioral signal composite
│   ├── disqualifier.py           # Multiplicative hard penalties
│   └── __init__.py
│
├── output/
│   ├── report_generator.py       # CSV writer + reasoning generator
│   └── __init__.py
│
├── parser/
│   ├── jd_parser.py              # Stub (JD hardcoded in scoring/jd_profile.py)
│   └── resume_parser.py          # Stub (replaced by feature_extractor.py)
│
├── reranker/
│   ├── llm_reranker.py           # Stub (Option B: offline, no LLM)
│   └── __init__.py
│
└── data/
    ├── candidates.jsonl          # NOT in git (465 MB)
    ├── sample_candidates.json    # 50 sample records
    ├── candidate_schema.json
    ├── validate_submission.py
    └── ...
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt** (no API keys needed):
```
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
numpy>=1.26.0
pandas>=2.0.0
```

### 2. Place the full dataset
```
data/candidates.jsonl    ← 100,000 candidate records (465 MB)
```

---

## How to Reproduce

### Step 1 — Pre-compute (run once)

Embeds all 100,000 candidates and builds the FAISS index. Takes ~30–60 minutes on CPU. Saves artifacts to `embeddings/`.

```bash
python precompute.py --candidates ./data/candidates.jsonl
```

Output artifacts:
```
embeddings/jd_embedding.npy           # JD vector (384-dim)
embeddings/candidate_embeddings.npy   # 100K × 384 float32 matrix (~147 MB)
embeddings/candidate_ids.json         # ID list in row order
embeddings/faiss_index.bin            # FAISS flat IP index (~147 MB)
embeddings/candidate_cache_meta.json  # Source path + count (cache validation)
```

> **Cache validation:** If you previously ran precompute on sample data, the embedder detects the source path mismatch and automatically rebuilds. Pass `--force` to override manually.

### Step 2 — Rank (≤5 min, no network)

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

### Step 3 — Validate

```bash
python data/validate_submission.py ./output/submission.csv
```

Expected output: `Submission is valid.`

---

## Single Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

> Requires Phase 1 artifacts in `embeddings/`. See Step 1 above if running from scratch.

---

## Offline / No-Network Compliance

| Constraint | Status |
|------------|--------|
| No network calls during ranking | ✅ Verified — zero HTTP/socket calls in rank.py |
| CPU only | ✅ FAISS flat index, sentence-transformers CPU mode |
| ≤16 GB RAM | ✅ ~147 MB embeddings + ~500 MB streaming overhead |
| No API keys required | ✅ Anthropic dependency fully removed |
| ≤5 min ranking step | ✅ Tested: ~40s on CPU for 100K candidates |

---

## Honeypot Handling

The dataset contains ~80 honeypot candidates with impossible profiles (e.g., 8 years experience at a 3-year-old company). The `scoring/honeypot_detector.py` module applies a near-zero multiplier to candidates flagged by any of 6 heuristic rules:

- Experience years exceeds company founding date
- Expert proficiency claimed with zero duration
- Implausibly many skills at expert level with zero endorsements
- Career timeline contradictions

Honeypots are scored but receive a final multiplier of 0.05 — effectively excluded from top-100.

---

## Key Design Decisions

**Why two phases?** The 5-minute constraint applies only to the ranking step. Pre-computation of embeddings for 100K candidates takes ~45 minutes — well outside the budget. Splitting into offline precompute + fast online rank satisfies both constraints.

**Why no LLM reranking?** Option B was chosen: fully offline, no API dependencies. The 5% LLM weight is redistributed to skill depth automatically when no cache is present.

**Why FAISS flat (exact) vs approximate?** With 100K × 384 dims, exact search takes ~10ms per query — fast enough. Approximate methods (IVF, HNSW) add complexity for no measurable benefit at this scale.

**Why all-MiniLM-L6-v2?** Best throughput/quality tradeoff for CPU: ~1,000 candidates/second, 384 dims, strong on technical domain matching.
