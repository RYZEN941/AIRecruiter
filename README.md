# 🤖 AIRecruiter — AI-Powered Recruitment Pipeline

An end-to-end AI recruitment pipeline that uses Claude API, sentence-transformers, and FAISS to intelligently rank candidates for a job.

## Architecture

```
JD (raw text)
  → parser/jd_parser.py         # Extract structured requirements via Claude API
  → embeddings/embedder.py      # Generate sentence-transformer embeddings
  → retrieval/faiss_index.py    # FAISS index build + top-50 retrieval
  → scoring/hybrid_scorer.py    # Weighted multi-signal scoring → top-20
  → reranker/llm_reranker.py    # Claude LLM deep analysis → top-10
  → output/report_generator.py  # Explainability report per candidate
  → main.py                     # Orchestrates everything
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
```bash
# Copy the template
cp .env.example .env

# Edit .env and add your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add your data
- Put your job description in `data/jd.txt`
- Put each resume (plain text, one per file) in `data/resumes/` as `.txt` files

> Sample data is already provided for 5 demo candidates.

### 4. Run the pipeline
```bash
python main.py
```

## Pipeline Stages

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1 | Data Loader | `data/` directory | JD text + resume texts |
| 2 | JD + Resume Parser | Raw text | Structured dicts (via Claude) |
| 3 | Embedder | Raw texts | 384-dim vectors |
| 4 | FAISS Retrieval | JD embedding | Top-50 by cosine similarity |
| 5 | Hybrid Scorer | Parsed dicts + FAISS scores | Top-20 ranked |
| 6 | LLM Reranker | JD text + candidates | Top-10 deep evaluated |
| 7 | Report Generator | Top-10 results | `output/ranking_report.md` |

## Scoring Signals

### Hybrid Scorer (Stage 5)
| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic Match | 35% | FAISS cosine similarity |
| Skill Match | 25% | Required skills intersection |
| Experience Fit | 20% | Gaussian proximity to JD years range |
| Project Impact | 10% | Normalised count of quantified achievements |
| Velocity | 10% | Career progression speed (0–1) |
| Red Flag Penalty | -5% each | Detected concerns (job hopping, unsupported claims) |

### LLM Reranker (Stage 6)
| Dimension | Weight |
|-----------|--------|
| Technical Fit | 35% |
| Experience Fit | 25% |
| Career Velocity | 15% |
| Domain Transfer | 15% |
| Growth Potential | 10% |

**Combined score = Hybrid × 0.4 + LLM × 0.6**

## Resume Parser — Differentiating Signals

Beyond standard fields, each resume is also analysed for:

- **`velocity_score`** — How fast the candidate progressed (junior → mid in <18 months = high velocity). Float 0–1.
- **`quantified_achievements`** — Bullet points with measurable impact numbers.
- **`skill_depth_chains`** — Technology dependency chains (e.g. Python → PyTorch → CUDA → TensorRT) showing true depth vs surface-level keyword claims.
- **`red_flags`** — Automatically detected concerns (job hopping, skills not backed by projects, etc.).

## Demo Candidates

| Candidate | Profile | Expected Rank |
|-----------|---------|---------------|
| `candidate_001` | Senior ML Engineer, deep RAG experience, fast promotions | 🥇 #1 |
| `candidate_002` | Right keywords but shallow projects, no metrics | Middle |
| `candidate_003` | Healthcare ML scientist, strong transferable skills | Top 3 |
| `candidate_004` | Red flags: 5 jobs in 4 years, no impact metrics | Bottom |
| `candidate_005` | No ML title but OSS NLP libraries with 14K stars | Top 3 |

## Output

The final report is saved to `output/ranking_report.md` and includes:
- Summary ranking table
- Per-candidate report cards with:
  - Matched / missing required skills
  - Quantified impact bullets
  - Career velocity narrative
  - Growth potential and ramp time estimate
  - Red flags with score penalties
  - Interview focus areas
  - Full score breakdown (hybrid signals + LLM dimensions)

## File Structure

```
AIRecruiter/
├── main.py                     # Pipeline orchestrator
├── requirements.txt
├── .env.example                # Copy → .env
├── data/
│   ├── jd.txt                  # Job description
│   └── resumes/
│       ├── candidate_001.txt
│       └── ...
├── parser/
│   ├── jd_parser.py            # Claude-based JD parsing
│   └── resume_parser.py        # Claude-based resume parsing
├── embeddings/
│   ├── embedder.py             # sentence-transformers
│   └── candidate_store.pkl     # Auto-generated embedding cache
├── retrieval/
│   └── faiss_index.py          # FAISS flat index
├── scoring/
│   └── hybrid_scorer.py        # Multi-signal weighted scoring
├── reranker/
│   └── llm_reranker.py         # Claude LLM deep evaluation
└── output/
    ├── report_generator.py     # Markdown report
    └── ranking_report.md       # Auto-generated output
```
