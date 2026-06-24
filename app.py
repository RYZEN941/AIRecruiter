"""
app.py — BeyondKeywords Ranker Sandbox
======================================
Streamlit app for the Redrob hackathon sandbox requirement (Section 10.5).

Accepts a small candidate sample (<=100 records) as JSON upload,
runs the full offline ranking pipeline, and returns a downloadable CSV.

No API keys. No network calls. CPU only.
"""

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make sure our modules are importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BeyondKeywords — AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9ff;
        border: 1px solid #e0e3ff;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .score-high  { color: #1a7a4a; font-weight: 600; }
    .score-mid   { color: #b87d00; font-weight: 600; }
    .score-low   { color: #c0392b; font-weight: 600; }
    .stAlert > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Cached model + scoring imports ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def get_jd_embedding(_model):
    from scoring.jd_profile import JD_EMBED_TEXT
    emb = _model.encode(
        [JD_EMBED_TEXT],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    return emb.reshape(1, -1)


# ── Core ranking function ─────────────────────────────────────────────────────

def run_ranking(candidates: list, model, jd_embedding: np.ndarray) -> pd.DataFrame:
    """
    Run the full ranking pipeline on a list of candidate dicts.
    Returns a DataFrame of results sorted by final_score.
    """
    import faiss
    from scoring.feature_extractor import extract_features, build_candidate_text
    from scoring.hybrid_scorer import score_candidate

    n = len(candidates)

    # 1. Embed candidates
    texts = [build_candidate_text(c) for c in candidates]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=64,
    ).astype(np.float32)

    # 2. Build tiny in-memory FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 3. Query
    k = min(n, 100)
    sims, idxs = index.search(jd_embedding.reshape(1, -1).astype(np.float32), k)

    # 4. Score
    results = []
    for idx, sim in zip(idxs[0], sims[0]):
        if idx == -1:
            continue
        candidate = candidates[idx]
        features = extract_features(candidate)
        sem_score = float(np.clip(sim, 0.0, 1.0))
        score_dict = score_candidate(features, sem_score, llm_cached_score=None)
        results.append(score_dict)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    # 5. Assign ranks
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return pd.DataFrame(results)


def build_submission_csv(df: pd.DataFrame) -> str:
    """Build submission-format CSV string from results DataFrame."""
    from output.report_generator import generate_reasoning

    rows = []
    for _, row in df.iterrows():
        score_dict = row.to_dict()
        reasoning = generate_reasoning(score_dict)
        rows.append({
            "candidate_id": score_dict["candidate_id"],
            "rank":         int(score_dict["rank"]),
            "score":        f"{score_dict['final_score']:.6f}",
            "reasoning":    reasoning,
        })

    out = io.StringIO()
    import csv
    writer = csv.DictWriter(out, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


# ── UI ─────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-header">🎯 BeyondKeywords — AI Candidate Ranker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Redrob Hackathon Sandbox · Fully offline · CPU only · No API keys</div>', unsafe_allow_html=True)
    st.divider()

    # Sidebar — JD snapshot
    with st.sidebar:
        st.markdown("### Job Description")
        st.markdown("""
**Role:** Senior AI Engineer  
**Company:** Redrob AI (Series A)

**Requirements:**
- 5–9 years experience
- Production ML/retrieval systems
- Vector DBs, Python, LLMs
- Product company experience

**Disqualifiers:**
- Consulting-only background
- Pure researcher (no shipping)
- <3 years or >12 years exp
""")
        st.divider()
        st.markdown("### About")
        st.caption("Team: **BeyondKeywords**  \nMember: Ritik Ranjan  \n[GitHub](https://github.com/RYZEN941/AIRecruiter)")

    # Load model once
    model = load_model()
    jd_embedding = get_jd_embedding(model)

    # File upload
    st.markdown("### Upload Candidate Data")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a JSON file — array of candidate records (≤100 candidates)",
            type=["json"],
            help="Must match the Redrob candidate schema. Use sample_candidates.json from the hackathon bundle to test.",
        )
    with col2:
        st.info("**Supported format**\n\nJSON array: `[{...}, {...}]`\n\nOr JSONL: one record per line\n\nMax 100 candidates")

    # Use sample data as default
    use_sample = st.checkbox("Use built-in sample data (50 candidates)", value=uploaded_file is None)

    candidates = None

    if use_sample and uploaded_file is None:
        sample_path = ROOT / "data" / "sample_candidates.json"
        if sample_path.exists():
            with open(sample_path, encoding="utf-8") as f:
                candidates = json.load(f)
            st.success(f"Loaded {len(candidates)} sample candidates from `data/sample_candidates.json`")
        else:
            st.warning("Sample file not found. Please upload a candidate JSON file.")

    elif uploaded_file is not None:
        raw = uploaded_file.read().decode("utf-8")
        try:
            # Try JSON array first
            candidates = json.loads(raw)
            if isinstance(candidates, dict):
                candidates = [candidates]
        except json.JSONDecodeError:
            # Try JSONL
            try:
                candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
            except Exception:
                st.error("Could not parse file. Must be a JSON array or JSONL.")
                return

        if len(candidates) > 100:
            st.warning(f"Uploaded {len(candidates)} candidates — truncating to 100 for sandbox (full run uses 100K).")
            candidates = candidates[:100]
        st.success(f"Loaded {len(candidates)} candidates from upload.")

    if candidates is None:
        st.info("Upload a candidate file or use the built-in sample data to begin.")
        return

    # Run ranking
    st.divider()
    st.markdown("### Ranking")

    if st.button("Run Ranker", type="primary", use_container_width=True):
        with st.spinner(f"Ranking {len(candidates)} candidates..."):
            df = run_ranking(candidates, model, jd_embedding)

        st.success(f"Ranked {len(df)} candidates in {len(df)} results.")

        # Summary metrics
        top = df.iloc[0] if len(df) > 0 else None
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total ranked", len(df))
        with c2:
            disq = int(df["is_disqualified"].sum()) if "is_disqualified" in df.columns else 0
            st.metric("Disqualified", disq)
        with c3:
            hp = int(df["is_likely_honeypot"].sum()) if "is_likely_honeypot" in df.columns else 0
            st.metric("Honeypots detected", hp)
        with c4:
            if top is not None:
                st.metric("Top score", f"{top['final_score']*100:.1f}%")

        st.divider()

        # Results table
        st.markdown("### Top Candidates")
        display_cols = ["rank", "candidate_id", "name", "current_title",
                        "years_of_experience", "final_score", "title_score",
                        "skill_depth_score", "availability_score", "location_score"]
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()

        # Format scores as percentages
        for col in ["final_score", "title_score", "skill_depth_score",
                    "availability_score", "location_score"]:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(600, 40 + 35 * len(display_df)),
        )

        # Per-candidate detail expander
        st.markdown("### Candidate Details")
        for _, row in df.head(10).iterrows():
            flags = row.get("disqualifier_flags", []) + row.get("honeypot_flags", [])
            flag_str = " | ".join(flags) if flags else "None"
            with st.expander(
                f"#{int(row['rank'])}  {row.get('name','?')}  —  "
                f"{row.get('current_title','?')}  ({row.get('years_of_experience',0):.1f}y)  "
                f"Score: {row['final_score']*100:.1f}%"
            ):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Final Score", f"{row['final_score']*100:.1f}%")
                    st.metric("Title Tier", row.get("title_tier", "?"))
                with c2:
                    st.metric("Skill Depth", f"{row.get('skill_depth_score',0)*100:.1f}%")
                    st.metric("Availability", f"{row.get('availability_score',0)*100:.1f}%")
                with c3:
                    st.metric("Semantic Sim.", f"{row.get('semantic_score',0)*100:.1f}%")
                    st.metric("Location", f"{row.get('location_score',0)*100:.1f}%")
                if flags:
                    st.warning(f"Flags: {flag_str}")

        # Download buttons
        st.divider()
        st.markdown("### Download")
        col1, col2 = st.columns(2)

        with col1:
            csv_str = build_submission_csv(df)
            st.download_button(
                label="Download submission.csv",
                data=csv_str.encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with col2:
            full_json = df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download full results (JSON)",
                data=full_json.encode("utf-8"),
                file_name="ranking_results.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
