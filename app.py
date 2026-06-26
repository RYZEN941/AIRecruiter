"""
app.py — BeyondKeywords Ranker Sandbox
======================================
Streamlit sandbox for the Redrob hackathon (Section 10.5).

UX fixes applied (v2):
  1. Session state — results persist across reruns; button never flickers.
  2. Consistent layout — results section always rendered; no scroll-to-top.
  3. Hero banner — gradient card with strong contrast, readable on any theme.
  4. Score colour coding — green / amber / red inline badges.
  5. Tabbed results — Table | Signal Breakdown | Download (no vertical scroll marathon).
  6. Sidebar enhanced — pipeline diagram + signal weights for judge context.
  7. Dataset source reset — switching between sample / upload clears old results.

No API keys. No network calls during ranking. CPU only.
"""

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="BeyondKeywords — AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a4e 0%, #2d2d8e 50%, #1a4a7a 100%);
    border-radius: 14px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 4px 24px rgba(30,30,120,0.18);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #b8c8e8;
    margin: 0 0 1rem;
}
.hero-badges span {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    color: #ddeeff;
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 12px;
    margin-right: 6px;
}

/* ── Score colour badges ── */
.badge-high { background:#e6f9f0; color:#1a7a4a; border:1px solid #a3e6c4;
              border-radius:6px; padding:2px 8px; font-weight:700; }
.badge-mid  { background:#fff8e6; color:#a06000; border:1px solid #ffd980;
              border-radius:6px; padding:2px 8px; font-weight:700; }
.badge-low  { background:#fdecea; color:#c0392b; border:1px solid #f5b7b1;
              border-radius:6px; padding:2px 8px; font-weight:700; }

/* ── Flag badges ── */
.flag-hp   { background:#fff3cd; color:#856404; border:1px solid #ffc107;
             border-radius:5px; padding:1px 7px; font-size:0.78rem; }
.flag-disq { background:#f8d7da; color:#842029; border:1px solid #f5c2c7;
             border-radius:5px; padding:1px 7px; font-size:0.78rem; }

/* ── Rank number ── */
.rank-num { font-size:1.1rem; font-weight:800; color:#3730a3; }

/* ── Section header — strong contrast on light theme ── */
.section-hdr {
    font-size: 1.15rem; font-weight: 700; color: #1F2937;
    border-left: 4px solid #7c9cff; padding-left: 10px;
    margin: 1.2rem 0 0.7rem;
}

/* ── Divider ── */
hr { margin: 1rem 0; border-color: rgba(0,0,0,0.10); }

/* ── Hide Streamlit default footer ── */
footer { visibility: hidden; }

/* ── Rankings table: dark text for light theme ── */
.rank-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}
.rank-table thead tr {
    background: rgba(80,100,220,0.10);
    border-bottom: 2px solid rgba(80,100,220,0.25);
}
.rank-table thead th {
    padding: 10px 8px;
    text-align: left;
    font-weight: 700;
    color: #1F2937;
    letter-spacing: 0.02em;
}
.rank-table thead th.center { text-align: center; }
.rank-table tbody tr {
    border-bottom: 1px solid rgba(0,0,0,0.06);
}
.rank-table tbody tr:hover {
    background: rgba(80,100,220,0.04);
}
.rank-table td {
    padding: 9px 8px;
    vertical-align: middle;
}
.td-id    { font-size:0.78rem; color:#4B5563; font-family:monospace; }
.td-name  { font-weight:700;   color:#111827; }
.td-title { color:#374151; }
.td-yoe   { text-align:center; color:#374151; }
.td-score { text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
# Must happen before any widget renders to avoid state reset on rerun.
def _init_state():
    defaults = {
        "results_df":   None,   # pd.DataFrame of ranked results
        "csv_str":      None,   # pre-built CSV string
        "json_str":     None,   # pre-built JSON string
        "n_ranked":     0,
        "data_source":  None,   # "sample" | "upload:<filename>"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Cached model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading embedding model (first run only — ~15s)...")
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


# ── Core ranking ───────────────────────────────────────────────────────────────
def run_ranking(candidates: list, model, jd_embedding: np.ndarray) -> pd.DataFrame:
    import faiss
    from scoring.feature_extractor import extract_features, build_candidate_text
    from scoring.hybrid_scorer import score_candidate

    n = len(candidates)
    texts = [build_candidate_text(c) for c in candidates]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=64,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    k = min(n, 100)
    sims, idxs = index.search(jd_embedding.reshape(1, -1).astype(np.float32), k)

    results = []
    for idx, sim in zip(idxs[0], sims[0]):
        if idx == -1:
            continue
        features = extract_features(candidates[idx])
        sem_score = float(np.clip(sim, 0.0, 1.0))
        score_dict = score_candidate(features, sem_score, llm_cached_score=None)
        results.append(score_dict)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return pd.DataFrame(results)


def _build_csv(df: pd.DataFrame) -> str:
    from output.report_generator import generate_reasoning
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        rows.append({
            "candidate_id": d["candidate_id"],
            "rank":         int(d["rank"]),
            "score":        f"{d['final_score']:.6f}",
            "reasoning":    generate_reasoning(d),
        })
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# ── Score badge helper ─────────────────────────────────────────────────────────
def _badge(score: float) -> str:
    pct = score * 100
    cls = "badge-high" if pct >= 55 else ("badge-mid" if pct >= 35 else "badge-low")
    return f'<span class="{cls}">{pct:.1f}%</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🎯 BeyondKeywords")
        st.caption("Redrob AI Hackathon · Team: **Ritik Ranjan**")
        st.markdown("[GitHub Repo](https://github.com/RYZEN941/AIRecruiter)", unsafe_allow_html=False)
        st.divider()

        st.markdown("### 📋 Target Role")
        st.markdown("""
**Senior AI Engineer**  
Redrob AI · Series A · India hybrid

| Requirement | Target |
|---|---|
| Experience | 5–9 years |
| Domain | ML / Retrieval / IR |
| Company type | Product startup |
| Location | Pune / Noida preferred |
""")
        st.divider()

        st.markdown("### ⚙️ Scoring Signals")
        st.markdown("""
| Signal | Weight |
|---|---|
| 🔵 Semantic match | 25% |
| 🟣 Title fit (tier 0–4) | 20% |
| 🟢 Skill depth | 25% |
| 🟡 Availability | 15% |
| 🟠 Experience fit | 10% |
| ⚪ Location | 5% |

Honeypot & disqualifier  
penalties applied multiplicatively.
""")
        st.divider()

        st.markdown("### 🔗 Pipeline")
        st.code("""Upload JSON
  → Embed (MiniLM-L6-v2)
  → FAISS in-memory
  → Feature extract
  → Hybrid score
  → Ranked CSV""", language=None)


# ── Hero banner ────────────────────────────────────────────────────────────────
def _render_hero():
    st.markdown("""
<div class="hero">
  <div class="hero-title">🎯 BeyondKeywords — AI Candidate Ranker</div>
  <div class="hero-sub">
    Redrob Hackathon Sandbox &nbsp;·&nbsp; Multi-signal offline ranking
    for 100K candidates &nbsp;·&nbsp; Senior AI Engineer JD
  </div>
  <div class="hero-badges">
    <span>🔒 Fully Offline</span>
    <span>🖥️ CPU Only</span>
    <span>🚫 No API Keys</span>
    <span>⚡ &lt;5 Min Ranking</span>
    <span>🐝 Honeypot Detection</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Data loading section ───────────────────────────────────────────────────────
def _load_candidates():
    """
    Renders the data source selector and returns a list of candidate dicts,
    or None if nothing is ready yet.

    Also detects when the data source changes and clears old results from state.
    """
    st.markdown('<div class="section-hdr">📂 Candidate Data Source</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload a JSON / JSONL file (≤100 candidates)",
            type=["json"],
            help="Must match the Redrob candidate schema. "
                 "Use data/sample_candidates.json from the hackathon bundle.",
            label_visibility="collapsed",
        )

    # key keeps checkbox state stable across reruns; no dynamic value= to avoid
    # fighting user intent when they switch between sample and upload.
    if "use_sample_cb" not in st.session_state:
        st.session_state["use_sample_cb"] = True
    use_sample = st.checkbox(
        "Use built-in sample data (50 candidates from hackathon bundle)",
        key="use_sample_cb",
    )

    with col_info:
        st.info("**Formats accepted**\n\n"
                "JSON array `[{…}]`\n\nJSONL (one record/line)\n\nMax 100 candidates")

    candidates = None
    source_key = None

    if use_sample and uploaded_file is None:
        sample_path = ROOT / "data" / "sample_candidates.json"
        if sample_path.exists():
            with open(sample_path, encoding="utf-8") as f:
                candidates = json.load(f)
            source_key = "sample"
            st.success(f"✅ Loaded **{len(candidates)}** sample candidates (data/sample_candidates.json)")
        else:
            st.error("sample_candidates.json not found in data/. Please upload a file.")
            return None

    elif uploaded_file is not None:
        raw = uploaded_file.read().decode("utf-8")
        try:
            candidates = json.loads(raw)
            if isinstance(candidates, dict):
                candidates = [candidates]
        except json.JSONDecodeError:
            try:
                candidates = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
            except Exception:
                st.error("❌ Could not parse file. Must be a JSON array or JSONL.")
                return None

        if len(candidates) > 100:
            st.warning(f"Uploaded {len(candidates)} candidates — truncating to 100 for sandbox.")
            candidates = candidates[:100]
        source_key = f"upload:{uploaded_file.name}"
        st.success(f"✅ Loaded **{len(candidates)}** candidates from **{uploaded_file.name}**")

    else:
        st.info("☝️ Upload a file or check 'Use built-in sample data' to begin.")
        return None

    # Clear old results if data source changed
    if source_key != st.session_state["data_source"]:
        st.session_state["results_df"] = None
        st.session_state["csv_str"]    = None
        st.session_state["json_str"]   = None
        st.session_state["n_ranked"]   = 0
        st.session_state["data_source"] = source_key

    return candidates


# ── Run section ────────────────────────────────────────────────────────────────
def _render_run_section(candidates: list, model, jd_embedding: np.ndarray):
    """
    Renders the Run Ranker button and triggers computation.
    Results are stored in session_state so they survive all future reruns —
    this eliminates both the flicker and the scroll-to-top on rerun.
    """
    st.markdown('<div class="section-hdr">🚀 Run Ranking Pipeline</div>', unsafe_allow_html=True)

    already_ranked = st.session_state["results_df"] is not None
    btn_label = "🔄 Re-run Ranker" if already_ranked else "▶ Run Ranker"

    if st.button(btn_label, type="primary", use_container_width=True, key="run_btn"):
        # Wrap computation in a spinner; store result in state, not local variable.
        # This is the key fix: once stored in state, results render on ALL
        # subsequent reruns without re-executing the pipeline.
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            with st.spinner(f"⏳ Embedding + ranking {len(candidates)} candidates..."):
                df = run_ranking(candidates, model, jd_embedding)
                csv_str  = _build_csv(df)
                json_str = df.to_json(orient="records", indent=2)

        progress_placeholder.empty()   # remove spinner container cleanly

        st.session_state["results_df"] = df
        st.session_state["csv_str"]    = csv_str
        st.session_state["json_str"]   = json_str
        st.session_state["n_ranked"]   = len(df)


# ── Results section ────────────────────────────────────────────────────────────
def _render_results():
    """
    Always rendered from session_state — never inside an `if st.button` block.
    This guarantees results stay visible across all reruns (checkbox changes,
    tab switches, expander opens) without scroll jumps.
    """
    df = st.session_state["results_df"]
    if df is None or df.empty:
        return

    n = len(df)
    top = df.iloc[0]
    disq = int(df["is_disqualified"].sum()) if "is_disqualified" in df.columns else 0
    hp   = int(df["is_likely_honeypot"].sum()) if "is_likely_honeypot" in df.columns else 0

    st.markdown('<div class="section-hdr">📊 Results</div>', unsafe_allow_html=True)

    # ── KPI strip ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Candidates ranked", n)
    k2.metric("Top score", f"{top['final_score']*100:.1f}%")
    k3.metric("Disqualified", disq)
    k4.metric("Honeypots caught", hp)
    k5.metric("Clean candidates", n - disq - hp)

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_table, tab_signals, tab_details, tab_download = st.tabs(
        ["🏆 Rankings", "📈 Signal Breakdown", "🔍 Candidate Details", "💾 Download"]
    )

    # ── Tab 1: Rankings ────────────────────────────────────────────────────────
    with tab_table:
        display_cols = [
            "rank", "candidate_id", "name", "current_title",
            "years_of_experience", "final_score",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        disp = df[display_cols].copy()

        # Render as styled HTML table — all colours set for dark-theme readability
        rows_html = ""
        for _, row in disp.iterrows():
            rank  = int(row["rank"])
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"<span class='rank-num'>#{rank}</span>")
            badge = _badge(row.get("final_score", 0))
            cid   = row.get("candidate_id", "")
            name  = row.get("name", "—")
            title = row.get("current_title", "—")
            yoe   = f"{row.get('years_of_experience', 0):.1f}y"
            rows_html += (
                f"<tr>"
                f"<td>{medal}</td>"
                f"<td class='td-id'>{cid}</td>"
                f"<td class='td-name'>{name}</td>"
                f"<td class='td-title'>{title}</td>"
                f"<td class='td-yoe'>{yoe}</td>"
                f"<td class='td-score'>{badge}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table class='rank-table'>"
            f"<thead><tr>"
            f"<th>Rank</th><th>ID</th><th>Name</th><th>Title</th>"
            f"<th class='center'>Exp</th><th class='center'>Score</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table>",
            unsafe_allow_html=True,
        )

    # ── Tab 2: Signal Breakdown ────────────────────────────────────────────────
    with tab_signals:
        sig_cols = [
            "rank", "name", "final_score", "semantic_score",
            "title_score", "skill_depth_score",
            "availability_score", "experience_score", "location_score",
        ]
        sig_cols = [c for c in sig_cols if c in df.columns]
        sig_df = df[sig_cols].head(20).copy()
        score_fields = [c for c in sig_cols if c.endswith("_score")]
        for col in score_fields:
            sig_df[col] = (sig_df[col] * 100).round(1)

        # Rename columns for display
        rename = {
            "final_score":       "Final %",
            "semantic_score":    "Semantic %",
            "title_score":       "Title %",
            "skill_depth_score": "Skills %",
            "availability_score":"Avail %",
            "experience_score":  "Exp %",
            "location_score":    "Loc %",
        }
        sig_df = sig_df.rename(columns=rename)
        st.dataframe(
            sig_df,
            use_container_width=True,
            hide_index=True,
            height=min(700, 44 + 36 * len(sig_df)),
        )

    # ── Tab 3: Candidate Details ───────────────────────────────────────────────
    with tab_details:
        st.caption("Showing top 10 candidates. Expand any row for full signal breakdown.")
        for _, row in df.head(10).iterrows():
            hp_flags   = row.get("honeypot_flags", []) or []
            disq_flags = row.get("disqualifier_flags", []) or []
            score      = row["final_score"]
            badge_html = _badge(score)
            medal      = {1: "🥇", 2: "🥈", 3: "🥉"}.get(int(row["rank"]), f"#{int(row['rank'])}")

            with st.expander(
                f"{medal}  {row.get('name', '?')}  —  "
                f"{row.get('current_title', '?')}  "
                f"({row.get('years_of_experience', 0):.1f}y)",
                expanded=(int(row["rank"]) == 1),
            ):
                st.markdown(
                    f"**Final score:** {badge_html} &nbsp;&nbsp; "
                    f"**ID:** `{row.get('candidate_id','?')}`",
                    unsafe_allow_html=True,
                )
                ca, cb, cc = st.columns(3)
                ca.metric("Semantic",   f"{row.get('semantic_score', 0)*100:.1f}%")
                ca.metric("Title tier", row.get("title_tier", "?"))
                cb.metric("Skill depth",   f"{row.get('skill_depth_score', 0)*100:.1f}%")
                cb.metric("Availability",  f"{row.get('availability_score', 0)*100:.1f}%")
                cc.metric("Experience fit",f"{row.get('experience_score', 0)*100:.1f}%")
                cc.metric("Location",      f"{row.get('location_score', 0)*100:.1f}%")

                if hp_flags:
                    st.markdown(
                        " ".join(f'<span class="flag-hp">🍯 {f}</span>' for f in hp_flags),
                        unsafe_allow_html=True,
                    )
                if disq_flags:
                    st.markdown(
                        " ".join(f'<span class="flag-disq">🚫 {f}</span>' for f in disq_flags),
                        unsafe_allow_html=True,
                    )

    # ── Tab 4: Download ────────────────────────────────────────────────────────
    with tab_download:
        st.markdown("#### Download your results")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**submission.csv** — hackathon format (candidate_id, rank, score, reasoning)")
            st.download_button(
                label="⬇ Download submission.csv",
                data=st.session_state["csv_str"].encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="dl_csv",
            )
        with dc2:
            st.markdown("**ranking_results.json** — full signal scores for all ranked candidates")
            st.download_button(
                label="⬇ Download full results (JSON)",
                data=st.session_state["json_str"].encode("utf-8"),
                file_name="ranking_results.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json",
            )
        st.info(
            "To validate submission.csv locally: "
            "`python data/validate_submission.py ./submission.csv`"
        )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    _render_sidebar()
    _render_hero()

    # Load model — cached after first call, never re-downloaded
    model       = load_model()
    jd_embedding = get_jd_embedding(model)

    # Data loading
    candidates = _load_candidates()

    if candidates is None:
        # No data yet — still render results if a previous run is in state
        _render_results()
        return

    st.divider()

    # Run button — triggers computation once, stores in state
    _render_run_section(candidates, model, jd_embedding)

    st.divider()

    # Results — always rendered from state, never inside the button block
    # This is what eliminates flicker and scroll-to-top
    _render_results()


if __name__ == "__main__":
    main()
