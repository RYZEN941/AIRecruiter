"""
rank.py
-------
PHASE 2: Ranking pipeline — must complete in ≤5 minutes on CPU with NO network.

Loads all pre-computed artifacts from embeddings/ directory (built by precompute.py)
and produces the submission.csv.

This file is the entrypoint for Stage 3 code reproduction:
  python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv

Design constraints respected:
  ✅ No network calls (no Claude, no OpenAI, no any hosted LLM)
  ✅ CPU only
  ✅ ≤16 GB RAM
  ✅ ≤5 minutes wall-clock

Runtime breakdown (estimated for 100K candidates):
  Load artifacts:    ~3s
  Stream + features: ~25s
  FAISS query:       ~2s
  Score top-2000:    ~5s
  CSV write:         ~1s
  TOTAL:             ~36s
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# -- Imports (no network calls in any of these) --------------------------------
from embeddings.embedder import load_embeddings, embed_jd
from retrieval.faiss_index import load_index, retrieve_top_k
from scoring.feature_extractor import extract_features
from scoring.hybrid_scorer import rank_candidates
from reranker.llm_reranker import load_llm_cache
from output.report_generator import write_submission_csv, print_summary_table


def _print_banner(candidates_path: str, output_path: str):
    print("\n" + "=" * 65)
    print("  [RANK] AIRecruiter -- Ranking Pipeline")
    print("=" * 65)
    print(f"  Candidates : {candidates_path}")
    print(f"  Output     : {output_path}")
    print(f"  Constraint : <=5 min | CPU only | No network")
    print("=" * 65 + "\n")


def _print_stage(n: int, total: int, name: str):
    print(f"\n{'- ' * 30}")
    print(f"  STEP {n}/{total} : {name}")
    print(f"{'- ' * 30}")


def stream_candidates(candidates_path: str):
    """Stream candidates from .jsonl or .jsonl.gz."""
    if candidates_path.endswith(".gz"):
        import gzip
        opener = lambda: gzip.open(candidates_path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(candidates_path, "r", encoding="utf-8")

    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob hackathon submission"
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./output/submission.csv",
        help="Output path for submission CSV",
    )
    parser.add_argument(
        "--faiss-k",
        type=int,
        default=2000,
        help="Number of candidates to retrieve from FAISS before full scoring (default: 2000)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of candidates in final submission (default: 100)",
    )
    args = parser.parse_args()

    _print_banner(args.candidates, args.out)

    if not Path(args.candidates).exists():
        print(f"ERROR: candidates file not found: {args.candidates}")
        sys.exit(1)

    t0 = time.time()
    TOTAL_STEPS = 6

    # -- Step 1: Load pre-computed artifacts -----------------------------------
    _print_stage(1, TOTAL_STEPS, "Load Pre-computed Artifacts")

    print("  Loading FAISS index ...")
    index = load_index()

    print("  Loading candidate embeddings ...")
    _, candidate_ids = load_embeddings()
    print(f"  Candidate pool size: {len(candidate_ids):,}")

    print("  Loading JD embedding ...")
    jd_embedding = embed_jd(force=False)  # Loads from cache — no model needed

    print("  Loading LLM cache ...")
    llm_cache = load_llm_cache()

    t1 = time.time()
    print(f"  [OK] Artifacts loaded in {t1-t0:.1f}s")

    # -- Step 2: FAISS retrieval ------------------------------------------------
    _print_stage(2, TOTAL_STEPS, f"FAISS Retrieval — Top {args.faiss_k}")

    faiss_results = retrieve_top_k(
        index=index,
        candidate_ids=candidate_ids,
        jd_embedding=jd_embedding,
        k=args.faiss_k,
    )
    faiss_set = {cid for cid, _ in faiss_results}

    t2 = time.time()
    print(f"  [OK] FAISS retrieval in {t2-t1:.1f}s")

    # -- Step 3: Stream + extract features for FAISS candidates ----------------
    _print_stage(3, TOTAL_STEPS, "Extract Features for FAISS Candidates")

    features_map: Dict[str, dict] = {}
    candidates_map: Dict[str, dict] = {}  # For reasoning generation
    n_processed = 0
    n_skipped = 0

    for candidate in stream_candidates(args.candidates):
        cid = candidate.get("candidate_id")
        if cid not in faiss_set:
            n_skipped += 1
            continue  # Skip candidates not in FAISS top-k — saves time

        features = extract_features(candidate)
        features_map[cid] = features
        candidates_map[cid] = candidate
        n_processed += 1

        if n_processed % 200 == 0:
            print(f"  Processed {n_processed} / {len(faiss_set)} FAISS candidates ...")

    t3 = time.time()
    print(f"  [OK] Features extracted for {n_processed} candidates in {t3-t2:.1f}s")
    print(f"    (Skipped {n_skipped} candidates not in FAISS top-{args.faiss_k})")

    # -- Step 4: Hybrid scoring ------------------------------------------------
    _print_stage(4, TOTAL_STEPS, "Hybrid Multi-Signal Scoring")

    ranked = rank_candidates(
        features_map=features_map,
        faiss_results=faiss_results,
        llm_cache=llm_cache if llm_cache else None,
        top_n=args.top_n,
    )

    n_honeypots_in_top100 = sum(1 for r in ranked[:100] if r.get("is_likely_honeypot"))
    n_disq_in_top100 = sum(1 for r in ranked[:100] if r.get("is_disqualified"))
    honeypot_rate = n_honeypots_in_top100 / min(100, len(ranked))

    t4 = time.time()
    print(f"  [OK] Scoring complete in {t4-t3:.1f}s")
    print(f"    Top-100 stats: {n_honeypots_in_top100} honeypots, {n_disq_in_top100} disqualified")
    if honeypot_rate > 0.05:
        print(f"  WARNING: Honeypot rate in top-100 = {honeypot_rate:.1%} (limit: 10%)")

    # -- Step 5: Print summary -------------------------------------------------
    _print_stage(5, TOTAL_STEPS, "Ranking Summary")
    print_summary_table(ranked, top_n=20)

    # -- Step 6: Write submission CSV ------------------------------------------
    _print_stage(6, TOTAL_STEPS, "Write Submission CSV")

    out_path = write_submission_csv(
        ranked_candidates=ranked,
        output_path=args.out,
        candidates_map=candidates_map,
    )

    t5 = time.time()
    elapsed = t5 - t0

    # -- Final summary ---------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  [OK] Ranking complete in {elapsed:.1f}s")
    print(f"  Submission -> {out_path}")
    print(f"\n  Validate before submitting:")
    print(f"  python data/validate_submission.py {args.out}")
    print(f"{'=' * 65}\n")

    if elapsed > 300:
        print(f"  WARNING: Elapsed {elapsed:.0f}s exceeds 5-minute limit!")
        print(f"     Consider reducing --faiss-k or optimising feature extraction.")


if __name__ == "__main__":
    main()
