"""
precompute.py
-------------
PHASE 1: Offline pre-computation pipeline.

Fully offline — no API keys, no network. Uses only sentence-transformers
and FAISS. Run this once before ranking.

Steps:
  1. Embed the JD text      -> embeddings/jd_embedding.npy
  2. Embed all candidates   -> embeddings/candidate_embeddings.npy
                               embeddings/candidate_ids.json
  3. Build FAISS index      -> embeddings/faiss_index.bin

Usage:
  python precompute.py --candidates ./data/candidates.jsonl

  # Force re-embed if cache already exists
  python precompute.py --candidates ./data/candidates.jsonl --force
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from embeddings.embedder import embed_jd, embed_candidates
from retrieval.faiss_index import build_index, save_index, retrieve_top_k


def _print_banner():
    print("\n" + "=" * 65)
    print("  [PRE-COMPUTE] AIRecruiter -- Pre-computation Pipeline")
    print("=" * 65)
    print("  Fully offline: sentence-transformers + FAISS only")
    print("=" * 65 + "\n")


def _print_stage(n: int, total: int, name: str):
    print(f"\n{'- ' * 30}")
    print(f"  STEP {n}/{total} : {name}")
    print(f"{'- ' * 30}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings and FAISS index for AIRecruiter"
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-embedding even if cache already exists",
    )
    args = parser.parse_args()

    candidates_path = args.candidates
    if not Path(candidates_path).exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        sys.exit(1)

    _print_banner()
    t0 = time.time()

    # -- Step 1: Embed JD ------------------------------------------------------
    _print_stage(1, 3, "Embed Job Description")
    jd_embedding = embed_jd(force=args.force)
    print(f"  JD embedding shape: {jd_embedding.shape}")

    # -- Step 2: Embed all candidates ------------------------------------------
    _print_stage(2, 3, f"Embed All Candidates from {candidates_path}")
    matrix, candidate_ids = embed_candidates(
        candidates_path=candidates_path,
        force=args.force,
    )
    print(f"  Embedding matrix: {matrix.shape}")

    # -- Step 3: Build and save FAISS index ------------------------------------
    _print_stage(3, 3, "Build FAISS Index")
    index = build_index(matrix, candidate_ids)
    save_index(index)

    # Sanity check — show top-5 semantic matches to JD
    print("\n  Sanity check — Top-5 semantic matches to JD:")
    top_results = retrieve_top_k(index, candidate_ids, jd_embedding, k=10)
    for cid, sim in top_results[:5]:
        print(f"    {cid}  cosine_sim={sim:.4f}")

    # -- Done ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  [OK] Pre-computation complete in {elapsed:.1f}s")
    print(f"  Artifacts saved to embeddings/:")
    print(f"    jd_embedding.npy")
    print(f"    candidate_embeddings.npy  ({matrix.shape[0]} x {matrix.shape[1]})")
    print(f"    candidate_ids.json        ({len(candidate_ids)} IDs)")
    print(f"    faiss_index.bin")
    print(f"\n  Next: python rank.py --candidates {candidates_path} --out ./output/submission.csv")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
