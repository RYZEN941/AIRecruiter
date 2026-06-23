"""
retrieval/faiss_index.py
------------------------
Updated for the Redrob hackathon:
  - Save/load index to/from disk (required for <5 min ranking)
  - Increased k from 50 to 2000 (need headroom in 100K pool)
  - Same IndexFlatIP for exact cosine similarity

Pre-computation saves the index to embeddings/faiss_index.bin.
Ranking step loads it from disk — no rebuild needed.

For 100K × 384-dim:
  Index size: ~147 MB
  Query time: ~10ms (exact search, single query vector)
"""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

INDEX_PATH = Path(__file__).parent.parent / "embeddings" / "faiss_index.bin"


def build_index(
    embeddings_matrix: np.ndarray,
    candidate_ids: List[str],
) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat inner-product index from the embeddings matrix.

    Parameters
    ----------
    embeddings_matrix : np.ndarray (N, 384) float32, L2-normalised
    candidate_ids     : list of N candidate_ids in row order

    Returns
    -------
    faiss.IndexFlatIP
    """
    n, dim = embeddings_matrix.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix.astype(np.float32))
    print(f"  [FAISS] Built IndexFlatIP with {index.ntotal} vectors (dim={dim})")
    return index


def save_index(index: faiss.IndexFlatIP) -> None:
    """Persist the FAISS index to disk."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    size_mb = INDEX_PATH.stat().st_size / 1e6
    print(f"  [FAISS] Index saved → {INDEX_PATH.name} ({size_mb:.1f} MB)")


def load_index() -> faiss.IndexFlatIP:
    """Load the FAISS index from disk. Must call build_index + save_index first."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. Run precompute.py first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    print(f"  [FAISS] Loaded index: {index.ntotal} vectors from {INDEX_PATH.name}")
    return index


def retrieve_top_k(
    index: faiss.IndexFlatIP,
    candidate_ids: List[str],
    jd_embedding: np.ndarray,
    k: int = 2000,
) -> List[Tuple[str, float]]:
    """
    Query the FAISS index and return top-K candidates with cosine similarities.

    Parameters
    ----------
    index         : faiss.IndexFlatIP
    candidate_ids : list of candidate_ids in row order (same as when built)
    jd_embedding  : np.ndarray of shape (1, 384), L2-normalised
    k             : number to retrieve (default 2000 — gives headroom for 100K pool)

    Returns
    -------
    List of (candidate_id, cosine_similarity) sorted descending.
    """
    k_actual = min(k, index.ntotal)
    query = jd_embedding.reshape(1, -1).astype(np.float32)

    similarities, indices = index.search(query, k_actual)

    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx == -1:
            continue
        cid = candidate_ids[idx]
        score = float(np.clip(sim, 0.0, 1.0))
        results.append((cid, score))

    print(
        f"  [FAISS] Top-{k_actual} retrieval: "
        f"score range [{results[-1][1]:.4f}, {results[0][1]:.4f}]"
    )
    return results  # Already sorted descending by FAISS
