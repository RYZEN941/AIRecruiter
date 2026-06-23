"""
embeddings/embedder.py
----------------------
Handles 100K candidate JSONL records for the Redrob hackathon.

Cache invalidation fix (v2):
  A sidecar file candidate_cache_meta.json is written alongside the
  embeddings. It stores the absolute source path and candidate count.
  On the next run, if the source path does not match the requested
  candidates file, the cache is automatically invalidated and rebuilt.
  Use --force to force a rebuild even when paths match.

Model: all-MiniLM-L6-v2
  - 384-dimensional L2-normalised output
  - ~1000 candidates/second in batch mode on CPU
  - 100K candidates ≈ ~100 seconds embedding time

Memory: 100K × 384 × 4 bytes = ~147 MB (well within 16 GB budget)
"""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from scoring.feature_extractor import build_candidate_text
from scoring.jd_profile import JD_EMBED_TEXT

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256

EMBEDDINGS_DIR = Path(__file__).parent
STORE_MATRIX_PATH = EMBEDDINGS_DIR / "candidate_embeddings.npy"
STORE_IDS_PATH    = EMBEDDINGS_DIR / "candidate_ids.json"
STORE_META_PATH   = EMBEDDINGS_DIR / "candidate_cache_meta.json"  # NEW
JD_EMBEDDING_PATH = EMBEDDINGS_DIR / "jd_embedding.npy"

_MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load model singleton."""
    global _MODEL
    if _MODEL is None:
        print(f"  [Embedder] Loading '{MODEL_NAME}' ...")
        _MODEL = SentenceTransformer(MODEL_NAME)
        print(f"  [Embedder] Model ready. dim={EMBEDDING_DIM}")
    return _MODEL


# ── Cache metadata helpers ────────────────────────────────────────────────────

def _save_meta(source_path: str, count: int) -> None:
    """Save sidecar metadata so we can detect stale caches later."""
    meta = {
        "source_path": str(Path(source_path).resolve()),
        "candidate_count": count,
    }
    with open(STORE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _cache_is_valid(candidates_path: str) -> bool:
    """
    Return True only if:
      1. All three artifact files exist (matrix, ids, meta).
      2. The cached source_path matches the requested candidates_path.

    This prevents stale 50-candidate sample artifacts from being silently
    reused when the full 100K dataset is requested.
    """
    if not (STORE_MATRIX_PATH.exists() and
            STORE_IDS_PATH.exists() and
            STORE_META_PATH.exists()):
        return False

    try:
        with open(STORE_META_PATH, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    cached_path = meta.get("source_path", "")
    requested_path = str(Path(candidates_path).resolve())

    if cached_path != requested_path:
        cached_count = meta.get("candidate_count", "?")
        print(
            f"  [Embedder] Cache mismatch — cached from:\n"
            f"               {cached_path}  ({cached_count} candidates)\n"
            f"             Requested:\n"
            f"               {requested_path}\n"
            f"  [Embedder] Invalidating stale cache and rebuilding ..."
        )
        return False

    return True


# ── JD embedding ──────────────────────────────────────────────────────────────

def embed_jd(force: bool = False) -> np.ndarray:
    """
    Embed the JD text and cache to disk.

    Returns np.ndarray of shape (1, 384), float32, L2-normalised.
    """
    if not force and JD_EMBEDDING_PATH.exists():
        jd_emb = np.load(str(JD_EMBEDDING_PATH))
        print(f"  [Embedder] Loaded cached JD embedding from {JD_EMBEDDING_PATH.name}")
        return jd_emb.reshape(1, -1)

    model = get_model()
    emb = model.encode(
        [JD_EMBED_TEXT],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(str(JD_EMBEDDING_PATH), emb)
    print(f"  [Embedder] JD embedding saved -> {JD_EMBEDDING_PATH.name}")
    return emb.reshape(1, -1)


# ── Candidate JSONL streaming ─────────────────────────────────────────────────

def stream_candidates(jsonl_path: str) -> Iterator[dict]:
    """Stream candidates one at a time from a .jsonl file (memory-efficient)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_candidates_gz(gz_path: str) -> Iterator[dict]:
    """Stream candidates from a gzip-compressed .jsonl.gz file."""
    import gzip
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _open_candidates(candidates_path: str) -> Iterator[dict]:
    """Auto-detect .jsonl vs .jsonl.gz and stream accordingly."""
    if candidates_path.endswith(".gz"):
        return stream_candidates_gz(candidates_path)
    return stream_candidates(candidates_path)


# ── Batch embedding ───────────────────────────────────────────────────────────

def embed_candidates(
    candidates_path: str,
    force: bool = False,
    progress_every: int = 5000,
) -> Tuple[np.ndarray, List[str]]:
    """
    Embed all candidates from a JSONL file and save to disk.

    Cache behaviour:
      - Loads from disk only if the cache was built from the SAME source file.
      - If the source path differs (e.g. sample vs. full dataset), the cache
        is automatically invalidated and the full file is re-embedded.
      - Pass force=True to always rebuild regardless of cache state.

    Parameters
    ----------
    candidates_path : path to candidates.jsonl or candidates.jsonl.gz
    force           : if True, re-embed even if a valid cache exists
    progress_every  : print progress every N candidates

    Returns
    -------
    (matrix, ids)
      matrix : np.ndarray (N, 384) float32, L2-normalised
      ids    : list of candidate_ids in matching row order
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    if not force and _cache_is_valid(candidates_path):
        print(f"  [Embedder] Loading cached embeddings from {STORE_MATRIX_PATH.name} ...")
        matrix = np.load(str(STORE_MATRIX_PATH))
        with open(STORE_IDS_PATH, "r") as f:
            ids = json.load(f)
        print(f"  [Embedder] Loaded {len(ids):,} embeddings (dim={matrix.shape[1]})")
        print(f"  [Embedder] Source: {Path(candidates_path).resolve()}")
        return matrix, ids

    # ── Full embedding run ────────────────────────────────────────────────────
    model = get_model()
    abs_path = str(Path(candidates_path).resolve())
    print(f"  [Embedder] Starting full embedding run ...")
    print(f"  [Embedder] Source: {abs_path}")

    all_ids: List[str] = []
    all_texts: List[str] = []

    for i, candidate in enumerate(_open_candidates(candidates_path)):
        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")
        text = build_candidate_text(candidate)
        all_ids.append(cid)
        all_texts.append(text)

        if (i + 1) % progress_every == 0:
            print(f"  [Embedder] Collected {i+1:,} candidates ...")

    total = len(all_ids)
    print(f"  [Embedder] Embedding {total:,} candidates in batches of {BATCH_SIZE} ...")

    all_embeddings = model.encode(
        all_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    ).astype(np.float32)

    # Persist matrix, IDs, and metadata sidecar
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(STORE_MATRIX_PATH), all_embeddings)
    with open(STORE_IDS_PATH, "w") as f:
        json.dump(all_ids, f)
    _save_meta(candidates_path, total)

    size_mb = all_embeddings.nbytes / 1e6
    print(
        f"  [Embedder] Done. {total:,} embeddings ({size_mb:.1f} MB) "
        f"-> {STORE_MATRIX_PATH.name}"
    )
    return all_embeddings, all_ids


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    """
    Load pre-computed embeddings from disk. Must call embed_candidates first.
    """
    if not STORE_MATRIX_PATH.exists() or not STORE_IDS_PATH.exists():
        raise FileNotFoundError(
            "Candidate embeddings not found. Run precompute.py first.\n"
            f"  Expected: {STORE_MATRIX_PATH}\n"
            f"            {STORE_IDS_PATH}"
        )
    matrix = np.load(str(STORE_MATRIX_PATH))
    with open(STORE_IDS_PATH, "r") as f:
        ids = json.load(f)
    return matrix, ids
