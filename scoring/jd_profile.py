"""
scoring/jd_profile.py
---------------------
Hardcoded structured representation of the Redrob hackathon JD.

Instead of calling Claude at runtime to parse the JD (which costs time and API
calls), we manually encode what the JD says and — critically — what it MEANS.
The JD explicitly warns that keyword matching is a trap. This profile is our
interpretation of the intent.

DO NOT use this as a simple keyword list.  The fields here drive the *weights*
and *reasoning* in downstream scorers, not just string matching.
"""

from datetime import date

# ─── JD text (used for embedding the JD itself) ──────────────────────────────
JD_EMBED_TEXT = """
Senior AI Engineer founding team Redrob AI Series A talent intelligence platform.
Production experience embeddings-based retrieval systems sentence-transformers
vector databases hybrid search infrastructure FAISS Pinecone Weaviate Milvus
Elasticsearch ranking evaluation NDCG MRR MAP Python NLP information retrieval
LLM fine-tuning LoRA QLoRA learning-to-rank recommendation systems search
product company startup shipping real users scale infrastructure.
5 to 9 years applied machine learning AI product company production deployment.
Pune Noida Hyderabad Mumbai Delhi NCR India hybrid.
"""

# ─── Required skills (hard signals) ──────────────────────────────────────────
# These are the skills the JD explicitly marks as absolute requirements.
# Skills listed as aliases will all match the same bucket.
REQUIRED_SKILL_BUCKETS = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers",
        "openai embeddings", "text embeddings",
        "bge", "e5", "embeddings", "embedding",
        "dense retrieval", "bi-encoder", "cross-encoder",
        "semantic search", "semantic similarity",
    ],
    "vector_db": [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "pgvector", "chroma",
        "vector database", "vector store", "vector search",
        "ann", "approximate nearest neighbor",
    ],
    "hybrid_search": [
        "bm25", "hybrid search", "hybrid retrieval",
        "sparse retrieval", "dense sparse", "reciprocal rank fusion",
        "rrf", "inverted index",
    ],
    "python_strong": [
        "python",
    ],
    "ranking_evaluation": [
        "ndcg", "mrr", "map", "mean average precision",
        "ranking evaluation", "a/b test", "a/b testing",
        "offline evaluation", "online evaluation",
        "information retrieval", "ir metrics", "recall@k",
        "precision@k", "learning to rank",
    ],
    "nlp_ir": [
        "nlp", "natural language processing",
        "information retrieval", "text mining",
        "named entity recognition", "ner",
        "transformers", "huggingface", "bert", "roberta",
        "llm", "large language model", "rag",
        "retrieval augmented generation",
    ],
}

# ─── Preferred skills (bonus signals, not blocking) ──────────────────────────
PREFERRED_SKILL_BUCKETS = {
    "llm_finetuning": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning",
        "instruction tuning", "rlhf", "dpo",
    ],
    "learning_to_rank": [
        "learning to rank", "ltr", "lambdarank", "listwise",
        "xgboost ranking", "neural ranking",
    ],
    "hrtech_marketplace": [
        "hr tech", "hrtech", "recruiting", "talent",
        "marketplace", "two-sided marketplace",
        "recommendation system", "recommender system",
    ],
    "distributed_inference": [
        "distributed training", "distributed inference",
        "model serving", "triton", "tensorrt", "onnx",
        "quantization", "distillation", "vllm",
        "ray", "spark ml", "kubernetes ml",
    ],
    "open_source": [
        "open source", "open-source", "github contributions",
        "oss contributor", "maintainer",
    ],
}

# ─── Disqualifying company patterns ──────────────────────────────────────────
# JD explicitly names these as "we've had bad fit experiences" if the candidate
# has worked ONLY at these companies their entire career.
CONSULTING_FIRM_KEYWORDS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "zensar", "niit technologies",
    "l&t infotech", "ltimindtree",
]

# ─── Disqualifying domain patterns ──────────────────────────────────────────
# Computer vision / speech / robotics WITHOUT NLP/IR signals
CV_ONLY_SKILL_SIGNALS = [
    "computer vision", "object detection", "image classification",
    "image segmentation", "yolo", "cnn", "convolutional",
    "pose estimation", "optical flow",
]
SPEECH_ONLY_SKILL_SIGNALS = [
    "speech recognition", "asr", "tts", "text to speech",
    "speaker diarization", "speech synthesis", "wake word",
]
# If a candidate has ONLY these domains (no NLP/IR signals) → strong penalty

# ─── Location targets ─────────────────────────────────────────────────────────
# Tier 1: Exactly where they want
LOCATION_TIER_1 = ["pune", "noida"]
# Tier 2: Acceptable with relocation
LOCATION_TIER_2 = ["delhi", "ncr", "gurugram", "gurgaon", "hyderabad", "mumbai",
                    "bengaluru", "bangalore", "delhi ncr"]
# Tier 3: India but further — relocation required
LOCATION_TIER_3_COUNTRY = "india"

# ─── Experience target ────────────────────────────────────────────────────────
EXPERIENCE_TARGET_MIN = 5.0   # years
EXPERIENCE_TARGET_MAX = 9.0   # years
EXPERIENCE_TARGET_SOFT_MIN = 4.0   # still consider with strong signals
EXPERIENCE_TARGET_SOFT_MAX = 12.0  # still consider if depth is strong

# ─── Availability expectations ────────────────────────────────────────────────
NOTICE_PERIOD_IDEAL_DAYS = 30    # JD says "love sub-30-day"
NOTICE_PERIOD_ACCEPTABLE_DAYS = 90   # buyout up to 30, rest flexible

# ─── The "ideal candidate" profile (used to generate reasoning) ──────────────
IDEAL_PROFILE = {
    "experience_years": "6-8",
    "product_company_years": "4-5",
    "has_shipped_ranking_or_search": True,
    "location_preferred": ["Pune", "Noida"],
    "active_on_platform": True,
    "notice_period_days_max": 30,
}

# ─── Score budget allocation (must sum to 1.0 before penalties) ──────────────
SIGNAL_WEIGHTS = {
    "semantic":      0.25,   # FAISS cosine similarity vs JD embedding
    "title_fit":     0.20,   # current title / career trajectory tier
    "skill_depth":   0.20,   # depth-weighted skill match (proficiency + duration)
    "experience":    0.10,   # years proximity to 5-9 range
    "availability":  0.15,   # behavioral signals composite
    "location":      0.05,   # location match
    "llm_bonus":     0.05,   # optional LLM pre-score if cached (else 0→redistributed)
}
# Disqualifier penalties are multiplicative (applied after weighted sum)
