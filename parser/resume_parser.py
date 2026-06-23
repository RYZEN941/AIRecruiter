"""
parser/resume_parser.py
------------------------
Legacy file — not used in the hackathon pipeline.

Candidates are now pre-structured JSON records (candidates.jsonl).
Feature extraction is handled by scoring/feature_extractor.py.
This file is kept for reference only and does not import anthropic.
"""


def parse_resume(resume_text: str) -> dict:
    """
    Stub — returns minimal dict so any legacy callers don't crash.
    Real extraction is in scoring/feature_extractor.py.
    """
    return {"raw_text": resume_text}
