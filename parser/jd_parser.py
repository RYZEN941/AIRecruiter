"""
parser/jd_parser.py
--------------------
Legacy file — not used in the hackathon pipeline.

The JD is hardcoded in scoring/jd_profile.py. This file is kept for
reference only and does not import anthropic.
"""


def parse_jd(jd_text: str) -> dict:
    """
    Stub — returns a minimal dict so any legacy callers don't crash.
    Real JD requirements are in scoring/jd_profile.py.
    """
    return {"raw_text": jd_text}
