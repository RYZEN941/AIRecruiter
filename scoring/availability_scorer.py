"""
scoring/availability_scorer.py
-------------------------------
Computes a composite availability score from the 23 Redrob behavioral signals.

The JD explicitly states:
  "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
   recruiter response rate is, for hiring purposes, not actually available."

This module converts raw signal values into a single availability multiplier
in [0, 1] that is applied on top of the skill/experience score.

Signal components:
  - open_to_work_flag
  - days_since_last_active (derived from last_active_date)
  - recruiter_response_rate
  - avg_response_time_hours
  - interview_completion_rate
  - notice_period_days (not pure availability, but affects hiring timeline)
  - offer_acceptance_rate (reliability signal)
  - verified_email + verified_phone (trust)
"""

from datetime import date, datetime
from typing import Optional


def _days_since(date_str: str | None) -> int:
    """Return integer days since the given date string (YYYY-MM-DD)."""
    if not date_str:
        return 999  # Unknown = treat as very stale
    try:
        last_dt = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - last_dt).days
    except (ValueError, TypeError):
        return 999


def _score_open_to_work(flag: bool) -> float:
    """
    open_to_work_flag.
    False doesn't mean unavailable — they may still respond — but it's a
    strong negative signal. Score 0.4 rather than 0.0 to avoid hard zeroing.
    """
    return 1.0 if flag else 0.4


def _score_recency(days_since_active: int) -> float:
    """
    Recency of last login.
    <14 days  = very active     1.00
    14-30     = active          0.90
    30-60     = recent          0.80
    60-90     = cooling         0.65
    90-180    = stale           0.45
    >180      = very stale      0.20
    """
    if days_since_active < 14:
        return 1.00
    elif days_since_active < 30:
        return 0.90
    elif days_since_active < 60:
        return 0.80
    elif days_since_active < 90:
        return 0.65
    elif days_since_active < 180:
        return 0.45
    else:
        return 0.20


def _score_response_rate(rate: float) -> float:
    """
    Fraction of recruiter messages replied to.
    <0.10 = almost never responds (effectively unavailable)
    0.10-0.25 = low responsiveness
    0.25-0.50 = moderate
    >0.50 = good
    """
    if rate < 0.10:
        return 0.30
    elif rate < 0.25:
        return 0.55
    elif rate < 0.50:
        return 0.80
    else:
        return 1.00


def _score_response_time(hours: float) -> float:
    """
    Median response time to recruiter messages.
    <4h   = excellent
    4-24h = good
    24-72h = ok
    72-168h = slow
    >168h  = very slow (1 week+)
    """
    if hours < 4:
        return 1.00
    elif hours < 24:
        return 0.90
    elif hours < 72:
        return 0.75
    elif hours < 168:
        return 0.55
    else:
        return 0.35


def _score_interview_completion(rate: float) -> float:
    """
    Fraction of scheduled interviews actually attended.
    High no-show rate = reliability concern.
    """
    if rate >= 0.90:
        return 1.00
    elif rate >= 0.75:
        return 0.90
    elif rate >= 0.60:
        return 0.75
    elif rate >= 0.40:
        return 0.55
    else:
        return 0.35


def _score_notice_period(days: int) -> float:
    """
    Notice period scoring aligned with JD preference:
    JD wants sub-30-day; can buy out 30 days; 30+ still OK but bar is higher.
    This is a hiring-timeline signal, not pure availability.
    <30    = ideal     1.00
    30-60  = good      0.85
    60-90  = ok        0.70
    90-120 = long      0.55
    >120   = very long 0.40
    """
    if days < 30:
        return 1.00
    elif days < 60:
        return 0.85
    elif days < 90:
        return 0.70
    elif days < 120:
        return 0.55
    else:
        return 0.40


def _score_offer_acceptance(rate: float) -> float:
    """
    Historical offer acceptance rate.
    -1 means no history (neutral).
    Low acceptance = waste of recruiter time.
    """
    if rate == -1:
        return 0.70  # No history — neutral-ish, slight discount
    elif rate >= 0.70:
        return 1.00
    elif rate >= 0.50:
        return 0.85
    elif rate >= 0.30:
        return 0.65
    else:
        return 0.45


def _score_verification(verified_email: bool, verified_phone: bool) -> float:
    """Trust multiplier — both verified = 1.0, neither = 0.8."""
    if verified_email and verified_phone:
        return 1.00
    elif verified_email or verified_phone:
        return 0.90
    else:
        return 0.80


def compute_availability_score(signals: dict) -> dict:
    """
    Compute the composite availability score from redrob_signals.

    Parameters
    ----------
    signals : dict — the redrob_signals object from the candidate record

    Returns
    -------
    dict with:
        availability_score : float [0, 1]
        components         : dict of individual sub-scores
        notice_period_days : int  (surfaced for reasoning generation)
    """
    open_to_work   = bool(signals.get("open_to_work_flag", False))
    last_active    = signals.get("last_active_date")
    response_rate  = float(signals.get("recruiter_response_rate", 0.5))
    response_time  = float(signals.get("avg_response_time_hours", 48))
    icr            = float(signals.get("interview_completion_rate", 0.7))
    notice_days    = int(signals.get("notice_period_days", 60))
    offer_acc      = float(signals.get("offer_acceptance_rate", -1))
    ver_email      = bool(signals.get("verified_email", False))
    ver_phone      = bool(signals.get("verified_phone", False))

    days_inactive = _days_since(last_active)

    components = {
        "open_to_work":         _score_open_to_work(open_to_work),
        "recency":              _score_recency(days_inactive),
        "response_rate":        _score_response_rate(response_rate),
        "response_time":        _score_response_time(response_time),
        "interview_completion": _score_interview_completion(icr),
        "notice_period":        _score_notice_period(notice_days),
        "offer_acceptance":     _score_offer_acceptance(offer_acc),
        "verification":         _score_verification(ver_email, ver_phone),
    }

    # Weighted average — recency and response_rate are most important
    weights = {
        "open_to_work":         0.20,
        "recency":              0.25,
        "response_rate":        0.20,
        "response_time":        0.10,
        "interview_completion": 0.10,
        "notice_period":        0.05,
        "offer_acceptance":     0.05,
        "verification":         0.05,
    }

    availability_score = sum(
        components[k] * weights[k] for k in weights
    )

    return {
        "availability_score": round(availability_score, 4),
        "components": components,
        "days_since_active": days_inactive,
        "notice_period_days": notice_days,
        "open_to_work": open_to_work,
        "recruiter_response_rate": response_rate,
    }
