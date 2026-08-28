"""
Multi-source reconciliation (PS6 requirement #8 / Annexure "Multi-Source
Reconciliation").

Deterministic, explainable, weighted-scoring reconciliation -- NOT machine
learning, NOT "pick whichever arrived last". Justification for the formula:

    confidence(reading) = source_authority_weight * freshness_decay(age)

- source_authority_weight: how trustworthy this source is for this kind of
  field, in general (e.g. a warehouse scan is more authoritative about
  physical location than an ERP estimate).
- freshness_decay: an exponential decay so a stale reading is discounted
  even from an authoritative source -- old evidence should count for less.

When two readings for the same field disagree beyond `tolerance`, the
higher-confidence reading wins and a CONFLICT is logged. If BOTH
confidences fall below `LOW_CONFIDENCE_THRESHOLD`, the field is marked
'unverified' -- the decision engine treats 'unverified' fields more
conservatively (see engine/decision_engine.py: risk_multiplier). Nothing is
ever silently trusted by default.
"""
import math
import time

SOURCE_AUTHORITY = {
    "erp": 0.6,
    "production_system": 0.8,
    "logistics": 0.85,
    "warehouse": 0.9,
    "iot": 0.75,
    "financial_system": 0.9,
    "invoice_system": 0.85,
}

LOW_CONFIDENCE_THRESHOLD = 0.35
DECAY_HALF_LIFE_SECONDS = 6 * 3600  # a reading loses half its weight every 6h


def freshness_decay(age_seconds: float) -> float:
    age_seconds = max(0.0, age_seconds)
    return 0.5 ** (age_seconds / DECAY_HALF_LIFE_SECONDS)


def confidence(source: str, timestamp: float, now: float = None) -> float:
    now = now if now is not None else time.time()
    weight = SOURCE_AUTHORITY.get(source, 0.5)
    return round(weight * freshness_decay(now - timestamp), 4)


def reconcile_field(readings: list, tolerance: float = 0.05, now: float = None):
    """readings: list of dicts {value, source, timestamp}, all claiming to
    describe the SAME field at roughly the same time.
    Returns (chosen_value, status, log_entry) where status is one of
    'AGREED', 'RESOLVED_BY_CONFIDENCE', 'UNVERIFIED'."""
    if not readings:
        return None, "NO_DATA", None
    if len(readings) == 1:
        r = readings[0]
        return r["value"], "AGREED", {
            "type": "SINGLE_SOURCE", "value": r["value"], "source": r["source"],
            "confidence": confidence(r["source"], r["timestamp"], now),
        }

    scored = [
        {**r, "confidence": confidence(r["source"], r["timestamp"], now)}
        for r in readings
    ]
    scored.sort(key=lambda r: r["confidence"], reverse=True)
    values = [r["value"] for r in scored]

    def _disagree(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) > tolerance
        return a != b

    if all(not _disagree(v, values[0]) for v in values):
        return values[0], "AGREED", {
            "type": "AGREEMENT", "value": values[0],
            "sources": [r["source"] for r in scored],
        }

    top = scored[0]
    if top["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return top["value"], "UNVERIFIED", {
            "type": "CONFLICT_LOW_CONFIDENCE",
            "readings": scored,
            "chosen": top["value"],
            "reason": "Sources disagree and even the best confidence is below threshold; "
                      "state marked unverified rather than silently trusted.",
        }

    return top["value"], "RESOLVED_BY_CONFIDENCE", {
        "type": "CONFLICT_DETECTED",
        "readings": scored,
        "chosen": top["value"],
        "chosen_source": top["source"],
        "reason": f"{top['source']} had highest confidence ({top['confidence']}) "
                  f"among disagreeing sources.",
    }
