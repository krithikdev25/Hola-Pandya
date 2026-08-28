"""Runtime wrapper for the trained ML risk model.

The model is a trained logistic-regression probability model.  Its coefficients
and operating threshold are loaded from ml/model_metrics.json, so the runtime
never invents a threshold.  The wrapper also exposes standardized feature
contributions for transparent risk analysis.
"""
import json, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, "ml")
FEATURES = [
    "buyer_risk_score", "buyer_payment_history", "supplier_reliability",
    "production_delay_days", "shipment_delay_days", "payment_delay_days",
    "inventory_age_days", "deterioration", "invoice_dispute_flag",
    "duplicate_financing_flag", "source_confidence", "stage_progress",
    "advance_rate_requested", "financing_utilization", "demand_volatility",
]
_model = _scaler = _metrics = None
_load_attempted = False


def _try_load():
    global _model, _scaler, _metrics, _load_attempted
    if _load_attempted: return
    _load_attempted = True
    try:
        _model = __import__('joblib').load(os.path.join(ML_DIR, 'risk_model.joblib'))
        _scaler = __import__('joblib').load(os.path.join(ML_DIR, 'risk_scaler.joblib'))
        with open(os.path.join(ML_DIR, 'model_metrics.json')) as f: _metrics = json.load(f)
    except Exception:
        _model = _scaler = None
        _metrics = {}


def model_is_loaded(): _try_load(); return _model is not None and _scaler is not None

def get_model_metrics(): _try_load(); return _metrics or {}

def get_operating_threshold():
    _try_load(); return float((_metrics or {}).get('operating_threshold', {}).get('threshold', 0.5))


def _fallback(features):
    # Explicit safety fallback only if artifacts are unavailable. It is never
    # described as the trained model output.
    score = (0.45*features.get('buyer_risk_score', .2) +
             0.20*(1-features.get('buyer_payment_history', .85)) +
             0.18*(1-features.get('supplier_reliability', .85)) +
             0.02*features.get('production_delay_days', 0) +
             0.025*features.get('shipment_delay_days', 0) +
             0.02*features.get('payment_delay_days', 0) +
             0.12*features.get('deterioration', 0) +
             0.15*features.get('invoice_dispute_flag', 0) +
             0.25*features.get('duplicate_financing_flag', 0) -
             0.15*(features.get('source_confidence', 1)-.5))
    return max(.01, min(.99, score))


def predict_risk(features: dict) -> dict:
    _try_load()
    if _model is None or _scaler is None:
        return {'risk_score': round(_fallback(features), 4), 'source': 'fallback_heuristic', 'drivers': [], 'threshold': get_operating_threshold()}
    x = [[float(features.get(f, 0.0)) for f in FEATURES]]
    xs = _scaler.transform(x)
    prob = float(_model.predict_proba(xs)[0][1])
    coefs = _model.coef_[0]
    contributions = []
    for f, z, c in zip(FEATURES, xs[0], coefs):
        contributions.append({'feature': f, 'standardized_value': round(float(z), 3), 'coefficient': round(float(c), 3), 'contribution': round(float(z*c), 3)})
    contributions.sort(key=lambda d: abs(d['contribution']), reverse=True)
    m = _metrics or {}
    low = float(m.get('risk_band_thresholds', {}).get('low_cut', 0.2))
    high = float(m.get('risk_band_thresholds', {}).get('high_cut', 0.5))
    band = 'LOW' if prob < low else ('MODERATE' if prob < high else 'HIGH')
    return {'risk_score': round(prob, 4), 'source': 'ml_model', 'threshold': get_operating_threshold(), 'risk_band': band, 'drivers': contributions[:6]}
