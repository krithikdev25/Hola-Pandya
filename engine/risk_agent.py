"""Autonomous risk-monitoring agent.

This is an agent in the engineering sense: it observes the current asset,
consults the trained probability model, measures change versus the prior
state, evaluates exposure, and emits an actionable risk assessment.  No LLM
is required.  The model remains the statistical component; policy remains
explicit and auditable.
"""
from engine.risk_model import predict_risk, get_operating_threshold


def assess(asset, features, portfolio_exposure=0.0):
    result = predict_risk(features)
    current = result['risk_score']
    previous = asset.risk.last_model_probability
    velocity = round(current - previous, 4)
    exposure_ratio = (asset.financial.total_exposure / asset.financial.exposure_limit
                      if asset.financial.exposure_limit else 0.0)
    threshold = result.get('threshold', get_operating_threshold())
    hard_stop = asset.physical.condition_flag == 'DAMAGED' or features.get('duplicate_financing_flag', 0) >= 1
    if hard_stop:
        recommendation = 'BLOCK_NEW_FINANCE'
    elif current >= threshold and velocity > 0.03:
        recommendation = 'REDUCE_OR_FREEZE'
    elif current >= threshold:
        recommendation = 'ENHANCED_MONITORING'
    elif velocity > 0.05:
        recommendation = 'WATCH_RISK_ACCELERATION'
    else:
        recommendation = 'CONTINUE'
    asset.risk.risk_velocity = velocity
    asset.risk.last_model_probability = current
    asset.risk.risk_band = result.get('risk_band', 'MODERATE')
    return {
        'model_probability': current,
        'threshold': threshold,
        'risk_band': asset.risk.risk_band,
        'risk_velocity': velocity,
        'exposure_ratio': round(exposure_ratio, 4),
        'portfolio_exposure': round(portfolio_exposure, 2),
        'recommendation': recommendation,
        'drivers': result.get('drivers', []),
        'model_source': result.get('source'),
    }
