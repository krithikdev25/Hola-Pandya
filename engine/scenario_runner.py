"""
Loads a scenario JSON, seeds an Asset from it, plays every event through
reassess(), and returns the full trace. Shared by tests, the CLI audit
script, and the API's /scenarios/{name}/run endpoint -- one code path,
not reimplemented three times.
"""
import json
import os

from engine.models import Asset, FinancialState, ContractualState, FinancingInstrument
from engine.decision_engine import reassess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN_DIR = os.path.join(BASE_DIR, "data", "scenarios")


def load_scenario(name_or_path: str) -> dict:
    path = name_or_path if os.path.isabs(name_or_path) else os.path.join(SCEN_DIR, name_or_path)
    if not path.endswith(".json"):
        path += ".json"
    with open(path) as f:
        return json.load(f)


def seed_asset_from_scenario(scen: dict) -> Asset:
    seed = scen["asset_seed"]
    value = seed["quantity"] * seed["agreed_price"]
    asset = Asset(id=scen["asset_id"], product_name=seed["product_name"], quantity=seed["quantity"])
    asset.contractual = ContractualState(
        po_id=seed["po_id"], buyer_id=seed["buyer_id"], supplier_id=seed.get("supplier_id", "supplier_unknown"),
        lender_id=seed.get("lender_id", "lender_prime"), agreed_price=seed["agreed_price"],
        buyer_risk_score=seed.get("buyer_risk_score_override", 0.2),
        buyer_payment_history=seed.get("buyer_payment_history_override", 0.85),
        supplier_reliability=seed.get("supplier_reliability_override", 0.85),
    )
    exposure_limit = seed.get("exposure_limit_override", value * 0.9)
    existing_exposure = seed.get("existing_exposure_override", 0.0)
    asset.financial = FinancialState(
        estimated_value=value, exposure_limit=exposure_limit, total_exposure=existing_exposure,
    )
    if existing_exposure > 0:
        asset.financial.existing_financing.append(
            {"instrument": "PRIOR_FINANCING", "amount": existing_exposure, "ts": 0}
        )
        # If this scenario seeds pre-existing exposure, mark an instrument as
        # already active -- otherwise the engine would (correctly, but
        # confusingly for a demo) label the next decision "INITIATE" instead
        # of "INCREASE/REDUCE/TRANSITION" against the already-outstanding amount.
        instrument_name = seed.get("existing_instrument", "INVENTORY_FINANCING")
        asset.financial.current_instrument = FinancingInstrument(instrument_name)
    return asset


def run_scenario(name_or_path: str) -> dict:
    scen = load_scenario(name_or_path)
    asset = seed_asset_from_scenario(scen)
    trace = []
    for event in scen["events"]:
        entry = reassess(asset, event)
        trace.append(entry)
    return {"scenario": scen["name"], "asset_id": asset.id, "final_state": asset.to_dict(), "trace": trace}
