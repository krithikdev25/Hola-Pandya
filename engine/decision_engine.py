"""
The financial decision engine. This is the actual product, per PS6
Section 15: "The backend/decision engine is the actual product," not a
CRUD layer feeding a dashboard.

Every call to reassess() runs the full autonomous loop:
    OBSERVE (event) -> REASON (recompute risk/value from state) ->
    DECIDE (amount, instrument, action) -> ACT (mutate asset state) ->
    (MONITOR/REASSESS happens on the next event, automatically)

No LLM, no external API, no human approval step. This is what "agentic"
means for this project (see PS6 Section 4 / Annexure "agentic behaviour
without AI").
"""
import time
import uuid
from dataclasses import asdict

from engine.models import (
    Asset, LifecycleStage, EventType, FinancingInstrument, DecisionAction,
)
from engine.state_machine import guard_transition, InvalidTransitionError
from engine.reconciliation import reconcile_field, confidence as calc_confidence
from engine.risk_agent import assess as assess_risk_agent

STAGE_ORDER = [
    LifecycleStage.PO_CREATED, LifecycleStage.MATERIAL_RECEIVED, LifecycleStage.PRODUCTION,
    LifecycleStage.FINISHED_GOODS, LifecycleStage.SHIPPED, LifecycleStage.WAREHOUSE,
    LifecycleStage.DELIVERED, LifecycleStage.INVOICED,
]


def _stage_progress(stage: LifecycleStage) -> float:
    base = {
        LifecycleStage.PRODUCTION_DELAYED: LifecycleStage.PRODUCTION,
        LifecycleStage.SHIPMENT_DELAYED: LifecycleStage.SHIPPED,
        LifecycleStage.PAYMENT_DELAYED: LifecycleStage.INVOICED,
        LifecycleStage.SETTLED: LifecycleStage.INVOICED,
    }.get(stage, stage)
    if base not in STAGE_ORDER:
        return 0.0
    return STAGE_ORDER.index(base) / (len(STAGE_ORDER) - 1)


def _assess_risk(asset: Asset, portfolio_exposure: float = 0.0) -> dict:
    features = {
        "buyer_risk_score": asset.contractual.buyer_risk_score,
        "buyer_payment_history": asset.contractual.buyer_payment_history,
        "supplier_reliability": asset.contractual.supplier_reliability,
        "production_delay_days": asset.physical.production_delay_days,
        "shipment_delay_days": asset.physical.shipment_delay_days,
        "payment_delay_days": asset.contractual.payment_delay_days,
        "inventory_age_days": asset.physical.inventory_age_days,
        "deterioration": 1.0 if asset.physical.condition_flag != "NORMAL" else 0.0,
        "invoice_dispute_flag": 1.0 if asset.contractual.invoice_dispute_flag else 0.0,
        "duplicate_financing_flag": 1.0 if any("DUPLICATE" in str(x).upper() for x in asset.risk.risk_factors) else 0.0,
        "source_confidence": asset.physical.confidence,
        "stage_progress": _stage_progress(asset.physical.stage),
        "advance_rate_requested": BASE_ADVANCE_RATE.get(asset.physical.stage, 0.0),
        "financing_utilization": (asset.financial.total_exposure / asset.financial.estimated_value
                                   if asset.financial.estimated_value else 0.0),
        "demand_volatility": 0.0,
    }
    return assess_risk_agent(asset, features, portfolio_exposure)

# --- Deterministic parameters (all explicit, all testable, all overridable) ---

BASE_ADVANCE_RATE = {
    LifecycleStage.PO_CREATED: 0.30,
    LifecycleStage.MATERIAL_RECEIVED: 0.35,
    LifecycleStage.PRODUCTION: 0.50,
    LifecycleStage.PRODUCTION_DELAYED: 0.35,   # risk goes up, so headline rate drops
    LifecycleStage.FINISHED_GOODS: 0.65,
    LifecycleStage.SHIPPED: 0.70,
    LifecycleStage.SHIPMENT_DELAYED: 0.55,
    LifecycleStage.WAREHOUSE: 0.72,
    LifecycleStage.DELIVERED: 0.80,
    LifecycleStage.INVOICED: 0.85,
    LifecycleStage.PAYMENT_DELAYED: 0.70,
    LifecycleStage.SETTLED: 0.0,
}

INSTRUMENT_BY_STAGE = {
    LifecycleStage.PO_CREATED: FinancingInstrument.PO_ADVANCE,
    LifecycleStage.MATERIAL_RECEIVED: FinancingInstrument.PO_ADVANCE,
    LifecycleStage.PRODUCTION: FinancingInstrument.INVENTORY_FINANCING,
    LifecycleStage.PRODUCTION_DELAYED: FinancingInstrument.INVENTORY_FINANCING,
    LifecycleStage.FINISHED_GOODS: FinancingInstrument.INVENTORY_FINANCING,
    LifecycleStage.SHIPPED: FinancingInstrument.INVENTORY_FINANCING,
    LifecycleStage.SHIPMENT_DELAYED: FinancingInstrument.INVENTORY_FINANCING,
    LifecycleStage.WAREHOUSE: FinancingInstrument.INVENTORY_FINANCING,
    LifecycleStage.DELIVERED: FinancingInstrument.RECEIVABLES_FINANCING,
    LifecycleStage.INVOICED: FinancingInstrument.RECEIVABLES_FINANCING,
    LifecycleStage.PAYMENT_DELAYED: FinancingInstrument.RECEIVABLES_FINANCING,
    LifecycleStage.SETTLED: FinancingInstrument.NONE,
}

# Event -> target lifecycle stage (only events that move the physical stage)
EVENT_STAGE_MAP = {
    EventType.PO_CREATED: LifecycleStage.PO_CREATED,
    EventType.MATERIAL_RECEIVED: LifecycleStage.MATERIAL_RECEIVED,
    EventType.PRODUCTION_STARTED: LifecycleStage.PRODUCTION,
    EventType.PRODUCTION_PROGRESS: LifecycleStage.PRODUCTION,
    EventType.PRODUCTION_DELAYED: LifecycleStage.PRODUCTION_DELAYED,
    EventType.PRODUCTION_COMPLETED: LifecycleStage.FINISHED_GOODS,
    EventType.SHIPMENT_CREATED: LifecycleStage.SHIPPED,
    EventType.SHIPMENT_DELAYED: LifecycleStage.SHIPMENT_DELAYED,
    EventType.WAREHOUSE_RECEIVED: LifecycleStage.WAREHOUSE,
    EventType.DELIVERY_CONFIRMED: LifecycleStage.DELIVERED,
    EventType.INVOICE_ISSUED: LifecycleStage.INVOICED,
    EventType.PAYMENT_DELAYED: LifecycleStage.PAYMENT_DELAYED,
    EventType.PAYMENT_RECEIVED: LifecycleStage.SETTLED,
}


def risk_multiplier(risk_score: float, confidence_val: float, delay_flag: bool,
                     condition_flag: str, buyer_risk_score: float) -> float:
    """Deterministic multiplier in [0.1, 1.0] applied to the base advance
    rate. Every factor here is named and testable -- this is the whole
    'reasoning' step, not a black box."""
    m = 1.0
    m *= (0.5 + 0.5 * confidence_val)          # low confidence -> shrink advance
    m *= (1.0 - 0.5 * risk_score)               # asset risk -> shrink advance
    m *= (1.0 - 0.3 * buyer_risk_score)         # buyer risk -> shrink advance
    if delay_flag:
        m *= 0.8
    if condition_flag == "DETERIORATED":
        m *= 0.6
    elif condition_flag == "DAMAGED":
        m *= 0.2
    return max(0.1, min(1.0, round(m, 4)))


def compute_financing_amount(asset: Asset) -> dict:
    """Returns the risk-adjusted TARGET total financing for this asset right
    now, both uncapped (eligible_amount: what the risk/value math alone
    would support) and capped at the exposure limit (capped_target: what the
    system will actually allow outstanding). Both numbers are absolute
    totals, not deltas -- the caller (reassess) compares capped_target
    against the asset's CURRENT total_exposure to decide whether to
    initiate/increase/reduce/hold, and logs the gap between eligible_amount
    and capped_target as the explicit "over-leverage prevented" evidence
    whenever the cap actually bites."""
    stage = asset.physical.stage
    base_rate = BASE_ADVANCE_RATE.get(stage, 0.0)
    rm = risk_multiplier(
        asset.risk.risk_score, asset.physical.confidence,
        asset.physical.delay_flag, asset.physical.condition_flag,
        asset.contractual.buyer_risk_score,
    )
    effective_rate = round(base_rate * rm, 4)
    eligible_amount = round(asset.financial.estimated_value * effective_rate, 2)

    # Explicit knockout rule (a hard stop layered on top of the continuous
    # risk multiplier, the way real underwriting combines a scorecard with
    # hard disqualifiers): a DAMAGED asset is not financeable regardless of
    # what the continuous risk score says. The continuous multiplier alone
    # has a floor of 0.1 (never fully zero, by design, so a merely risky-but-
    # intact asset is still financeable at a reduced rate) -- this rule is
    # what actually lets the system reach a genuine REJECT for catastrophic
    # physical risk.
    knockout = asset.physical.condition_flag == "DAMAGED"
    if knockout:
        eligible_amount = 0.0
    capped_target = round(max(0.0, min(eligible_amount, asset.financial.exposure_limit)), 2)
    capped_by_exposure_limit = eligible_amount > asset.financial.exposure_limit
    return {
        "base_rate": base_rate, "risk_multiplier": rm, "effective_rate": effective_rate,
        "eligible_amount": eligible_amount, "capped_target": capped_target,
        "capped_by_exposure_limit": capped_by_exposure_limit, "knockout": knockout,
    }


def _log(asset: Asset, event: dict, prior_stage, new_stage, decision, amount,
          instrument, reasons, action, extra=None):
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.time(),
        "event": event,
        "prior_stage": prior_stage.value if prior_stage else None,
        "new_stage": new_stage.value if new_stage else None,
        "decision": decision.value,
        "amount": amount,
        "instrument": instrument.value if instrument else None,
        "reasons": reasons,
        "action": action,
    }
    if extra:
        entry["reconciliation"] = extra
    asset.audit_log.append(entry)
    return entry


def reassess(asset: Asset, event: dict) -> dict:
    """The core autonomous loop. `event` = {type, payload, source, timestamp}.
    Mutates `asset` in place and returns the audit log entry describing what
    happened. This function is called automatically whenever an event
    arrives -- there is no human-approval step in this path."""
    reasons = []
    prior_stage = asset.physical.stage
    etype = EventType(event["type"])
    payload = event.get("payload", {})
    source = event.get("source", "unknown")
    ts = event.get("timestamp", time.time())

    # ---- 1. Handle multi-source reconciliation, if this event carries
    #         a conflicting reading against the asset's current physical field ----
    reconciliation_log = None
    if etype in (EventType.PRODUCTION_PROGRESS,) and "production_pct" in payload:
        readings = [
            {"value": asset.physical.production_pct, "source": asset.physical.source,
             "timestamp": asset.physical.last_updated},
            {"value": payload["production_pct"], "source": source, "timestamp": ts},
        ]
        chosen, status, reconciliation_log = reconcile_field(readings, tolerance=5.0)
        asset.physical.production_pct = chosen
        asset.physical.confidence = 1.0 if status == "AGREED" else (
            0.5 if status == "RESOLVED_BY_CONFIDENCE" else 0.25
        )
        if status != "AGREED":
            reasons.append(f"Reconciliation: {status} — {reconciliation_log.get('reason', '')}")
    else:
        asset.physical.confidence = calc_confidence(source, ts)

    # ---- 2. Apply the physical stage transition (guarded) ----
    target_stage = EVENT_STAGE_MAP.get(etype, asset.physical.stage)
    try:
        guard_transition(prior_stage, target_stage)
        asset.physical.stage = target_stage
    except InvalidTransitionError as e:
        reasons.append(str(e))
        entry = _log(asset, event, prior_stage, prior_stage, DecisionAction.REJECT,
                     0.0, asset.financial.current_instrument, reasons,
                     "Event rejected: illegal state transition.", reconciliation_log)
        return entry

    asset.physical.last_updated = ts
    asset.physical.source = source
    asset.physical.delay_flag = etype in (EventType.PRODUCTION_DELAYED, EventType.SHIPMENT_DELAYED,
                                           EventType.PAYMENT_DELAYED)

    # ---- 3. Update physical/contractual side effects from the event ----
    # (these feed the risk MODEL below as features; they no longer bump
    # risk_score directly -- the trained model is the single place that
    # turns these observations into a risk_score.)
    if etype == EventType.DETERIORATION_DETECTED:
        asset.physical.condition_flag = payload.get("condition", "DETERIORATED")
        if "deterioration_detected" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("deterioration_detected")
        reasons.append("Deterioration detected -> condition flag set (feeds risk model).")
    if etype == EventType.BUYER_RISK_CHANGED:
        asset.contractual.buyer_risk_score = payload.get("buyer_risk_score", asset.contractual.buyer_risk_score)
        reasons.append(f"Buyer risk updated to {asset.contractual.buyer_risk_score} (feeds risk model).")
    if etype == EventType.PRODUCTION_DELAYED:
        asset.physical.production_delay_days += payload.get("delay_days", 3.0)
        if "production_delayed" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("production_delayed")
        reasons.append(f"Production delay recorded: {asset.physical.production_delay_days} total days (feeds risk model).")
    if etype == EventType.SHIPMENT_DELAYED:
        asset.physical.shipment_delay_days += payload.get("delay_days", 2.0)
        if "shipment_delayed" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("shipment_delayed")
        reasons.append(f"Shipment delay recorded: {asset.physical.shipment_delay_days} total days (feeds risk model).")
    if etype == EventType.PAYMENT_DELAYED:
        if "payment_delayed" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("payment_delayed")
        reasons.append("Payment delay recorded (feeds risk model via buyer/contractual signal).")
    if etype == EventType.INVOICE_ISSUED:
        asset.contractual.invoice_id = payload.get("invoice_id", asset.contractual.invoice_id)
    if etype == EventType.INVOICE_DISPUTED:
        asset.contractual.invoice_dispute_flag = True
        if "invoice_dispute" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("invoice_dispute")
        reasons.append("Invoice dispute detected -> receivable quality downgraded.")
    if etype == EventType.SUPPLIER_RISK_CHANGED:
        asset.contractual.supplier_reliability = float(payload.get("supplier_reliability", asset.contractual.supplier_reliability))
        reasons.append(f"Supplier reliability updated to {asset.contractual.supplier_reliability:.2f}.")
    if etype == EventType.BUYER_RISK_CHANGED:
        asset.contractual.buyer_risk_score = float(payload.get("buyer_risk_score", asset.contractual.buyer_risk_score))
        reasons.append(f"Buyer risk updated to {asset.contractual.buyer_risk_score:.2f}.")
    if etype == EventType.INVENTORY_AGING:
        asset.physical.inventory_age_days = float(payload.get("inventory_age_days", asset.physical.inventory_age_days))
        reasons.append(f"Inventory age updated to {asset.physical.inventory_age_days:.1f} days.")
    if etype == EventType.DUPLICATE_FINANCING_ALERT:
        if "DUPLICATE_FINANCING" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("DUPLICATE_FINANCING")
        reasons.append("Duplicate-financing signal detected -> new financing is blocked pending resolution.")
    if etype == EventType.LOCATION_MISMATCH:
        asset.physical.location_verified = False
        asset.physical.confidence = min(asset.physical.confidence, 0.35)
        if "location_mismatch" not in asset.risk.risk_factors:
            asset.risk.risk_factors.append("location_mismatch")
        reasons.append("Location mismatch -> source confidence reduced and asset put under enhanced monitoring.")
    if etype == EventType.PAYMENT_DELAYED:
        asset.contractual.payment_delay_days += float(payload.get("delay_days", 7.0))

    if etype == EventType.FINANCING_REQUESTED:
        requested = float(payload.get("requested_amount", 0.0))
        asset.financial.requested_financing = max(asset.financial.requested_financing, requested)
        reasons.append(f"Financing request observed: {requested:.2f}.")

    # ---- 3b. Consult the trained ML risk agent (the ONE ML touchpoint) ----
    risk_result = _assess_risk(asset)
    asset.risk.risk_score = risk_result["model_probability"]
    reasons.append(
        f"ML risk probability={risk_result['model_probability']:.3f}; learned action threshold={risk_result['threshold']:.3f}; "
        f"band={risk_result['risk_band']}; velocity={risk_result['risk_velocity']:+.3f}; "
        f"agent_recommendation={risk_result['recommendation']} (model={risk_result['model_source']})."
    )
    if risk_result.get("drivers"):
        top = ", ".join(f"{d['feature']}:{d['contribution']:+.2f}" for d in risk_result['drivers'][:4])
        reasons.append(f"Top model drivers (standardized coefficient contributions): {top}")

    # ---- 4. Settlement short-circuit ----
    if etype == EventType.PAYMENT_RECEIVED:
        settled_amount = asset.financial.total_exposure
        asset.financial.total_exposure = 0.0
        asset.financial.existing_financing.append(
            {"instrument": "SETTLEMENT", "amount": -settled_amount, "ts": ts}
        )
        asset.financial.current_instrument = FinancingInstrument.NONE
        reasons.append("Payment received -> outstanding exposure settled in full.")
        return _log(asset, event, prior_stage, asset.physical.stage, DecisionAction.SETTLE,
                    settled_amount, FinancingInstrument.NONE, reasons,
                    "Existing financing settled; exposure reset to 0.", reconciliation_log)

    # ---- 5. Recompute financing decision (this runs on EVERY event) ----
    # `capped_target` is the ABSOLUTE total the system will allow outstanding
    # right now -- not a delta. Every branch below compares it against the
    # asset's CURRENT total_exposure; nothing here ever discards existing
    # exposure the way an earlier (buggy) headroom-based version did.
    calc = compute_financing_amount(asset)
    target_instrument = INSTRUMENT_BY_STAGE.get(asset.physical.stage, FinancingInstrument.NONE)
    current_instrument = asset.financial.current_instrument
    current_total = asset.financial.total_exposure
    capped_target = calc["capped_target"]

    reasons.append(
        f"stage={asset.physical.stage.value} base_rate={calc['base_rate']} "
        f"risk_multiplier={calc['risk_multiplier']} -> effective_rate={calc['effective_rate']}, "
        f"risk-adjusted eligible_amount={calc['eligible_amount']}, "
        f"exposure_limit={asset.financial.exposure_limit}, capped_target={capped_target}"
    )
    if calc["knockout"]:
        reasons.append("KNOCKOUT RULE: asset condition is DAMAGED -> not financeable regardless of risk score.")
    if calc["capped_by_exposure_limit"]:
        reasons.append(
            f"OVER-LEVERAGE GUARD: risk model would support {calc['eligible_amount']}, "
            f"but this is capped at the exposure_limit ({asset.financial.exposure_limit}) "
            f"to prevent duplicate financing / over-leverage on this asset."
        )

    agent_block = risk_result["recommendation"] == "BLOCK_NEW_FINANCE" and current_total <= 0.0
    agent_reduce = risk_result["recommendation"] == "REDUCE_OR_FREEZE" and current_total > 0.0
    if agent_block:
        decision, action_desc = DecisionAction.REJECT, (
            f"ML risk agent blocked new financing: probability {risk_result['model_probability']:.3f} "
            f">= learned threshold {risk_result['threshold']:.3f} or hard-stop signal."
        )
    elif agent_reduce:
        reduced_target = min(capped_target, round(current_total * 0.90, 2))
        decision, action_desc = DecisionAction.REDUCE, (
            f"ML risk agent froze/instructed de-risking; reduce outstanding exposure to {reduced_target}."
        )
        capped_target = reduced_target
    elif current_instrument == FinancingInstrument.NONE:
        if capped_target > 0:
            decision, action_desc = DecisionAction.INITIATE, f"Initiate {target_instrument.value} of {capped_target}."
        else:
            decision, action_desc = DecisionAction.REJECT, "Risk-adjusted eligible amount is 0 -- no financing initiated."
    elif target_instrument != current_instrument and target_instrument != FinancingInstrument.NONE:
        decision, action_desc = DecisionAction.TRANSITION, (
            f"Transition {current_instrument.value} -> {target_instrument.value}, new total {capped_target}."
        )
    elif capped_target > current_total * 1.05:
        decision, action_desc = DecisionAction.INCREASE, f"Increase financing to {capped_target}."
    elif capped_target < current_total * 0.95:
        decision, action_desc = DecisionAction.REDUCE, f"Reduce financing to {capped_target}."
    else:
        decision, action_desc = DecisionAction.NO_ACTION, (
            "No material change to financing required "
            f"(current {current_total} already matches risk-adjusted target {capped_target})."
        )

    if decision in (DecisionAction.INITIATE, DecisionAction.INCREASE, DecisionAction.TRANSITION, DecisionAction.REDUCE):
        delta = round(capped_target - current_total, 2)
        if abs(delta) > 0.005:
            asset.financial.existing_financing.append(
                {"instrument": target_instrument.value, "amount": delta, "ts": ts}
            )
        asset.financial.total_exposure = capped_target
        asset.financial.current_instrument = target_instrument
        approved = capped_target
    else:
        approved = current_total if decision == DecisionAction.NO_ACTION else 0.0

    return _log(asset, event, prior_stage, asset.physical.stage, decision, approved,
                target_instrument, reasons, action_desc, reconciliation_log)
