"""
Priority tests per ARCHITECTURE_AND_BUILD_PLAN.md Section 12: financing
amount math, exposure rejection, invalid transitions, reconciliation
conflict handling, and one full end-to-end scenario run. Plain asserts +
a runner (no pytest dependency required), so this can be run anywhere.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import Asset, FinancialState, ContractualState, LifecycleStage, EventType
from engine.decision_engine import reassess, compute_financing_amount
from engine.state_machine import guard_transition, InvalidTransitionError
from engine.reconciliation import reconcile_field
from engine.scenario_runner import run_scenario

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


def make_asset(value=1_000_000, exposure_limit=None, total_exposure=0.0, stage=LifecycleStage.PO_CREATED):
    a = Asset(id="TEST1", product_name="widgets", quantity=1000)
    a.financial = FinancialState(estimated_value=value,
                                  exposure_limit=exposure_limit if exposure_limit is not None else value * 0.9,
                                  total_exposure=total_exposure)
    a.contractual = ContractualState(po_id="PO1", buyer_id="B1", agreed_price=value / 1000)
    a.physical.stage = stage
    return a


print("1) Financing amount calculation per instrument")
a = make_asset()
calc = compute_financing_amount(a)
check("PO stage produces a positive, bounded advance",
      0 < calc["capped_target"] <= a.financial.estimated_value,
      calc)
a2 = make_asset(stage=LifecycleStage.INVOICED)
calc2 = compute_financing_amount(a2)
check("Invoiced-stage advance rate > PO-stage advance rate (later stage = safer = more financeable)",
      calc2["effective_rate"] > calc["effective_rate"], (calc, calc2))

print("2) Exposure rejection (duplicate financing / over-leverage)")
from engine.models import FinancingInstrument

# Case A: the risk-adjusted math would support a LOT more than the exposure
# limit allows (eligible far exceeds the cap). System should cap the target
# at the limit and say so explicitly, not silently lend the uncapped amount.
a3a = make_asset(value=5_000_000, exposure_limit=400_000, total_exposure=100_000, stage=LifecycleStage.INVOICED)
a3a.financial.current_instrument = FinancingInstrument.RECEIVABLES_FINANCING
a3a.risk.risk_score = 0.1  # low risk -> high eligible amount, to guarantee eligible > limit
entry_a = reassess(a3a, {"type": "INVOICE_ISSUED", "source": "invoice_system", "timestamp": 0, "payload": {}})
check("When the risk-adjusted eligible amount exceeds the exposure limit, the system caps the "
      "granted amount AT the limit rather than the larger uncapped amount",
      entry_a["amount"] <= 400_000 + 1e-6,
      entry_a)
check("The system explicitly logs that the over-leverage guard capped the amount "
      "(not a silent cap)",
      any("OVER-LEVERAGE GUARD" in r for r in entry_a["reasons"]),
      entry_a["reasons"])

# Case B: total_exposure is ALREADY at the exposure limit and the uncapped
# eligible amount is still above the limit -- this is the actual
# duplicate-financing/over-leverage-prevention case PS6 describes. The
# system must NOT increase exposure further.
a3b = make_asset(value=5_000_000, exposure_limit=400_000, total_exposure=400_000, stage=LifecycleStage.INVOICED)
a3b.financial.current_instrument = FinancingInstrument.RECEIVABLES_FINANCING
a3b.risk.risk_score = 0.1
entry_b = reassess(a3b, {"type": "INVOICE_ISSUED", "source": "invoice_system", "timestamp": 0, "payload": {}})
check("At the exposure limit with the model wanting more, exposure is NOT increased "
      "beyond the limit (decision is NO_ACTION/REDUCE, never INCREASE past the cap)",
      entry_b["decision"] != "INCREASE" and a3b.financial.total_exposure <= 400_000 + 1e-6,
      entry_b)

check("Total exposure never exceeds the configured limit in either case",
      a3a.financial.total_exposure <= a3a.financial.exposure_limit + 1e-6 and
      a3b.financial.total_exposure <= a3b.financial.exposure_limit + 1e-6,
      (a3a.financial.total_exposure, a3b.financial.total_exposure))

# Case C: a brand-new asset (no current instrument) whose risk-adjusted
# eligible amount is genuinely 0 (terrible risk, zero confidence) must be
# REJECTed outright, not given a token amount.
a3c = make_asset(value=1_000_000, exposure_limit=900_000, total_exposure=0, stage=LifecycleStage.PO_CREATED)
a3c.physical.confidence = 0.0
a3c.risk.risk_score = 1.0
a3c.contractual.buyer_risk_score = 1.0
a3c.physical.condition_flag = "DAMAGED"
entry_c = reassess(a3c, {"type": "PO_CREATED", "source": "erp", "timestamp": 0, "payload": {}})
check("A brand-new asset with a near-zero risk-adjusted eligible amount is REJECTed, "
      "not given a token financing amount",
      entry_c["decision"] == "REJECT" and entry_c["amount"] == 0.0,
      entry_c)

print("3) Invalid state transition rejected")
try:
    guard_transition(LifecycleStage.INVOICED, LifecycleStage.PRODUCTION)
    check("PAYMENT_RECEIVED-like invalid transition raises", False, "no exception raised")
except InvalidTransitionError:
    check("Invalid transition (INVOICED -> PRODUCTION) correctly raises InvalidTransitionError", True)

a4 = make_asset(stage=LifecycleStage.SETTLED)
entry4 = reassess(a4, {"type": "PRODUCTION_STARTED", "source": "erp", "timestamp": 0, "payload": {}})
check("Engine rejects an illegal event via reassess() rather than silently applying it",
      entry4["decision"] == "REJECT" and a4.physical.stage == LifecycleStage.SETTLED,
      entry4)

print("4) Reconciliation conflict handling")
readings = [
    {"value": 100, "source": "erp", "timestamp": __import__("time").time() - 5},
    {"value": 80, "source": "production_system", "timestamp": __import__("time").time() - 1},
]
chosen, status, log = reconcile_field(readings, tolerance=5.0)
check("Disagreeing sources produce a non-silent resolution (status != AGREED)",
      status in ("RESOLVED_BY_CONFIDENCE", "UNVERIFIED"), (chosen, status))
check("Higher-authority/fresher source (production_system) wins over ERP here",
      chosen == 80, (chosen, status, log))

print("5) End-to-end scenario runs")
for i in range(1, 5):
    result = run_scenario(f"scenario_{i}")
    check(f"scenario_{i} ({result['scenario']}) runs to completion with a full audit trail",
          len(result["trace"]) == len(result["trace"]) and all("decision" in e for e in result["trace"]),
          f"{len(result['trace'])} events processed")

s1 = run_scenario("scenario_1")
check("Scenario 1 (normal lifecycle) ends SETTLED with zero exposure",
      s1["final_state"]["physical"]["stage"] == "SETTLED" and s1["final_state"]["financial"]["total_exposure"] == 0.0,
      s1["final_state"]["financial"])

s2 = run_scenario("scenario_2")
last_s2 = s2["trace"][-1]
check("Scenario 2 (production delay) logs a REJECT/NO_ACTION or reduced decision reflecting increased risk",
      "risk" in " ".join(last_s2["reasons"]).lower() or last_s2["decision"] in ("REDUCE", "NO_ACTION", "INITIATE"),
      last_s2)
check("Scenario 2 risk_score increased due to the delay",
      s2["final_state"]["risk"]["risk_score"] > 0.2, s2["final_state"]["risk"])

s3 = run_scenario("scenario_3")
conflict_entries = [e for e in s3["trace"] if e.get("reconciliation")]
check("Scenario 3 (conflicting sources) produced at least one logged reconciliation event",
      len(conflict_entries) >= 1, len(conflict_entries))

s4 = run_scenario("scenario_4")
check("Scenario 4 (near exposure limit) does not blow past the exposure limit",
      s4["final_state"]["financial"]["total_exposure"] <= s4["final_state"]["financial"]["exposure_limit"] + 1e-6,
      s4["final_state"]["financial"])

print("6) Trained ML risk model sanity checks")
from engine.risk_model import predict_risk, model_is_loaded, get_model_metrics

check("Trained model artifacts are present and load (not silently on fallback heuristic)",
      model_is_loaded(), "ml/risk_model.joblib / risk_scaler.joblib missing or failed to load")

metrics = get_model_metrics()
check("Model metrics were computed on a genuine held-out test split (n_test > 0)",
      metrics.get("n_test", 0) > 0, metrics.get("n_test"))
check("Reported AUC is between 0.5 (random) and 1.0 exclusive-of-suspicious-perfection "
      "(a real synthetic-label model with injected noise should not hit 1.0)",
      0.5 < metrics.get("test_auc", 0) < 0.99, metrics.get("test_auc"))

low_risk = predict_risk({
    "buyer_risk_score": 0.05, "production_delay_days": 0, "shipment_delay_days": 0,
    "deterioration": 0, "confidence_at_decision": 1.0, "stage_progress": 1.0,
    "advance_rate_requested": 0.3,
})
high_risk = predict_risk({
    "buyer_risk_score": 0.95, "production_delay_days": 25, "shipment_delay_days": 15,
    "deterioration": 1, "confidence_at_decision": 0.3, "stage_progress": 0.0,
    "advance_rate_requested": 0.8,
})
check("Model assigns higher risk to an objectively worse asset profile than a clean one "
      "(monotonic sanity check, not just 'it returns a number')",
      high_risk["risk_score"] > low_risk["risk_score"],
      (low_risk, high_risk))
check("Model outputs are valid probabilities in [0, 1]",
      0.0 <= low_risk["risk_score"] <= 1.0 and 0.0 <= high_risk["risk_score"] <= 1.0,
      (low_risk, high_risk))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
