"""Generate a large, structured synthetic credit-risk training set.

This is synthetic data, not a claim about real default rates.  The label is
sampled from a noisy latent risk process with nonlinear interactions so the
model has a meaningful prediction problem instead of learning a trivial
rule.  The training script later learns the relationship and derives the
operating threshold from validation data rather than hard-coding it.
"""
import argparse, csv, math, os, random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "raw", "historical_financing_outcomes.csv")

STAGES = ["PO_CREATED","MATERIAL_RECEIVED","PRODUCTION","FINISHED_GOODS","SHIPPED","WAREHOUSE","DELIVERED","INVOICED"]


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def bounded(rng, fn, lo=0.0, hi=1.0):
    return min(hi, max(lo, fn()))


def generate_row(rng, i):
    buyer_risk = rng.betavariate(2.2, 6.0)
    buyer_payment = rng.betavariate(7.0, 2.0)
    supplier_reliability = rng.betavariate(7.0, 2.2)
    production_delay = max(0.0, rng.gauss(2.8, 4.8))
    shipment_delay = max(0.0, rng.gauss(1.8, 3.2))
    payment_delay = max(0.0, rng.gauss(2.5, 7.0))
    inventory_age = max(0.0, rng.gauss(18, 16))
    deterioration = 1 if rng.random() < 0.075 else 0
    invoice_dispute = 1 if rng.random() < 0.055 else 0
    duplicate_financing = 1 if rng.random() < 0.035 else 0
    source_confidence = bounded(rng, lambda: rng.gauss(0.84, 0.12), 0.25, 1.0)
    stage_idx = rng.randint(0, len(STAGES)-1)
    stage_progress = stage_idx / (len(STAGES)-1)
    advance_requested = bounded(rng, lambda: rng.gauss(0.58, 0.17), 0.15, 0.95)
    utilization = bounded(rng, lambda: rng.gauss(0.54, 0.22), 0.0, 1.0)
    demand_volatility = bounded(rng, lambda: rng.gauss(0.32, 0.18), 0.0, 1.0)
    order_value = rng.lognormvariate(math.log(450_000), 0.85)
    order_value = min(25_000_000, max(25_000, order_value))

    delay_burden = min(1.0, (production_delay + shipment_delay + 0.7*payment_delay) / 35.0)
    counterparty_weakness = 0.55*buyer_risk + 0.45*(1-supplier_reliability)
    leverage_pressure = max(0.0, utilization + advance_requested - 1.0)

    # Synthetic latent process.  These weights are the data-generating
    # mechanism only; the ML model is trained independently to recover signal.
    latent = (
        2.7*buyer_risk
        + 1.55*(1-buyer_payment)
        + 1.65*(1-supplier_reliability)
        + 1.55*delay_burden
        + 0.95*deterioration
        + 0.85*invoice_dispute
        + 1.65*duplicate_financing
        + 0.75*leverage_pressure
        + 0.025*inventory_age
        + 0.9*demand_volatility
        - 1.05*source_confidence
        - 0.65*stage_progress
        + 0.8*advance_requested
        + 1.1*(buyer_risk*utilization)
        + 0.9*((1-supplier_reliability)*delay_burden)
    )
    z = -4.15 + 1.25*latent
    p = sigmoid(z)
    label = 1 if rng.random() < p else 0
    # Independent noise prevents perfect self-generated labels.
    if rng.random() < 0.025:
        label = 1 - label

    return {
        "record_id": f"H{i:06d}",
        "buyer_risk_score": round(buyer_risk, 5),
        "buyer_payment_history": round(buyer_payment, 5),
        "supplier_reliability": round(supplier_reliability, 5),
        "production_delay_days": round(production_delay, 2),
        "shipment_delay_days": round(shipment_delay, 2),
        "payment_delay_days": round(payment_delay, 2),
        "inventory_age_days": round(inventory_age, 2),
        "deterioration": deterioration,
        "invoice_dispute_flag": invoice_dispute,
        "duplicate_financing_flag": duplicate_financing,
        "source_confidence": round(source_confidence, 5),
        "stage_progress": round(stage_progress, 5),
        "advance_rate_requested": round(advance_requested, 5),
        "financing_utilization": round(utilization, 5),
        "demand_volatility": round(demand_volatility, 5),
        "order_value": round(order_value, 2),
        "adverse_outcome": label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=30000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = [generate_row(rng, i+1) for i in range(args.n)]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    positive = sum(r["adverse_outcome"] for r in rows)/len(rows)
    print(f"Generated {len(rows):,} historical financing records -> {OUT_PATH}")
    print(f"Adverse-outcome base rate: {positive:.3%}")

if __name__ == "__main__":
    main()
