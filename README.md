# SCF Risk Command Center — V3

An event-driven supply-chain financing risk-monitoring system for a hackathon
prototype. It models the financing lifecycle of physical assets while tracking
multiple parties (lender, borrower/supplier, buyer, logistics, warehouse/IoT,
financial systems) and continuously reassessing risk.

## Core loop

`OBSERVE → RECONCILE → UPDATE STATE → ML PREDICT → RISK AGENT → DECIDE → ACT → AUDIT → MONITOR`

The dashboard is not a mockup: scenario replay and live event injection both
call the same FastAPI + decision-engine path.

## ML: real probability model, not a hard-coded score

- **Model:** logistic regression on standardized operational + financial features
- **Training data:** 30,000 synthetic historical financing outcomes
- **Split:** 18k train / 6k validation / 6k test
- **Features:** buyer risk, buyer payment history, supplier reliability, production/shipment/payment delays, inventory age, deterioration, invoice disputes, duplicate-financing signal, source confidence, lifecycle progress, advance request, financing utilization, demand volatility
- **Formula:** `p = 1/(1+exp(-(b0 + Σ wi zi)))`
- **Threshold:** selected from the validation set by minimizing `5×FN + 1×FP`; saved to `ml/model_metrics.json` and loaded at runtime. It is **not manually hard-coded**.
- **Current seed-42 holdout:** ROC-AUC ≈ 0.753, PR-AUC ≈ 0.593, Brier ≈ 0.170, risky-class recall ≈ 0.825 at the learned threshold.

The dataset and metrics are synthetic and should be presented as a simulation,
not as real-world credit statistics.

## Agent

`engine/risk_agent.py` is the autonomous monitoring layer. It takes the current
asset state and model output, then computes:

- continuous adverse-outcome probability
- learned operating threshold
- LOW / MODERATE / HIGH risk band
- risk velocity (change since previous observation)
- exposure utilization
- top model drivers using standardized coefficient contributions
- action recommendation: CONTINUE, WATCH_RISK_ACCELERATION,
  ENHANCED_MONITORING, REDUCE_OR_FREEZE, or BLOCK_NEW_FINANCE

Hard controls such as damaged-asset knockout and duplicate-financing alerts are
kept separate from the statistical probability model.

## Monitoring levels

- **Event/source:** who sent the evidence and what changed
- **Asset/transaction:** one traveling asset object through its lifecycle
- **Counterparty:** buyer, borrower/supplier or lender exposure across linked assets
- **Portfolio:** total value, exposure, utilization, high-risk count and exposure concentration by buyer/supplier/lender

## Inputs

### A. Scenario replay
10 deterministic scenarios cover normal operation, production shock, conflicting
sources, damage, buyer deterioration, supplier failure, invoice dispute/payment
delay/recovery, duplicate financing, logistics/location failure, and multi-party
recovery.

### B. Live event injection
The dashboard lets a role send an event directly to `/assets/{id}/events` with a
JSON payload. This is the professional demo path: the judge can change buyer
risk, supplier reliability, inventory age, delay days, invoice dispute status,
duplicate-financing state or location verification and watch the agent react.

### C. Batch operational data
The generator creates 250 synthetic assets and per-source CSVs for PO, production,
logistics, warehouse/IoT, invoices and financing.

## Run

```bash
python3 -m pip install fastapi uvicorn pydantic scikit-learn pandas numpy joblib --break-system-packages

python3 generator/synthetic_generator.py --seed 42 --n-assets 250
python3 generator/historical_data_generator.py --seed 42 --n 30000
python3 ml/train_risk_model.py

python3 tests/test_engine.py
python3 tests/test_professional.py

uvicorn api.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`.

Useful endpoints:

- `/docs` — API documentation
- `/scenarios` — 10 demo event streams
- `/monitor/model` — model metrics + learned threshold
- `/monitor/portfolio` — portfolio-level monitor
- `/monitor/counterparty/{party_id}` — counterparty-level monitor
- `/assets` — current monitored assets
- `/assets/{asset_id}/history` — append-only decision trace
- `/assets/{asset_id}/raw-evidence` — source evidence before reconciliation
- `/assets/{asset_id}/events` — live event injection

## Verification

The baseline suite has **25/25 passing checks**. The professional suite adds
**15 checks** for dataset size, genuine train/validation/test separation, learned
threshold selection, monotonic risk sanity checks across key features, safer
counterparty behavior, and all 10 scenarios.

See `ARCHITECTURE_V3.md` for the judge-facing architecture and mathematical
explanation.
