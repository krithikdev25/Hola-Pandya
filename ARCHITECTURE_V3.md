# SCF Risk Command Center — V3 architecture

## What changed

The original prototype had one small logistic model and four scripted scenarios.
V3 turns it into a multi-party, event-driven risk-monitoring system:

**event → reconcile → update asset state → ML probability → learned threshold → risk agent → financing policy → audit → monitor again**

There is no LLM in the control loop. "Agent" means an autonomous software
component that repeatedly observes state, reasons from the model + policy,
acts, and re-evaluates when new evidence arrives.

## ML model

A logistic-regression probability model is trained on 30,000 synthetic historical
financing outcomes. The model uses 15 operational/financial features:

- buyer risk and payment history
- supplier reliability
- production, shipment and payment delays
- inventory age
- deterioration, invoice disputes, duplicate-financing alerts
- source confidence
- lifecycle progress
- requested advance rate
- financing utilization
- demand volatility

The model is a real statistical probability model:

`p(adverse) = 1 / (1 + exp(-(b0 + Σ wi * standardized_feature_i)))`

The coefficients are stored in `ml/model_metrics.json` and are available to the
runtime for explanation.

### Threshold is learned, not hard-coded

Training uses three splits: train / validation / test.
The operating threshold is searched across validation probabilities and chosen
by minimizing:

`cost = 5 × false_negatives + 1 × false_positives`

The selected threshold is saved in `model_metrics.json` and loaded by the risk
agent. With the current seed it is **0.205**. It is therefore an empirical
operating threshold from held-out validation data, while the 5:1 cost ratio is
an explicit business policy.

Current holdout metrics with seed 42:

- 30,000 total rows
- 18,000 train / 6,000 validation / 6,000 test
- ROC-AUC ≈ 0.753
- PR-AUC ≈ 0.593
- Brier score ≈ 0.170
- risky-class recall at the learned threshold ≈ 0.825

These are synthetic-data metrics and must not be represented as real-world
credit-performance claims.

## Multi-level monitoring

### Level 1 — Event / source
Who sent the evidence? ERP, production system, IoT, warehouse, logistics,
buyer, supplier/borrower, invoice system, financial system.

### Level 2 — Asset / transaction
One `Asset` carries physical, financial, contractual and risk state. Every event
causes a fresh model prediction and financing reassessment.

### Level 3 — Counterparty
`GET /monitor/counterparty/{party_id}` aggregates linked assets and reports
exposure, average risk and maximum risk for a buyer, supplier/borrower or lender.

### Level 4 — Lender / facility
Each asset records `lender_id`; portfolio aggregation exposes exposure by lender.

### Level 5 — Portfolio
`GET /monitor/portfolio` aggregates total value, total exposure, utilization,
high-risk assets and exposure concentration by buyer, supplier and lender.

## Roles

- **Lender** — financing/exposure owner
- **Borrower/Supplier** — physical producer and financing requester
- **Buyer** — commercial counterparty / receivable obligor
- **Production** — manufacturing evidence
- **Logistics** — shipment/location evidence
- **Warehouse/IoT** — physical condition/location evidence
- **Financial system** — financing/payment evidence
- **System agent** — autonomous reassessment and policy action

The demo `X-Actor-Role` header is only a prototype authorization boundary; it is
not production authentication.

## Input modes

1. **Scenario replay** — 10 deterministic event streams for judge demos.
2. **Live event injection** — choose an asset, actor role, event type and JSON
   payload in the dashboard; the same `/assets/{id}/events` endpoint invokes the
   agent. There is no separate demo-only decision path.
3. **Operational batch** — 250 synthetic assets across PO, production, logistics,
   warehouse, invoice and financing CSV sources.

## Judge-proof demo scenarios

1. Clean lifecycle → automatic settlement
2. Production shock → financing de-risking
3. ERP vs IoT conflict → source reconciliation
4. Physical damage → hard stop
5. Buyer credit deterioration → receivable risk increase
6. Supplier reliability collapse → risk reassessment
7. Invoice dispute → payment delay → recovery
8. Duplicate financing → block + exposure guard
9. Logistics delay + location failure + inventory aging
10. Multi-party recovery → model risk falls as evidence improves

The important demo is not "click Run Scenario". The point is that the same
asset receives new evidence from different actors, the probability changes,
model drivers are logged, the learned threshold is consulted, financing changes
without human approval, and the resulting state remains auditable.
