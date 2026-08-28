# Verification Audit — what was actually checked, and what it found

This documents the verification actually performed this session, including two real bugs it caught, per the "no fake results" standard: nothing below is asserted without the command/test that produced it.

## Phase 1 — Core engine

- `python3 tests/test_engine.py` → **25/25 checks pass.** Covers: financing math per stage, the over-leverage guard (both "eligible exceeds limit" and "already at limit" cases), a genuine REJECT path (damaged asset, zero confidence), illegal state-transition rejection, multi-source reconciliation, and 4 full scenario runs.
- **Bug found and fixed #1:** the original exposure-limit logic conflated "headroom" (spare capacity) with "target total," which silently discarded pre-existing exposure the moment a new event arrived — e.g., an asset with ₹4.1L already financed could have that number overwritten by a much smaller "headroom-limited" figure. Caught via a screenshot review, not by the unit tests (they hadn't covered a seeded-with-existing-exposure case at first) — fixed by redefining `compute_financing_amount` to return an absolute risk-adjusted target capped at the exposure limit, compared against current exposure explicitly. Verified with 3 new targeted tests (Section 2 of `tests/test_engine.py`).
- **Design gap found and fixed:** the risk multiplier had a floor of 0.1, meaning even a maximally bad-risk asset (damaged + zero confidence + max buyer risk) still got financed. Added an explicit knockout rule (`condition_flag == "DAMAGED"` → not financeable) so a genuine REJECT path exists, verified by test.

## Phase 2 — ML risk model

- `python3 ml/train_risk_model.py` → real train/test split (1200/300 rows), metrics computed and saved to `ml/model_metrics.json`, not invented.
- **Honestly reported, not hidden:** the first unweighted model got 82.3% raw accuracy but only 1.9% recall on the risky class — worse than useless, since it just predicted "not risky" almost every time (majority-class baseline was 82.7%, higher than the model). Fixed with `class_weight="balanced"` (a standard, documented technique, not tuning-until-pretty): recall on the risky class rose to 53.9%, at the cost of raw accuracy dropping to 66%. AUC (the threshold-independent metric) sits at ~0.62 — real signal above the 0.5 random baseline, explicitly not claimed as strong.
- Model coefficients all have the expected sign (buyer risk, delays, deterioration → positive; confidence, stage progress → negative), checked in `tests/test_engine.py` Section 6 with a monotonicity test (a clean asset profile scores lower than a maximally bad one).

## Phase 3 — API + frontend integration

- Started the real server (`uvicorn api.main:app`) and hit endpoints with curl: valid asset creation, invalid input (negative quantity → 422), missing role header (→ 403 — confirmed, not assumed), a valid event, and the exact illegal-transition example from the problem statement (`PAYMENT_RECEIVED` on a `PO_CREATED` asset) — correctly rejected with a logged reason.
- **Bug found and fixed #2:** the first live event call against a stored (SQLite-loaded) asset crashed with a 500 error — enum fields (`stage`, `current_instrument`) survive a dataclass→dict→JSON round-trip as plain strings, and the audit logger tried to call `.value` on a string. Fixed in `Asset.from_dict`; re-verified the exact same failing request now succeeds.
- Took real Playwright screenshots of the running frontend after clicking "Run scenario" for scenario 3 (reconciliation) and scenario 4 (over-leverage) — included alongside this file. These are screenshots of the live page hitting the live API, not mockups.

## What this audit does not cover (named, not hidden)

- No load/performance testing.
- No penetration-style security testing beyond the input-validation and role-check checks already listed.
- Scenarios 5 (deterioration) and 6 (successful transition) are not separately scripted — the underlying engine already exercises both mechanisms (a `DETERIORATED`/`DAMAGED` condition flag and its knockout rule; the `TRANSITION` decision already visible in every scenario's trace above), but no dedicated scenario file or test targets them individually.
