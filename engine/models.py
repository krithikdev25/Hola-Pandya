"""
Core data model for the traveling asset object.
One Asset = one financed purchase order, carrying physical + financial +
contractual state through its whole lifecycle. This is the "financial layer
that travels with the physical asset" required by PS6 Section 6 / Final
Constraints -- deliberately ONE object, not separate PO/Invoice/Shipment
records the way every inspected repo does it.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time


class LifecycleStage(str, Enum):
    PO_CREATED = "PO_CREATED"
    MATERIAL_RECEIVED = "MATERIAL_RECEIVED"
    PRODUCTION = "PRODUCTION"
    PRODUCTION_DELAYED = "PRODUCTION_DELAYED"
    FINISHED_GOODS = "FINISHED_GOODS"
    SHIPPED = "SHIPPED"
    SHIPMENT_DELAYED = "SHIPMENT_DELAYED"
    WAREHOUSE = "WAREHOUSE"
    DELIVERED = "DELIVERED"
    INVOICED = "INVOICED"
    PAYMENT_DELAYED = "PAYMENT_DELAYED"
    SETTLED = "SETTLED"


class EventType(str, Enum):
    PO_CREATED = "PO_CREATED"
    MATERIAL_RECEIVED = "MATERIAL_RECEIVED"
    PRODUCTION_STARTED = "PRODUCTION_STARTED"
    PRODUCTION_PROGRESS = "PRODUCTION_PROGRESS"
    PRODUCTION_DELAYED = "PRODUCTION_DELAYED"
    PRODUCTION_COMPLETED = "PRODUCTION_COMPLETED"
    SHIPMENT_CREATED = "SHIPMENT_CREATED"
    SHIPMENT_DELAYED = "SHIPMENT_DELAYED"
    WAREHOUSE_RECEIVED = "WAREHOUSE_RECEIVED"
    DELIVERY_CONFIRMED = "DELIVERY_CONFIRMED"
    INVOICE_ISSUED = "INVOICE_ISSUED"
    PAYMENT_DELAYED = "PAYMENT_DELAYED"
    PAYMENT_RECEIVED = "PAYMENT_RECEIVED"
    DETERIORATION_DETECTED = "DETERIORATION_DETECTED"
    BUYER_RISK_CHANGED = "BUYER_RISK_CHANGED"
    FINANCING_REQUESTED = "FINANCING_REQUESTED"
    SUPPLIER_RISK_CHANGED = "SUPPLIER_RISK_CHANGED"
    INVOICE_DISPUTED = "INVOICE_DISPUTED"
    INVENTORY_AGING = "INVENTORY_AGING"
    DUPLICATE_FINANCING_ALERT = "DUPLICATE_FINANCING_ALERT"
    LOCATION_MISMATCH = "LOCATION_MISMATCH"


class FinancingInstrument(str, Enum):
    NONE = "NONE"
    PO_ADVANCE = "PO_ADVANCE"
    INVENTORY_FINANCING = "INVENTORY_FINANCING"
    RECEIVABLES_FINANCING = "RECEIVABLES_FINANCING"


class DecisionAction(str, Enum):
    INITIATE = "INITIATE"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    TRANSITION = "TRANSITION"
    REFINANCE = "REFINANCE"
    SETTLE = "SETTLE"
    REJECT = "REJECT"
    NO_ACTION = "NO_ACTION"


@dataclass
class PhysicalState:
    stage: LifecycleStage = LifecycleStage.PO_CREATED
    production_pct: float = 0.0
    location: str = "supplier_facility"
    condition_flag: str = "NORMAL"          # NORMAL | DETERIORATED | DAMAGED
    delay_flag: bool = False
    production_delay_days: float = 0.0
    shipment_delay_days: float = 0.0
    last_updated: float = field(default_factory=time.time)
    source: str = "erp"
    confidence: float = 1.0                  # 0..1, set by reconciliation
    inventory_age_days: float = 0.0
    location_verified: bool = True


@dataclass
class FinancialState:
    estimated_value: float = 0.0
    existing_financing: list = field(default_factory=list)   # list of {instrument, amount, ts}
    total_exposure: float = 0.0
    exposure_limit: float = 0.0
    requested_financing: float = 0.0
    current_instrument: FinancingInstrument = FinancingInstrument.NONE


@dataclass
class ContractualState:
    po_id: str = ""
    buyer_id: str = ""
    supplier_id: str = ""
    lender_id: str = ""
    buyer_risk_score: float = 0.2            # 0 (safe) .. 1 (risky)
    buyer_payment_history: float = 0.85      # 0..1, higher = better
    supplier_reliability: float = 0.85       # 0..1, higher = better
    agreed_price: float = 0.0
    payment_terms_days: int = 30
    delivery_deadline: Optional[float] = None
    invoice_id: Optional[str] = None
    receivable_due_date: Optional[float] = None
    invoice_dispute_flag: bool = False
    payment_delay_days: float = 0.0


@dataclass
class RiskState:
    risk_score: float = 0.2                  # 0..1, higher = riskier
    risk_factors: list = field(default_factory=list)
    risk_velocity: float = 0.0
    last_model_probability: float = 0.2
    risk_band: str = "LOW"


@dataclass
class Asset:
    id: str
    product_name: str
    quantity: int
    physical: PhysicalState = field(default_factory=PhysicalState)
    financial: FinancialState = field(default_factory=FinancialState)
    contractual: ContractualState = field(default_factory=ContractualState)
    risk: RiskState = field(default_factory=RiskState)
    audit_log: list = field(default_factory=list)  # list of dicts, append-only

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Asset":
        """Reconstruct an Asset from a plain dict (e.g. loaded from JSON/SQLite).
        JSON round-tripping turns Enum fields into plain strings, so those
        must be converted back explicitly -- a dataclass(**kwargs) call alone
        does not know PhysicalState.stage should be a LifecycleStage."""
        a = Asset(id=d["id"], product_name=d["product_name"], quantity=d["quantity"])

        phys = dict(d["physical"])
        phys["stage"] = LifecycleStage(phys["stage"])
        a.physical = PhysicalState(**phys)

        fin = dict(d["financial"])
        fin["current_instrument"] = FinancingInstrument(fin["current_instrument"])
        a.financial = FinancialState(**fin)

        a.contractual = ContractualState(**d["contractual"])
        a.risk = RiskState(**d["risk"])
        a.audit_log = d.get("audit_log", [])
        return a
