"""
Explicit lifecycle transition guard.
Satisfies the "state-machine security" requirement: reject impossible
transitions (e.g. PAYMENT_RECEIVED -> PRODUCTION_STARTED) instead of
silently accepting any event in any state. This is deliberately a plain
dict, not a framework, so it can be read and defended in one screen.
"""
from engine.models import LifecycleStage as S

ALLOWED_TRANSITIONS = {
    S.PO_CREATED:          {S.MATERIAL_RECEIVED},
    S.MATERIAL_RECEIVED:   {S.PRODUCTION},
    S.PRODUCTION:          {S.PRODUCTION_DELAYED, S.FINISHED_GOODS},
    S.PRODUCTION_DELAYED:  {S.PRODUCTION, S.FINISHED_GOODS},
    S.FINISHED_GOODS:      {S.SHIPPED},
    S.SHIPPED:             {S.SHIPMENT_DELAYED, S.WAREHOUSE, S.DELIVERED},
    S.SHIPMENT_DELAYED:    {S.WAREHOUSE, S.DELIVERED},
    S.WAREHOUSE:           {S.DELIVERED},
    S.DELIVERED:           {S.INVOICED},
    S.INVOICED:            {S.PAYMENT_DELAYED, S.SETTLED},
    S.PAYMENT_DELAYED:     {S.SETTLED},
    S.SETTLED:             set(),
}


class InvalidTransitionError(Exception):
    pass


def guard_transition(current: S, target: S) -> None:
    """Raise InvalidTransitionError if current -> target is not an allowed edge.
    A no-op (current == target) is always allowed (e.g. a progress update
    that doesn't change stage)."""
    if current == target:
        return
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidTransitionError(
            f"Illegal lifecycle transition: {current.value} -> {target.value}. "
            f"Allowed from {current.value}: {sorted(s.value for s in allowed)}"
        )
