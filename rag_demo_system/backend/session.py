"""Per-session state that augments VoiceSession: ClientProfile and related types.

ClientProfile is an incrementally populated bag of calculator parameters.
It is the single source of truth for what we know about the client's
leasing request. The calculator is never called until the profile is
complete AND confirmed by the client through a semantic read-back gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

ClientType = Literal["Физическое лицо", "ИП", "Юридическое лицо"]
ScheduleType = Literal["0", "1"]  # 0 = annuity, 1 = linear / declining


class ProfileState(str, Enum):
    COLLECTING = "COLLECTING"
    READBACK_PENDING = "READBACK_PENDING"
    CONFIRMED = "CONFIRMED"
    CHANGE_PENDING = "CHANGE_PENDING"


@dataclass
class ClientProfile:
    """Leasing client parameters collected during the session.

    All data fields are Optional so population is incremental. The profile
    is complete (`is_complete_for_calc() is True`) only when every
    calculator-required field is set. `confirmed_at` is stamped by the
    state machine after the client explicitly confirms the read-back.
    """

    name: Optional[str] = None
    client_type: Optional[ClientType] = None
    subject: Optional[str] = None
    cost: Optional[float] = None
    currency: Optional[str] = None
    condition_new: Optional[int] = None
    age_years: Optional[int] = None
    prepaid_pct: Optional[float] = None
    prepaid_amount: Optional[float] = None
    term_months: Optional[int] = None
    type_schedule: Optional[ScheduleType] = None

    confirmed_at: Optional[float] = None
    last_change_pending: Optional[str] = None
    locked_fields: set[str] = field(default_factory=set)

    state: ProfileState = ProfileState.COLLECTING
    readback_emitted_at: Optional[float] = None
    change_emitted_at: Optional[float] = None
    pending_change: Optional[dict[str, Any]] = None  # {"field": str, "old_value": Any, "new_value": Any}

    _CORE_FIELDS = (
        "client_type",
        "subject",
        "cost",
        "currency",
        "condition_new",
        "term_months",
        "type_schedule",
    )

    def missing_fields(self) -> set[str]:
        missing: set[str] = set()
        for f_name in self._CORE_FIELDS:
            if getattr(self, f_name) is None:
                missing.add(f_name)
        if self.prepaid_pct is None and self.prepaid_amount is None:
            missing.add("prepaid")
        if self.condition_new == 0 and self.age_years is None:
            missing.add("age_years")
        return missing

    def is_complete_for_calc(self) -> bool:
        return not self.missing_fields()

    def apply_patches(self, patches: dict[str, Any]) -> dict[str, Any]:
        """Merge non-None patches into the profile, respecting locked_fields.

        Returns a dict of fields actually changed (for logging / telemetry).
        """
        changed: dict[str, Any] = {}
        if not patches:
            return changed
        for k, v in patches.items():
            if v is None:
                continue
            if k in self.locked_fields:
                continue
            if not hasattr(self, k):
                continue
            old = getattr(self, k)
            if old != v:
                setattr(self, k, v)
                changed[k] = v
        return changed

    def apply_pending_change(self) -> bool:
        """Apply pending_change to the profile, clear it. Return True if applied."""
        if not self.pending_change:
            return False
        field_name = self.pending_change.get("field")
        new_value = self.pending_change.get("new_value")
        if field_name and hasattr(self, field_name):
            setattr(self, field_name, new_value)
        self.pending_change = None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "client_type": self.client_type,
            "subject": self.subject,
            "cost": self.cost,
            "currency": self.currency,
            "condition_new": self.condition_new,
            "age_years": self.age_years,
            "prepaid_pct": self.prepaid_pct,
            "prepaid_amount": self.prepaid_amount,
            "term_months": self.term_months,
            "type_schedule": self.type_schedule,
            "confirmed_at": self.confirmed_at,
            "last_change_pending": self.last_change_pending,
            "locked_fields": sorted(self.locked_fields),
            "state": self.state.value,
            "readback_emitted_at": self.readback_emitted_at,
            "change_emitted_at": self.change_emitted_at,
            "pending_change": self.pending_change,
        }
