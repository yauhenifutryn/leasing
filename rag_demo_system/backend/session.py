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

# CP-2.2b: "ИП" dropped. profile_hygiene._normalize_client_type collapses all
# business forms (ИП / самозанятый / ООО / бизнесмен / etc.) to "Юридическое
# лицо" before they reach the profile; the calculator API only accepts these
# two values. Three sources of truth collapsed into one (E-Codex finding).
ClientType = Literal["Физическое лицо", "Юридическое лицо"]
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

    # Fix 1.2 (2026-04-19) — preserve the client's pre-conversion figures so
    # every render path (readback, calc-result voice summary, SMS body) can
    # disclose both amounts: "20000 долларов, это 60000 белорусских рублей
    # по курсу 3 к 1". Populated only on the Физлицо + USD direct-call path
    # at the USD -> BYN conversion site (app.py, DirectTool block).
    original_cost: Optional[float] = None
    original_currency: Optional[str] = None

    # Bare-"Да" → SMS support (live call cdbcf56b 2026-04-26). Set to "sms"
    # by execute_action's FireCalc handler when the post-calc readback is
    # spoken (which always ends with "Хотите изменить или отправить по
    # СМС?"). apply_turn reads this on the next turn: if the user replies
    # with a bare affirmation and no change_field, the answer is "yes,
    # send the SMS" — fire FireSMS. Cleared on any other turn that
    # consumes the offer or supersedes it.
    last_offer: Optional[str] = None

    # Bug 1 (live call bf7a95a8 2026-04-26): subject was filled silently
    # ("Я бы себе хотел что-то купить" → readback announced Легковой
    # автомобиль the user never confirmed). This flag is the structural
    # gate: only flips True when subject is captured via a path with
    # explicit utterance evidence (apply_turn step 5 first-time capture
    # from grounded patches, or change-confirm cycle). When the flag is
    # False but ``subject`` happens to be set anyway (any future leak),
    # ``missing_fields()`` treats subject as missing so the clarify gate
    # asks the user before the readback fires. Defaults True only when
    # subject was provided at construction (test fixtures, snapshots
    # rebuilt from disk) — see ``__post_init__``.
    subject_user_grounded: bool = False

    confirmed_at: Optional[float] = None
    last_change_pending: Optional[str] = None
    locked_fields: set[str] = field(default_factory=set)

    state: ProfileState = ProfileState.COLLECTING
    readback_emitted_at: Optional[float] = None
    change_emitted_at: Optional[float] = None
    # Shape: {"field": str, "old_value": Any, "new_value": Any}  (single-field, legacy)
    #    or: {"changes": {field_name: {"old": Any, "new": Any}, ...}}  (multi-field, Fix 28)
    # The multi-field shape supports one turn that modifies several calculator
    # params ("легковой за 80 тысяч" = subject + cost). A single-field payload
    # is still accepted for backward compatibility with older call sites.
    pending_change: Optional[dict[str, Any]] = None

    _CORE_FIELDS = (
        "client_type",
        "subject",
        "cost",
        "currency",
        "condition_new",
        "term_months",
        "type_schedule",
    )

    def __post_init__(self) -> None:
        # Construction-time assumption (bug 1 fix, 2026-04-26): when callers
        # provide ``subject`` at construction, treat it as already user-
        # grounded. Test fixtures and snapshot rebuilds rely on this so
        # they don't all have to flip the flag manually. The runtime
        # capture path (apply_turn) explicitly flips the flag when subject
        # lands via grounded patches; new ClientProfile() with no subject
        # leaves the flag False as intended.
        if self.subject is not None and not self.subject_user_grounded:
            self.subject_user_grounded = True

    def missing_fields(self) -> set[str]:
        missing: set[str] = set()
        for f_name in self._CORE_FIELDS:
            if getattr(self, f_name) is None:
                missing.add(f_name)
        if self.prepaid_pct is None and self.prepaid_amount is None:
            missing.add("prepaid")
        if self.condition_new == 0 and self.age_years is None:
            missing.add("age_years")
        # Bug 1 (live call bf7a95a8 2026-04-26): silent-default subject
        # must surface as missing so the clarify gate fires before the
        # readback. The flag is True iff subject was captured from a path
        # with utterance evidence (apply_turn first-time capture or
        # change-confirm) or provided at construction.
        if self.subject is not None and not self.subject_user_grounded:
            missing.add("subject")
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

    def apply_additive_patches(self, patches: dict[str, Any]) -> dict[str, Any]:
        """Additive-capture variant of apply_patches with prepaid-sibling-clear.

        Use this when applying first-time captures on COLLECTING state (apply_turn
        step 5) instead of raw setattr, so the prepaid_pct / prepaid_amount
        slot-invariant from apply_pending_change is preserved.

        Like apply_patches, returns a dict of fields actually changed (for
        logging / telemetry). Sibling clears do NOT appear in the returned
        dict — only the caller-provided patches that were applied.

        The returned ``changed`` dict reports only the caller-provided patches
        that were actually applied. Prepaid sibling clears (when setting
        prepaid_pct nulls a prior prepaid_amount, or vice versa) are a side
        effect and are NOT included in the return value — matches the
        ``apply_pending_change`` reporting convention, so a future callsite
        doesn't surprise itself with an unexpected extra key.
        """
        changed: dict[str, Any] = {}
        if not patches:
            return changed
        _applied_prepaid_pct = False
        _applied_prepaid_amount = False
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
                if k == "prepaid_pct":
                    _applied_prepaid_pct = True
                elif k == "prepaid_amount":
                    _applied_prepaid_amount = True
        if _applied_prepaid_pct and self.prepaid_amount is not None:
            self.prepaid_amount = None
        elif _applied_prepaid_amount and self.prepaid_pct is not None:
            self.prepaid_pct = None
        return changed

    def apply_pending_change(self) -> bool:
        """Apply pending_change to the profile, clear it. Return True if applied.

        Supports both single-field and multi-field pending_change shapes.
        The `changes` multi-field dict (Fix 28) is iterated in insertion order.

        Fix 42d: prepaid_pct and prepaid_amount share a semantic slot. When
        one is set via pending_change, the other is cleared to prevent the
        stale value from shadowing in direct-call params build (pct is
        preferred over amount). Mirrors the sticky-patch counterpart-clear
        logic (Fix 40c).

        2026-04-24 (Codex adversarial high #1): locked_fields are now honoured,
        consistent with apply_patches() and apply_additive_patches(). A change
        staged for a locked field is silently skipped — the field retains its
        current value. This matches the contract of the other two apply helpers
        and prevents a confirmed pending_change from bypassing the lock.
        """
        if not self.pending_change:
            return False
        _applied_prepaid_pct = False
        _applied_prepaid_amount = False
        # Multi-field shape.
        _changes = self.pending_change.get("changes")
        if isinstance(_changes, dict) and _changes:
            _applied_any = False
            for field_name, vals in _changes.items():
                if field_name in self.locked_fields:
                    # Locked fields are silently skipped — same semantics as
                    # apply_patches() and apply_additive_patches(). The change
                    # is not applied; _applied_any stays False for this field.
                    print(
                        f"[ClientProfile] apply_pending_change: skipping locked "
                        f"field={field_name!r}",
                        flush=True,
                    )
                    continue
                if not hasattr(self, field_name):
                    # Codex adversarial pass 4 (2026-04-20): log loudly but
                    # skip unknown fields. If NO known fields got applied, we
                    # fall through to return False below without clearing
                    # pending_change, so the caller can decide not to advance
                    # the profile to CONFIRMED on a malformed payload.
                    print(
                        f"[ClientProfile] apply_pending_change: ignoring unknown "
                        f"field={field_name!r} — state-loss guard",
                        flush=True,
                    )
                    continue
                new_value = vals.get("new") if isinstance(vals, dict) else vals
                setattr(self, field_name, new_value)
                _applied_any = True
                if field_name == "prepaid_pct":
                    _applied_prepaid_pct = True
                elif field_name == "prepaid_amount":
                    _applied_prepaid_amount = True
            if not _applied_any:
                print(
                    f"[ClientProfile] apply_pending_change: no known/unlocked fields "
                    f"in {list(_changes.keys())} — leaving pending_change for retry",
                    flush=True,
                )
                return False
            if _applied_prepaid_pct and self.prepaid_amount is not None:
                self.prepaid_amount = None
            elif _applied_prepaid_amount and self.prepaid_pct is not None:
                self.prepaid_pct = None
            self.pending_change = None
            return True
        # Legacy single-field shape.
        field_name = self.pending_change.get("field")
        new_value = self.pending_change.get("new_value")
        if field_name and field_name not in self.locked_fields and hasattr(self, field_name):
            setattr(self, field_name, new_value)
            if field_name == "prepaid_pct" and self.prepaid_amount is not None:
                self.prepaid_amount = None
            elif field_name == "prepaid_amount" and self.prepaid_pct is not None:
                self.prepaid_pct = None
            self.pending_change = None
            return True
        if field_name and field_name in self.locked_fields:
            print(
                f"[ClientProfile] apply_pending_change: legacy single-field "
                f"pending_change skipping locked field={field_name!r}",
                flush=True,
            )
            return False
        # Legacy single-field with unknown attribute — same fail-closed behaviour.
        print(
            f"[ClientProfile] apply_pending_change: legacy single-field "
            f"pending_change has unknown field={field_name!r} — leaving for retry",
            flush=True,
        )
        return False

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
