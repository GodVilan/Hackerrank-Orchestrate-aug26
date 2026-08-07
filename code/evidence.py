"""Candidate pool construction and evidence enforcement.

The pool is built deterministically in code and scoped to the counterparty. The
model selects from it; it never proposes an ID of its own, and any ID it returns
that is not in the pool is dropped rather than trusted.

No retrieval, no ranking, no top-k. Measured pool sizes are mean 5.8 and max 21,
so the whole pool fits in context and ranking would only add a failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import schema

#: Precedence used to reduce a message_events row to a single reaction word.
#:
#: The event table sets five independent booleans and 287 of its 412 rows set
#: more than one, so a single label requires an explicit order. Strongest
#: negative signal first: a message that was opened *and* reported is evidence
#: of a report, not of an open.
_REACTION_ORDER = (
    ("message_reported", "reported"),
    ("notification_dismissed", "dismissed"),
    ("message_replied", "replied"),
    ("message_opened", "opened"),
)


@dataclass(frozen=True)
class Candidate:
    """One historical message offered to the model as possible evidence."""

    message_id: str
    created_at: str
    text: str
    media_type: str
    forwarded_count: str
    reaction: str

    def render(self) -> str:
        """One line, pipe-separated, exactly as the prompt presents it."""
        text = (self.text or "").replace("\n", " ").strip()[:120]
        return (
            f"{self.message_id} | {self.created_at} | {text} | "
            f"{self.media_type or '-'} | {self.forwarded_count} | {self.reaction}"
        )


def _reaction(recipient: str, message_id: str, events) -> str:
    """Reduce the event row for (recipient, message_id) to one word."""
    event = events.get((recipient, message_id))
    if event is None:
        return "no_record"
    for column, label in _REACTION_ORDER:
        if event.get(column) == "1":
            return label
    return "no_record"


def _to_candidate(row, recipient: str, events) -> Candidate:
    return Candidate(
        message_id=row["message_id"],
        created_at=row["created_at"],
        text=row.get("message_text", ""),
        media_type=row.get("media_type", ""),
        forwarded_count=row.get("forwarded_count", "0"),
        reaction=_reaction(recipient, row["message_id"], events),
    )


def build_candidate_pool(message_row, indices) -> list[Candidate]:
    """Every historical message this recipient had with this counterparty.

    Scope depends on ``conversation_type``:

    * ``business`` — history where ``user_id == recipient`` and
      ``business_id`` matches
    * ``group`` — history where ``user_id == recipient`` and ``group_id``
      matches, with rows from the same ``sender_user_id`` listed first
    * ``personal`` — history where ``user_id == recipient`` and
      ``sender_user_id`` matches

    Within each partition, newest first. The pool is never truncated.
    """
    recipient = message_row["user_id"]
    conversation_type = message_row.get("conversation_type", "")
    events = indices.message_events

    mine = [row for row in indices.message_history if row["user_id"] == recipient]

    if conversation_type == "business":
        business_id = message_row.get("business_id", "")
        if not business_id:
            return []
        scoped = [row for row in mine if row.get("business_id") == business_id]
        scoped.sort(key=lambda r: r["created_at"], reverse=True)
        return [_to_candidate(r, recipient, events) for r in scoped]

    if conversation_type == "group":
        group_id = message_row.get("group_id", "")
        if not group_id:
            return []
        scoped = [row for row in mine if row.get("group_id") == group_id]
        sender_id = message_row.get("sender_user_id", "")
        same_sender = [r for r in scoped if sender_id and r.get("sender_user_id") == sender_id]
        others = [r for r in scoped if not (sender_id and r.get("sender_user_id") == sender_id)]
        same_sender.sort(key=lambda r: r["created_at"], reverse=True)
        others.sort(key=lambda r: r["created_at"], reverse=True)
        return [_to_candidate(r, recipient, events) for r in same_sender + others]

    sender_id = message_row.get("sender_user_id", "")
    if not sender_id:
        return []
    scoped = [row for row in mine if row.get("sender_user_id") == sender_id]
    scoped.sort(key=lambda r: r["created_at"], reverse=True)
    return [_to_candidate(r, recipient, events) for r in scoped]


def enforce_evidence(
    selected_ids, pool: list[Candidate]
) -> tuple[list[str], bool]:
    """Constrain the model's evidence selection to the supplied pool.

    Enforced here rather than requested in the prompt, so that a model that
    invents an ID cannot put one in ``output.csv`` no matter what it returns.

    Returns ``(ids, violation)``. ``violation`` is True when anything was
    dropped, for either reason:

    * an ID that is not in the pool — the hard failure the caller retries once,
      naming the violation, before falling back to ``none``
    * more than :data:`schema.MAX_EVIDENCE_IDS` ids, or a repeated id — both
      mean the model returned a shape the tool schema already forbids

    An empty pool short-circuits to ``([], False)`` **without reading
    ``selected_ids`` at all**: there was nothing to select from, so there is no
    model behaviour to judge. ``none`` is the correct answer, not a failure.
    """
    if not pool:
        return [], False

    allowed = {candidate.message_id for candidate in pool}
    kept: list[str] = []
    dropped = False

    for message_id in selected_ids or ():
        if message_id in allowed and message_id not in kept:
            kept.append(message_id)
        else:
            dropped = True

    if len(kept) > schema.MAX_EVIDENCE_IDS:
        kept = kept[: schema.MAX_EVIDENCE_IDS]
        dropped = True

    return kept, dropped
