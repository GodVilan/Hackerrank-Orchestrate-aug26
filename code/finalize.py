"""Turn a Decision into the output row. Adds no reasoning of its own.

``finalize(decision, features, model_read, evidence_ids) -> dict`` with exactly
``schema.OUTPUT_COLUMNS``.

The reason string is looked up from the catalogue by rule key. The model's prose
never reaches the output — two rows that fired the same branch are byte-identical
in the ``reason`` column by construction.

Variant selection (order vs booking vs payment) happens in
``gates._notify_sub_rule`` from ``why_user_knows_account``, which is a joined
field. It is deliberately **not** duplicated here: two copies of the same
mapping in two modules is how they drift apart. ``test_finalize`` asserts the
selected variant matches the joined field for every business row.
"""

from __future__ import annotations

import schema
from gates import Decision
from schema import RuleKey

#: The reading handed to the ladder for a row that could not be routed.
#: Every field is the least-committal value its enum allows. Under this reading
#: no Layer 1 content branch and no Layer 4 branch can fire, so only branches
#: that rest on joined features alone can produce a verdict.
NEUTRAL_READ = schema.ModelRead(
    message_type="unknown",
    content_risk="none",
    urgency="none",
    promotional=False,
    is_router_injection_attempt=False,
    asks_user_for_action=False,
    evidence_message_ids=(),
    proposed_action="digest",
    media_summary="",
)

#: Rule keys a degraded row is allowed to keep, because the branch that emits
#: them rests on joined features rather than on anything the model said.
#:
#: MUTE_OPTED_OUT_MARKETING and DIGEST_QUIET_HOURS are listed for completeness —
#: their branches read `promotional` and `urgency`, so under NEUTRAL_READ they
#: cannot fire at all. They are here so the set reads as the policy rather than
#: as an artifact of what happens to be reachable.
DETERMINISTIC_RULE_KEYS = frozenset(
    {
        RuleKey.MUTE_MUTED_GROUP,
        RuleKey.MUTE_IMPERSONATION_DOMAIN,
        RuleKey.MUTE_HIGH_REPORT_SENDER,
        RuleKey.MUTE_FORWARD_PATTERN,
        RuleKey.MUTE_SIMILAR_IGNORED,
        RuleKey.MUTE_OPTED_OUT_MARKETING,
        RuleKey.DIGEST_QUIET_HOURS,
    }
)


def degrade(decision: Decision) -> Decision:
    """Reduce a neutral-reading ladder result to what is actually defensible.

    A row that failed to route still has all its joined features, so a branch
    that fired on those alone — muted group, impersonation signature, reported
    sender, forwarding pattern — is as sound as it would have been on a healthy
    row. That verdict is kept, with its real action, rule key, and reason.

    Anything else would be a content judgement, and no content was read. Those
    rows become DIGEST_INSUFFICIENT_SIGNAL, whose catalogue string says exactly
    that. The alternative — emitting DIGEST_GROUP_INFO with a reason claiming
    the message is useful group information — would be a fabricated finding
    rather than a degraded one.

    ``tier_signals`` is cleared either way: the confidence tier is pinned to 0
    for every degraded row.
    """
    if decision.rule_key in DETERMINISTIC_RULE_KEYS:
        return Decision(decision.action, decision.rule_key, [], None)
    return Decision("digest", RuleKey.DIGEST_INSUFFICIENT_SIGNAL, [], None)


def confidence_tier(
    decision: Decision, features, model_read, evidence_ids, pool_size: int
) -> tuple[int, list[str]]:
    """Count independent corroborating signals, capped at 3.

    The four candidate signals, per README "Confidence":

    1. a deterministic feature directly supports the branch that fired
    2. sender history is at least moderate, or there is a business relationship
       with recorded activity
    3. the evidence pool was non-empty and at least one id was selected
    4. the model's own reading agrees with the ladder's action

    **Override:** when the model's ``proposed_action`` disagrees with the
    ladder, the tier is pinned to 0 regardless of the other three. A row where
    the semantic read and the deterministic ladder point in different
    directions is the least trustworthy row in the file, and the confidence
    column should say so.
    """
    if model_read.proposed_action != decision.action:
        return 0, ["disagreement_override"]

    held: list[str] = []
    if decision.tier_signals:
        held.append("deterministic_support")
    if features.sender_history_strength in {"moderate", "strong"} or (
        features.has_relationship and features.activity_count_180d > 0
    ):
        held.append("counterparty_history")
    if pool_size > 0 and evidence_ids:
        held.append("evidence_selected")
    held.append("model_agrees")  # reached only when proposed_action == action

    return min(len(held), schema.CONFIDENCE_MAX_TIER), held


def confidence_for(action: str, tier: int) -> float:
    """Band floor plus one step per corroborating signal, clamped and rounded."""
    raw = schema.CONFIDENCE_BASE[action] + schema.CONFIDENCE_STEP * tier
    clamped = min(max(raw, schema.CONFIDENCE_MIN), schema.CONFIDENCE_MAX)
    return round(clamped, 2)


#: How many of the model's selected ids actually reach ``output.csv``.
#:
#: The tool schema still permits 2 (``schema.MAX_EVIDENCE_IDS``) and the pool
#: enforcement in ``evidence.py`` still validates up to 2 — this truncates at
#: the emission step only. Measured on the labeled rows: the model returned a
#: second id on most rows and gold cites exactly one on 25 of 30, so the second
#: id was a false positive far more often than a true one. Keeping only the
#: first raised evidence F1 and roughly doubled exact-set-match.
EMITTED_EVIDENCE_IDS = 1


def format_evidence(evidence_ids) -> str:
    """The first selected id, or the `none` sentinel. Never an empty string."""
    ids = [i for i in (evidence_ids or ()) if i]
    if not ids:
        return schema.EVIDENCE_NONE
    return schema.EVIDENCE_SEPARATOR.join(ids[:EMITTED_EVIDENCE_IDS])


def finalize(
    decision: Decision,
    features,
    model_read,
    evidence_ids,
    *,
    pool_size: int = 0,
    degraded: bool = False,
) -> dict:
    """Build one output row with exactly ``schema.OUTPUT_COLUMNS``.

    ``degraded`` pins the tier to 0 and forces empty evidence: a row we could
    not read has no corroboration to count and no grounds to cite history.
    """
    if degraded:
        tier, signals = 0, ["degraded"]
        evidence_ids = ()
    else:
        tier, signals = confidence_tier(
            decision, features, model_read, evidence_ids, pool_size
        )

    row = {
        "message_id": features.message_id,
        "action": decision.action,
        "message_type": model_read.message_type,
        "reason": schema.REASONS[decision.rule_key],
        "confidence": f"{confidence_for(decision.action, tier):.2f}",
        "evidence_message_ids": format_evidence(evidence_ids),
    }

    # Invariants that must hold for every row, degraded or not.
    assert row["action"] in schema.ACTIONS, row
    assert schema.RULE_ACTION[decision.rule_key] == row["action"], (
        f"{features.message_id}: rule_key {decision.rule_key.value} implies "
        f"{schema.RULE_ACTION[decision.rule_key]}, row says {row['action']}"
    )
    assert row["message_type"] in schema.MESSAGE_TYPES, row
    assert row["reason"], row
    assert set(row) == set(schema.OUTPUT_COLUMNS), row

    row["_tier"] = tier
    row["_tier_signals"] = signals
    row["_rule_key"] = decision.rule_key.value
    return row
