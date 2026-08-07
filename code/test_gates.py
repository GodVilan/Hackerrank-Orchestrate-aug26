"""Contract tests for the precedence ladder.

    python code/test_gates.py        # standalone
    pytest code/test_gates.py        # also works

Pure: hand-built fixtures, no dataset read, no network. That is the point of
``gates.decide`` being a pure function — the ordering claim is testable without
spending a token.

One case is different. ``test_injection_sample_msg_053`` is built from the real
feature values of the only labeled row that exercises Layer 1's injection
branch, frozen here as literals. Every other case tests the ladder against my
own assumptions about what a row looks like; that one tests it against a row the
organizers labeled.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gates  # noqa: E402
from features import AdvisoryFlags, MessageFeatures  # noqa: E402
import schema as schema_module  # noqa: E402
from schema import ModelRead, RuleKey  # noqa: E402


def mf(**overrides) -> MessageFeatures:
    """A neutral feature block. Every test states only what it depends on."""
    base = dict(
        message_id="test_msg",
        user_id="u_test",
        conversation_type="group",
        created_at="2026-07-31 12:00",
    )
    base.update(overrides)
    return MessageFeatures(**base)


def read(**overrides) -> ModelRead:
    """A benign model reading. Overrides carry the signal under test."""
    base = dict(
        message_type="personal",
        content_risk="none",
        urgency="none",
        promotional=False,
        is_router_injection_attempt=False,
        asks_user_for_action=False,
        proposed_action="digest",
    )
    base.update(overrides)
    return ModelRead(**base)


# --------------------------------------------------------------------------
# The eight required cases
# --------------------------------------------------------------------------


def test_muted_group_without_mention() -> None:
    d = gates.decide(
        mf(is_group=True, group_muted_by_user=True, is_direct_mention=False),
        read(),
    )
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_MUTED_GROUP, d


def test_muted_group_with_mention_falls_through_to_notify() -> None:
    """The mute is lifted, not converted to notify — Layers 3-4 still decide."""
    d = gates.decide(
        mf(
            is_group=True,
            group_muted_by_user=True,
            is_direct_mention=True,
            has_sender=True,
            n_prior=8,
        ),
        read(asks_user_for_action=True),
    )
    assert d.action == "notify", d
    assert d.rule_key is RuleKey.NOTIFY_MENTION_IN_MUTED_GROUP, d


def test_muted_group_with_mention_but_nothing_to_notify_about() -> None:
    """Lifting the mute is not the same as forcing a notify."""
    d = gates.decide(
        mf(
            is_group=True,
            group_muted_by_user=True,
            is_direct_mention=True,
            has_sender=True,
            n_prior=8,
        ),
        read(asks_user_for_action=False, message_type="greeting"),
    )
    assert d.action == "digest", d
    assert d.rule_key is RuleKey.DIGEST_GREETING, d


def test_opted_out_promotion() -> None:
    d = gates.decide(
        mf(
            conversation_type="business",
            is_business=True,
            business_id="business_x",
            has_relationship=True,
            opted_out=True,
            promotions_opted_out_at="2026-07-17 23:55",
        ),
        read(promotional=True, message_type="promotion"),
    )
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_OPTED_OUT_MARKETING, d


def test_quiet_hours_downgrades_notify_to_digest() -> None:
    d = gates.decide(
        mf(
            is_group=True,
            dnd_active=True,
            is_direct_mention=True,
            has_sender=True,
            n_prior=8,
        ),
        read(urgency="low", asks_user_for_action=True),
    )
    assert d.action == "digest", d
    assert d.rule_key is RuleKey.DIGEST_QUIET_HOURS, d
    assert d.downgraded_from == ("notify", RuleKey.NOTIFY_DIRECT_REQUEST), d


def test_injection_attempt() -> None:
    d = gates.decide(mf(), read(is_router_injection_attempt=True))
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_INJECTION_ATTEMPT, d


def test_verified_business_matching_an_order() -> None:
    d = gates.decide(
        mf(
            conversation_type="business",
            is_business=True,
            business_id="business_y",
            verified=True,
            domain_match=True,
            has_relationship=True,
            why_user_knows_account="recent_grocery_delivery",
            activity_count_180d=3,
        ),
        read(urgency="high", message_type="business_update"),
    )
    assert d.action == "notify", d
    assert d.rule_key is RuleKey.NOTIFY_BIZ_MATCHES_ORDER, d
    assert "verified_business_relationship" in d.tier_signals, d


def test_unknown_sender_asking_for_an_otp() -> None:
    d = gates.decide(
        mf(
            conversation_type="personal",
            has_sender=True,
            sender_user_id="u_new",
            n_prior=0,
            advisory=AdvisoryFlags(credential_language_flag=True),
        ),
        read(content_risk="scam", message_type="scam", urgency="high"),
    )
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_SCAM_OTP, d


def test_injection_sample_msg_053() -> None:
    """The one labeled row that exercises Layer 1's injection branch.

    Feature values below are the real ones for sample_msg_053, captured once
    from the dataset and frozen so this suite stays pure. Gold: mute / scam.

    Note n_prior is 12 and the recipient opens 83% of this sender's messages —
    a strong positive history. The injection branch must win anyway.
    """
    features = mf(
        message_id="sample_msg_053",
        user_id="u_009",
        conversation_type="personal",
        created_at="2026-07-31 11:22",
        dnd_window="22:00-08:30",
        user_open_rate_30d=0.9077,
        user_dismiss_rate_30d=0.0923,
        user_engagement_denom_30d=130,
        trailing_daily_load=9.3571,
        trailing_daily_days=14,
        has_sender=True,
        sender_user_id="u_050",
        n_prior=12,
        open_rate=0.8333,
        dismiss_rate=0.1667,
        sender_history_strength="strong",
        global_sender_n=22,
        global_sender_open_rate=0.4545,
        global_sender_dismiss_rate=0.5455,
        global_sender_report_rate=0.4545,
        advisory=AdvisoryFlags(
            credential_language_flag=True, injection_pattern_flag=True
        ),
    )
    d = gates.decide(
        features,
        read(
            is_router_injection_attempt=True,
            content_risk="scam",
            message_type="scam",
            urgency="high",
            proposed_action="mute",
        ),
    )
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_INJECTION_ATTEMPT, d


# --------------------------------------------------------------------------
# The three cases the resolved decisions make load-bearing
# --------------------------------------------------------------------------


def test_quiet_hours_never_upgrades_a_mute() -> None:
    """Decision 2's entire point: the downgrade applies only over notify."""
    features = mf(
        is_group=True,
        dnd_active=True,
        has_sender=True,
        n_prior=11,
        dismiss_rate=1.0,
        mean_forwarded_count=6.55,
        forwarded_count=6,
    )
    d = gates.decide(features, read(message_type="greeting", urgency="low"))
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_FORWARD_PATTERN, d
    assert d.downgraded_from is None, d


def test_suspicious_personal_from_known_sender_is_not_first_contact() -> None:
    """Decision 1's entire point: has_relationship is business-only."""
    features = mf(
        conversation_type="personal",
        has_sender=True,
        sender_user_id="u_known",
        n_prior=23,
        sender_history_strength="strong",
        has_relationship=False,  # business-only field; False on every personal row
    )
    d = gates.decide(features, read(content_risk="suspicious"))
    assert d.rule_key is not RuleKey.MUTE_SCAM_FIRST_CONTACT, d
    assert d.action != "mute", d


def test_suspicious_personal_from_unknown_sender_is_first_contact() -> None:
    """The same branch must still fire when the sender is genuinely new."""
    features = mf(
        conversation_type="personal", has_sender=True, sender_user_id="u_new", n_prior=0
    )
    d = gates.decide(features, read(content_risk="suspicious"))
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_SCAM_FIRST_CONTACT, d


def test_unsolicited_business_does_not_fire_on_a_group_promotion() -> None:
    """A neighbour's for-sale post is not an unsolicited business message."""
    features = mf(
        is_group=True,
        group_type="marketplace",
        has_sender=True,
        sender_user_id="u_peer",
        n_prior=7,
    )
    d = gates.decide(features, read(promotional=True, message_type="promotion"))
    assert d.rule_key is not RuleKey.MUTE_UNSOLICITED_BUSINESS, d
    assert d.action != "mute", d


def test_high_global_report_sender_does_not_mute_on_first_contact() -> None:
    """Variant D: reputation alone must not mute a stranger's first message."""
    features = mf(
        conversation_type="personal",
        has_sender=True,
        sender_user_id="u_reported",
        n_prior=0,
        report_rate=0.0,
        global_sender_n=21,
        global_sender_report_rate=0.6667,
    )
    d = gates.decide(features, read())
    assert d.rule_key is not RuleKey.MUTE_HIGH_REPORT_SENDER, d


def test_high_global_report_sender_mutes_once_there_is_any_history() -> None:
    """One prior interaction is enough to let the global reputation bind."""
    features = mf(
        conversation_type="personal",
        has_sender=True,
        sender_user_id="u_reported",
        n_prior=1,
        report_rate=0.0,  # this recipient has never reported them
        open_rate=1.0,  # and opens everything they send
        global_sender_n=21,
        global_sender_report_rate=0.6667,
    )
    d = gates.decide(features, read())
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_HIGH_REPORT_SENDER, d
    assert "global_sender_report_rate" in d.tier_signals, d


def test_high_global_report_beats_strong_positive_per_pair_history() -> None:
    """The four missed rows in one case: n_prior=12, 83% open, still muted."""
    features = mf(
        conversation_type="group",
        is_group=True,
        has_sender=True,
        sender_user_id="u_050",
        n_prior=12,
        open_rate=0.8333,
        report_rate=0.0,
        sender_history_strength="strong",
        global_sender_n=22,
        global_sender_report_rate=0.4545,
    )
    d = gates.decide(features, read())
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_HIGH_REPORT_SENDER, d


def test_globally_clean_sender_is_not_muted_by_a_local_report() -> None:
    """The known gap (msg_046): documented, not silently passing."""
    features = mf(
        conversation_type="group",
        is_group=True,
        has_sender=True,
        sender_user_id="u_041",
        n_prior=2,
        report_rate=1.0,
        global_sender_n=24,
        global_sender_report_rate=0.0833,
    )
    d = gates.decide(features, read())
    assert d.rule_key is not RuleKey.MUTE_HIGH_REPORT_SENDER, d


def test_impersonation_beats_positive_engagement() -> None:
    """Layer 1 cannot be overridden by Layer 3 history."""
    features = mf(
        conversation_type="business",
        is_business=True,
        business_id="business_fake",
        verified=False,
        official_domain="chase.com",
        domain_used_by_sender="chase-secure-alert.com",
        domain_match=False,
        official_domain_missing=False,
        account_age_days=24,
        domain_used_by_sender_age_days=10,
        report_rate_per_1k=48.08,
        has_relationship=True,
        biz_open_rate=1.0,
    )
    d = gates.decide(features, read(urgency="high", message_type="payment"))
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_IMPERSONATION_DOMAIN, d


# --------------------------------------------------------------------------
# Pinned: verified-brand exemption and the hoisted business-match branch.
# Each of these fails if the change is silently reverted.
# --------------------------------------------------------------------------


def _verified_brand(**overrides) -> MessageFeatures:
    base = dict(
        conversation_type="business",
        is_business=True,
        business_id="business_ok",
        brand_name="Verified Brand",
        verified=True,
        official_domain="brand.com",
        domain_used_by_sender="brand.com",
        domain_match=True,
        official_domain_missing=False,
        account_age_days=4267,
        report_rate_per_1k=0.59,
        has_relationship=False,
        n_prior=0,
    )
    base.update(overrides)
    return mf(**base)


def test_verified_brand_is_exempt_from_first_contact_mute() -> None:
    """1. verified + domain match + suspicious + no relationship -> NOT muted."""
    d = gates.decide(_verified_brand(), read(content_risk="suspicious"))
    assert d.rule_key is not RuleKey.MUTE_SCAM_FIRST_CONTACT, d
    assert d.action != "mute", d


def test_unverified_sender_still_mutes_as_first_contact() -> None:
    """2. same inputs, unverified -> the branch DOES fire."""
    d = gates.decide(
        _verified_brand(verified=False, domain_match=False),
        read(content_risk="suspicious"),
    )
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_SCAM_FIRST_CONTACT, d


def test_verified_brand_exemption_also_applies_to_the_scam_sub_rule() -> None:
    """The Layer 1 branch and _scam_sub_rule must not diverge."""
    key = gates._scam_sub_rule(_verified_brand(), read(content_risk="scam"))
    assert key is not RuleKey.MUTE_SCAM_FIRST_CONTACT, key
    key_unverified = gates._scam_sub_rule(
        _verified_brand(verified=False, domain_match=False), read(content_risk="scam")
    )
    assert key_unverified is RuleKey.MUTE_SCAM_FIRST_CONTACT, key_unverified


def test_impersonation_still_fires_on_a_verified_sender() -> None:
    """3. The exemption must not weaken the impersonation signature."""
    features = _verified_brand(
        verified=False,  # signature requires verified == 0
        domain_match=False,
        domain_used_by_sender="brand-secure.in",
        account_age_days=24,
        domain_used_by_sender_age_days=10,
        report_rate_per_1k=48.08,
    )
    assert gates.impersonation_signature(features) is True
    d = gates.decide(features, read(content_risk="suspicious"))
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_IMPERSONATION_DOMAIN, d


def test_business_match_does_not_notify_below_high_urgency() -> None:
    """4. The business-match branch lives in Layer 4, behind `urgency == high`.

    The Layer 3 hoist of this branch was reverted: it converted 7 of the 110
    shipped rows from digest to notify, including a cinema feedback survey and
    an opted-in promotion the model had read as suspicious, and it made
    DIGEST_PROMO_OPTED_IN unreachable for every verified business with a
    relationship. It bought one row on the 30 labeled rows, which is inside the
    noise band. This test pins the restored ordering so the hoist cannot return
    silently.
    """
    features = _verified_brand(
        has_relationship=True,
        why_user_knows_account="recent_grocery_delivery",
        activity_count_180d=3,
    )
    d = gates.decide(features, read(urgency="low", message_type="business_update"))
    assert d.action == "digest", d
    assert d.rule_key is not RuleKey.NOTIFY_BIZ_MATCHES_ORDER, d


def test_business_match_selects_the_booking_variant_at_high_urgency() -> None:
    """The variant selector still reads the joined field, now from Layer 4."""
    features = _verified_brand(
        has_relationship=True,
        why_user_knows_account="upcoming_clinic_appointment",
        activity_count_180d=2,
    )
    d = gates.decide(features, read(urgency="high"))
    assert d.action == "notify", d
    assert d.rule_key is RuleKey.NOTIFY_BIZ_MATCHES_BOOKING, d


def test_business_match_is_gated_on_urgency() -> None:
    """Urgency is now the gate: notify only at 'high', digest otherwise."""
    features = _verified_brand(
        has_relationship=True,
        why_user_knows_account="recent_grocery_delivery",
        activity_count_180d=3,
    )
    by_urgency = {
        u: gates.decide(features, read(urgency=u)) for u in ("none", "low", "high")
    }
    assert by_urgency["high"].rule_key is RuleKey.NOTIFY_BIZ_MATCHES_ORDER, by_urgency
    assert by_urgency["none"].action == "digest", by_urgency
    assert by_urgency["low"].action == "digest", by_urgency


def test_group_mute_beats_a_verified_business_relationship() -> None:
    """Layer 2's group mute wins over a verified brand the user transacts with.

    Originally this guarded Layer 2 against the *adjacent* Layer 3 hoisted
    business-notify branch; that branch was reverted and no longer exists, so
    the case it covered is gone and this now guards Layer 2 against Layer 4 —
    a weaker assertion, since two layers separate them rather than one.
    """
    features = _verified_brand(
        conversation_type="group",
        is_group=True,
        group_muted_by_user=True,
        is_direct_mention=False,
        has_relationship=True,
        why_user_knows_account="recent_grocery_delivery",
        activity_count_180d=3,
    )
    d = gates.decide(features, read(urgency="low"))
    assert d.action == "mute", d
    assert d.rule_key is RuleKey.MUTE_MUTED_GROUP, d
    # Still meaningful at high urgency, where Layer 4 would otherwise notify.
    d_high = gates.decide(features, read(urgency="high"))
    assert d_high.rule_key is RuleKey.MUTE_MUTED_GROUP, d_high


def test_business_match_requires_every_conjunct() -> None:
    """Dropping any one conjunct must stop the branch firing."""
    ok = dict(
        has_relationship=True,
        why_user_knows_account="recent_grocery_delivery",
        activity_count_180d=3,
    )
    for drop in (
        {"verified": False},
        {"domain_match": False},
        {"has_relationship": False},
        {"activity_count_180d": 0},
    ):
        features = _verified_brand(**{**ok, **drop})
        d = gates.decide(features, read(urgency="low"))
        assert d.rule_key is not RuleKey.NOTIFY_BIZ_MATCHES_ORDER, (drop, d)


def test_every_rule_key_maps_to_its_action() -> None:
    """A branch that returns an action its rule key contradicts is a bug."""
    from schema import RULE_ACTION

    for key, action in RULE_ACTION.items():
        assert key.value.lower().startswith(action), (key, action)


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())


def test_finalize_emits_only_the_first_evidence_id() -> None:
    """6. Two ids in, exactly one out, and it is the first."""
    import finalize as fin

    assert fin.format_evidence(["message_0001", "message_0002"]) == "message_0001"
    assert fin.format_evidence(["message_0009"]) == "message_0009"
    assert fin.format_evidence([]) == schema_module.EVIDENCE_NONE
    assert fin.format_evidence(None) == schema_module.EVIDENCE_NONE
    # The schema cap itself must be untouched.
    assert schema_module.MAX_EVIDENCE_IDS == 2
    assert fin.EMITTED_EVIDENCE_IDS == 1
