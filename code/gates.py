"""The precedence ladder. A pure function: no I/O, no model calls.

``decide(features, model_read) -> Decision`` is the only entry point. It never
imports ``router``, never touches the filesystem, and never reads
``features.advisory`` — the two regex pre-flags are context for the model, not
inputs to a decision.

Layer order is the whole design. Each layer overrides everything below it, and a
branch that fires returns immediately, so content reasoning cannot invert an
integrity or consent decision — it is never reached when those fire.

  Layer 1  SAFETY            scam, impersonation, injection, reported senders
  Layer 2  HARD USER STATE   group mute, promotion opt-out
  Layer 3  PERSONALIZATION   this user's history with this counterparty
  Layer 4  CONTENT SUBSTANCE what the message actually says

Two amendments to the ladder as originally specified, both recorded in
code/README.md:

* **Quiet hours are a downgrade modifier, not a Layer 2 branch.** As a returning
  branch they would pre-empt Layers 3 and 4 and convert a mute into a digest —
  an upgrade, contradicting "quiet hours never mute". :func:`_ladder` therefore
  contains no DND branch; :func:`decide` applies the downgrade afterwards and
  only over a ``notify``.
* **Layer 1's first-contact branch reads**
  :func:`features.is_known_counterparty`, not ``has_relationship``, which is
  business-only and would otherwise mark every group and personal sender as a
  first contact.

Layer 4's digest branches were left open in the specification. The mapping
implemented below runs in this order, first match wins:

  promotion       + allows_promotions        -> DIGEST_PROMO_OPTED_IN
  promotion       (business, has_relationship)-> DIGEST_MATCHES_INTEREST
  business_update + verified                 -> DIGEST_BIZ_LEGIT
  business_update (unverified)               -> DIGEST_BIZ_NON_URGENT
  payment         (business, verified)       -> DIGEST_BIZ_LEGIT
  event                                      -> DIGEST_EVENT_FUTURE
  greeting                                   -> DIGEST_GREETING
  personal        + known counterparty       -> DIGEST_CASUAL
  personal        (unknown counterparty)     -> DIGEST_UNKNOWN_SENDER_BENIGN
  any             + strong sender history    -> DIGEST_TRUSTED_NON_URGENT
  any             + unknown counterparty     -> DIGEST_UNKNOWN_SENDER_BENIGN
  default                                    -> DIGEST_GROUP_INFO
"""

from __future__ import annotations

from typing import NamedTuple

import config
from features import MessageFeatures, is_known_counterparty
from schema import ModelRead, RuleKey


class Decision(NamedTuple):
    """What the ladder concluded, and what supported it.

    ``tier_signals`` names the deterministic features that backed the branch
    that fired; ``finalize.py`` counts them into the confidence tier. It is a
    list of names, not a number, so the reason a row is confident stays
    inspectable.

    ``downgraded_from`` is set only by the quiet-hours modifier and carries the
    pre-downgrade ``(action, rule_key)`` so ``run_stats.json`` can record what
    the ladder concluded before the downgrade. Three-tuple unpacking still
    works, since the field defaults to None.
    """

    action: str
    rule_key: RuleKey
    tier_signals: list[str]
    downgraded_from: tuple[str, RuleKey] | None = None


# --------------------------------------------------------------------------
# Layer 1 helper
# --------------------------------------------------------------------------


def impersonation_signature(features: MessageFeatures) -> bool:
    """Is this business sending from a domain it does not own?

    All conjuncts must hold together. An absent ``official_domain`` is not a
    mismatch — there is nothing to compare against — which is what keeps this
    off legitimate accounts that simply never registered one.

    Thresholds and their basis live in ``config.py``.
    """
    if not features.is_business:
        return False
    if features.verified:
        return False
    if features.official_domain_missing:
        return False
    if features.domain_match:
        return False
    young = (
        features.account_age_days < config.IMPERSONATION_MAX_ACCOUNT_AGE_DAYS
        or features.domain_used_by_sender_age_days
        < config.IMPERSONATION_MAX_DOMAIN_AGE_DAYS
    )
    if not young:
        return False
    return features.report_rate_per_1k > config.T_REPORT_PER_1K


def is_verified_brand(features: MessageFeatures) -> bool:
    """A verified business sending from the domain it actually owns.

    The exemption for the first-contact branches. "First contact" is a proxy for
    "we cannot vouch for this sender", and it is the wrong proxy when the
    platform's own records already vouch for them: verification plus a matching
    domain is a stronger identity signal than the absence of a
    ``user_business_history`` row is a risk signal — and 11 of the 30 business
    rows have no such row at all.

    Deliberately narrow. It does **not** exempt anything from the impersonation
    signature, which fires on report rate and domain mismatch and must keep
    firing on verified senders.
    """
    return features.is_business and features.verified and features.domain_match


def _biz_variant(features: MessageFeatures) -> RuleKey:
    """Select the business notify variant from ``why_user_knows_account``.

    One selector, used by both the Layer 3 business-match branch and the Layer 4
    urgency sub-rule, so the two cannot drift apart. The field is a joined value;
    the model's text is never consulted.
    """
    why = features.why_user_knows_account
    if any(token in why for token in ("booking", "reservation", "appointment")):
        return RuleKey.NOTIFY_BIZ_MATCHES_BOOKING
    if any(token in why for token in ("payment", "bill", "wallet", "card", "bank")):
        return RuleKey.NOTIFY_PAYMENT_LEGIT
    return RuleKey.NOTIFY_BIZ_MATCHES_ORDER


def _scam_sub_rule(features: MessageFeatures, model_read: ModelRead) -> RuleKey:
    """Pick the scam variant. Credential framing first, then support framing."""
    text = model_read.media_summary.lower()
    if features.advisory.credential_language_flag or "otp" in text:
        return RuleKey.MUTE_SCAM_OTP
    if model_read.message_type in {"business_update", "payment"} and (
        features.is_business and not features.verified
    ):
        return RuleKey.MUTE_SCAM_FAKE_SUPPORT
    # Same verified-brand exemption as the Layer 1 first-contact branch. The two
    # must not diverge: both answer "is an absent relationship row evidence of
    # risk here?", and for a verified brand on its own domain it is not.
    if (features.n_prior == 0 or not is_known_counterparty(features)) and not (
        is_verified_brand(features)
    ):
        return RuleKey.MUTE_SCAM_FIRST_CONTACT
    return RuleKey.MUTE_SCAM_OTP


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def _ladder(
    features: MessageFeatures,
    model_read: ModelRead,
    *,
    skip_safety_and_state: bool = False,
) -> Decision:
    """Layers 1 to 4. Contains no quiet-hours branch by design.

    ``skip_safety_and_state`` is the Config B ablation switch and is False on
    every shipping path. It removes Layers 1 and 2 only; Layers 3 and 4 are
    unchanged, so the ablation measures what those two layers contribute rather
    than replacing the whole ladder.
    """
    signals: list[str] = []
    mention_lifts_mute = False

    if not skip_safety_and_state:
        # ---- LAYER 1 · SAFETY ---------------------------------------------
        if model_read.is_router_injection_attempt:
            return Decision("mute", RuleKey.MUTE_INJECTION_ATTEMPT, signals)

        if model_read.content_risk == "scam":
            if features.report_rate_per_1k > config.T_REPORT_PER_1K:
                signals.append("report_rate_per_1k")
            if features.report_rate >= config.T_REPORT:
                signals.append("sender_report_rate")
            return Decision("mute", _scam_sub_rule(features, model_read), signals)

        if impersonation_signature(features):
            signals.append("impersonation_signature")
            return Decision("mute", RuleKey.MUTE_IMPERSONATION_DOMAIN, signals)

        # Variant D: the sender's reputation across ALL recipients, not this
        # one's experience. `n_prior > 0` suppresses first contacts, which fall
        # through to the suspicious-and-unknown branch below and then to Layer
        # 4, so a stranger is judged on what they wrote rather than on who else
        # has reported them.
        if (
            features.global_sender_report_rate >= config.T_REPORT
            and features.global_sender_n >= 3
            and features.n_prior > 0
        ):
            signals.append("global_sender_report_rate")
            return Decision("mute", RuleKey.MUTE_HIGH_REPORT_SENDER, signals)

        if (
            model_read.content_risk == "suspicious"
            and not is_known_counterparty(features)
            and not is_verified_brand(features)
        ):
            signals.append("no_prior_counterparty")
            return Decision("mute", RuleKey.MUTE_SCAM_FIRST_CONTACT, signals)

        # ---- LAYER 2 · HARD USER STATE ------------------------------------
        # A muted group with a direct mention does NOT return here. The mute is
        # lifted, not converted to notify: control falls through and Layers 3-4
        # decide on the merits, with NOTIFY_MENTION_IN_MUTED_GROUP substituted
        # for whichever notify rule fires.
        if features.group_muted_by_user:
            if not features.is_direct_mention:
                signals.append("group_muted_by_user")
                return Decision("mute", RuleKey.MUTE_MUTED_GROUP, signals)
            mention_lifts_mute = True
            signals.append("direct_mention_in_muted_group")

        if features.opted_out and model_read.promotional:
            signals.append("promotions_opted_out")
            return Decision("mute", RuleKey.MUTE_OPTED_OUT_MARKETING, signals)

    # ---- LAYER 3 · PERSONALIZATION ---------------------------------------
    if features.dismiss_rate >= config.T_DISMISS and features.n_prior >= 5:
        signals.append("sender_dismiss_rate")
        key = (
            RuleKey.MUTE_FORWARD_PATTERN
            if features.mean_forwarded_count >= config.T_FWD_SENDER_MEAN
            else RuleKey.MUTE_SIMILAR_IGNORED
        )
        return Decision("mute", key, signals)

    if features.forwarded_count >= config.T_FWD_MESSAGE and model_read.message_type in {
        "forward",
        "greeting",
    }:
        signals.append("forwarded_count")
        return Decision("mute", RuleKey.MUTE_FORWARD_PATTERN, signals)

    if (
        model_read.promotional
        and features.is_business
        and features.has_relationship
        and not features.allows_promotions
    ):
        signals.append("business_relationship")
        return Decision("digest", RuleKey.DIGEST_OFFER_RELEVANT, signals)

    # Gated to business rows: "unsolicited business" is meaningless for a
    # neighbour's for-sale post in a marketplace group.
    if model_read.promotional and features.is_business and not features.has_relationship:
        signals.append("no_business_relationship")
        return Decision("mute", RuleKey.MUTE_UNSOLICITED_BUSINESS, signals)

    # ---- LAYER 4 · CONTENT SUBSTANCE -------------------------------------
    if model_read.urgency == "high":
        key = _notify_sub_rule(features)
        if mention_lifts_mute:
            key = RuleKey.NOTIFY_MENTION_IN_MUTED_GROUP
        _add_notify_signals(features, signals)
        return Decision("notify", key, signals)

    if features.is_direct_mention and model_read.asks_user_for_action:
        key = (
            RuleKey.NOTIFY_MENTION_IN_MUTED_GROUP
            if mention_lifts_mute
            else RuleKey.NOTIFY_DIRECT_REQUEST
        )
        signals.append("direct_mention")
        return Decision("notify", key, signals)

    return Decision("digest", _digest_sub_rule(features, model_read), signals)


def _notify_sub_rule(features: MessageFeatures) -> RuleKey:
    """Select the notify variant from joined context, never from model prose."""
    if features.is_business:
        return _biz_variant(features)
    if features.group_type == "school_group":
        return RuleKey.NOTIFY_SCHOOL_OPERATIONAL
    if features.group_type in {"coworker", "college_faculty"}:
        return RuleKey.NOTIFY_WORK_DEADLINE
    if features.sender_role == "admin":
        return RuleKey.NOTIFY_ADMIN_TIME_SENSITIVE
    return RuleKey.NOTIFY_CLOSE_CONTACT_URGENT


def _add_notify_signals(features: MessageFeatures, signals: list[str]) -> None:
    if features.is_business and features.verified and features.has_relationship:
        signals.append("verified_business_relationship")
    if features.sender_role == "admin":
        signals.append("sender_is_admin")


def _digest_sub_rule(features: MessageFeatures, model_read: ModelRead) -> RuleKey:
    """The terminal digest mapping. First match wins; order is documented above."""
    known = is_known_counterparty(features)
    mtype = model_read.message_type

    if mtype == "promotion":
        if features.allows_promotions:
            return RuleKey.DIGEST_PROMO_OPTED_IN
        if features.is_business and features.has_relationship:
            return RuleKey.DIGEST_MATCHES_INTEREST
    if mtype == "business_update":
        return (
            RuleKey.DIGEST_BIZ_LEGIT if features.verified else RuleKey.DIGEST_BIZ_NON_URGENT
        )
    if mtype == "payment" and features.is_business and features.verified:
        return RuleKey.DIGEST_BIZ_LEGIT
    if mtype == "event":
        return RuleKey.DIGEST_EVENT_FUTURE
    if mtype == "greeting":
        return RuleKey.DIGEST_GREETING
    if mtype == "personal":
        return RuleKey.DIGEST_CASUAL if known else RuleKey.DIGEST_UNKNOWN_SENDER_BENIGN
    if features.sender_history_strength == "strong":
        return RuleKey.DIGEST_TRUSTED_NON_URGENT
    if not known:
        return RuleKey.DIGEST_UNKNOWN_SENDER_BENIGN
    return RuleKey.DIGEST_GROUP_INFO


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _config_b(features: MessageFeatures, model_read: ModelRead) -> Decision:
    """The Config B ablation: Layers 1 and 2 removed, the model decides.

    ``action`` is ``model_read.proposed_action`` written straight through on
    every row, so the model's own verdict reaches ``output.csv`` unfiltered —
    including on the rows Layers 1 and 2 would have caught. That is the
    alternative Decision 1 rejected, made measurable.

    The reason column still needs a rule key, and ``finalize`` asserts that the
    key's implied action matches the row's action. When the shortened ladder
    already agrees with the model, its key is kept. When it does not, the key
    comes from the ladder's **own** sub-rule selectors for that action — no new
    mapping is invented here, and nothing in Config A reads this function.

    The quiet-hours downgrade is not applied: it is a Layer 2 modifier, and
    Layer 2 is what this ablation removes. It binds on zero rows either way.
    """
    base = _ladder(features, model_read, skip_safety_and_state=True)
    action = model_read.proposed_action
    if base.action == action:
        return Decision(action, base.rule_key, base.tier_signals)

    if action == "notify":
        key = _notify_sub_rule(features)
    elif action == "digest":
        key = _digest_sub_rule(features, model_read)
    else:
        key = _scam_sub_rule(features, model_read)
    return Decision(action, key, [])


def decide(
    features: MessageFeatures,
    model_read: ModelRead,
    *,
    ablate_layers_1_2: bool = False,
) -> Decision:
    """Run the ladder, then apply the quiet-hours downgrade.

    ``ablate_layers_1_2`` selects the Config B ablation and defaults to False on
    every shipping path; see :func:`_config_b`.

    The downgrade applies only when the ladder produced ``notify`` and the
    content is not high urgency. Mute and digest pass through untouched, so
    quiet hours can never upgrade a mute into a digest.

    ``tier_signals`` carry over unchanged: the downgrade changes the action, not
    how many independent signals agreed. Because ``CONFIDENCE_BASE["digest"]``
    is below ``CONFIDENCE_BASE["notify"]``, a downgraded row correctly reports
    lower confidence than the notify it replaced.
    """
    if ablate_layers_1_2:
        return _config_b(features, model_read)

    result = _ladder(features, model_read)

    if (
        features.dnd_active
        and result.action == "notify"
        and model_read.urgency != "high"
    ):
        return Decision(
            "digest",
            RuleKey.DIGEST_QUIET_HOURS,
            result.tier_signals,
            downgraded_from=(result.action, result.rule_key),
        )

    return result
