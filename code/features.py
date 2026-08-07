"""Every deterministic join and derivation. No model calls, no decisions.

One entry point:

    build_features(message_row, indices) -> MessageFeatures

Everything here is a join over the provided CSVs or arithmetic on joined values.
Nothing in this module is ever posed to the model as a question, and nothing
here reads a model output. ``gates.py`` consumes the result; it does not extend
it.

Three conventions the whole module follows:

* **Every rate is paired with its denominator.** A rate of 0.0 over n=0 and a
  rate of 0.0 over n=29 are different facts, and downstream code must be able to
  tell them apart. Division by zero always yields 0.0 with the denominator
  exposed alongside.
* **Absent context is a value, not an error.** A personal message has no group
  fields; a business with no prior relationship has ``has_relationship=False``.
  Those defaults are documented per field rather than left as ``None`` traps.
* **Advisory flags are quarantined.** ``credential_language_flag`` and
  ``injection_pattern_flag`` live on :class:`AdvisoryFlags`, reachable only via
  ``features.advisory``. They are regex hints passed to the model as context and
  **must never be read by decision logic** — a naive injection regex
  false-positives on legitimate messages about delivery-code instructions. The
  nesting is deliberate: any reference to ``.advisory`` inside ``gates.py`` is a
  contract violation visible in review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import data_layer

# --------------------------------------------------------------------------
# Index bundle
# --------------------------------------------------------------------------


@dataclass
class Indices:
    """The loaded dataset, plus history aggregates computed once.

    Built via :meth:`load`. Passed to every ``build_features`` call so the
    function stays injectable and testable without touching the filesystem.
    """

    users: dict[str, dict[str, str]]
    groups: dict[str, dict[str, str]]
    group_members: dict[tuple[str, str], dict[str, str]]
    businesses: dict[str, dict[str, str]]
    user_business: dict[tuple[str, str], dict[str, str]]
    message_history: list[dict[str, str]]
    message_events: dict[tuple[str, str], dict[str, str]]
    daily_summary: dict[str, list[dict[str, str]]]

    @classmethod
    def load(cls) -> Indices:
        return cls(
            users=data_layer.index_users(),
            groups=data_layer.index_groups(),
            group_members=data_layer.index_group_members(),
            businesses=data_layer.index_businesses(),
            user_business=data_layer.index_user_business(),
            message_history=data_layer.load_message_history(),
            message_events=data_layer.index_message_events(),
            daily_summary=data_layer.index_daily_summary(),
        )

    # -- history aggregates, built once on first access --------------------

    def _aggregate(self, key_fn) -> dict:
        """Fold message_history joined to message_events into per-key counters.

        The join is on ``(message_id, user_id)``: one history row is one message
        delivered to one recipient, and the event row records how that recipient
        reacted. A history row with no matching event counts toward ``n`` but
        contributes no reaction.
        """
        out: dict[object, dict[str, float]] = {}
        for row in self.message_history:
            key = key_fn(row)
            if key is None:
                continue
            stats = out.setdefault(
                key,
                {
                    "n": 0,
                    "opened": 0,
                    "replied": 0,
                    "dismissed": 0,
                    "reported": 0,
                    "fwd_sum": 0.0,
                    "n_events": 0,
                },
            )
            stats["n"] += 1
            stats["fwd_sum"] += _as_int(row.get("forwarded_count"))
            event = self.message_events.get((row["user_id"], row["message_id"]))
            if event is not None:
                stats["n_events"] += 1
                stats["opened"] += event.get("message_opened") == "1"
                stats["replied"] += event.get("message_replied") == "1"
                stats["dismissed"] += event.get("notification_dismissed") == "1"
                stats["reported"] += event.get("message_reported") == "1"
        return out

    @cached_property
    def sender_pair_stats(self) -> dict[tuple[str, str], dict[str, float]]:
        """Keyed ``(recipient_user_id, sender_user_id)``."""
        return self._aggregate(
            lambda r: (r["user_id"], r["sender_user_id"]) if r["sender_user_id"] else None
        )

    @cached_property
    def sender_global_stats(self) -> dict[str, dict[str, float]]:
        """Keyed ``sender_user_id``, across every recipient."""
        return self._aggregate(lambda r: r["sender_user_id"] or None)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _rate(numerator: float, denominator: float) -> float:
    """Guarded division. Zero denominator yields 0.0, never an exception."""
    if not denominator:
        return 0.0
    return round(numerator / denominator, 4)


def _parse_dt(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def in_dnd_window(created_at: str, window: str) -> bool:
    """Is the message timestamp inside ``HH:MM-HH:MM``?

    Handles the midnight wrap (``22:00-07:00`` covers 23:30 and 02:00 but not
    12:00). The interval is half-open: the start minute is inside, the end
    minute is outside. A malformed or empty window is False.
    """
    if not window or "-" not in window:
        return False
    start, _, end = window.partition("-")
    start, end = start.strip(), end.strip()
    if not (_HHMM.fullmatch(start) and _HHMM.fullmatch(end)):
        return False
    stamp = created_at[11:16]
    if not _HHMM.fullmatch(stamp):
        return False
    if start == end:
        return False
    if start < end:
        return start <= stamp < end
    return stamp >= start or stamp < end


_HHMM = re.compile(r"\d{2}:\d{2}")


def forwarded_band(count: int) -> str:
    """Band the raw forward count into {"0", "1-2", "3-5", "6+"}."""
    if count <= 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def history_strength(n_prior: int) -> str:
    """Bucket prior-interaction depth so nothing treats n=2 like n=29."""
    if n_prior <= 0:
        return "none"
    if n_prior <= 2:
        return "weak"
    if n_prior <= 9:
        return "moderate"
    return "strong"


# --------------------------------------------------------------------------
# Advisory regexes — hints only, never decisions
# --------------------------------------------------------------------------

# Scheme-qualified URLs plus bare domains. The bare-domain arm matters: every
# link in this dataset except one is written without a scheme
# ("pay-check-secure.com", "bit.ly/verify-quick"), so a scheme-only regex
# reports has_link=False on every impersonation row. Verified across all 552
# text rows in messages, sample_messages, and message_history: 10 distinct
# matches, no false positives.
_LINK_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9][a-z0-9-]+(?:\.[a-z0-9-]+)*"
    r"\.(?:com|in|net|org|co|io|me|pro|link|xyz|ru|app|dev|info|biz|online|site)"
    r"\b(?:/\S*)?",
    re.IGNORECASE,
)

_SHORTENER_RE = re.compile(
    r"\b(?:"
    r"bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|cutt\.ly|"
    r"rb\.gy|rebrand\.ly|shorturl\.at|tiny\.cc|s\.id|linktr\.ee|"
    r"wa\.link|wame\.pro|short\.link|clck\.ru|trib\.al"
    r")\b",
    re.IGNORECASE,
)

_CREDENTIAL_RE = re.compile(
    r"\b(?:otp|one[- ]time (?:password|code|pin)|cvv|"
    r"(?:login|verification|security|access|auth(?:entication)?) code|"
    r"wallet pin|account pin|upi pin|mpin|"
    r"seed phrase|recovery phrase|"
    r"(?:share|send|confirm|enter|provide|verify) (?:your |the )?"
    r"(?:pin|password|otp|code)"
    r")\b",
    re.IGNORECASE,
)

# Every arm is a generic shape, not a phrase lifted from a row in this dataset.
# The last two match the *form* of machine-readable metadata appearing in prose
# — a "field = assertion" pair, or a priority-looking key — rather than any
# particular token, because an arm written from a row we have read is fitted to
# the file we are trying to predict. Advisory only: widening or narrowing this
# regex changes no deterministic behaviour, since gates.py never reads it.
_INJECTION_RE = re.compile(
    r"(?:"
    r"ignore (?:all )?(?:previous|prior|above) \w+|"
    r"system (?:note|prompt|instruction|message) (?:for|to)|"
    r"routing override|"
    r"assistant instruction|"
    r"(?:always )?(?:mark|classify|set|treat) this (?:\w+ )?(?:as|to)\b|"
    r"notification router|"
    r"you must (?:mark|classify|notify)|"
    r"internal .{0,20}metadata|"
    r"user_priority|"
    r"\b\w+\s*=\s*(?:true|notify|high)\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AdvisoryFlags:
    """Regex pre-flags passed to the model as context.

    **Never read by decision logic.** These are lexical hints with known false
    positives; the model weighs them against the message as a whole, and
    ``gates.py`` ignores them entirely.
    """

    credential_language_flag: bool = False
    injection_pattern_flag: bool = False


# --------------------------------------------------------------------------
# Feature record
# --------------------------------------------------------------------------


@dataclass
class MessageFeatures:
    """Deterministic features for one incoming message."""

    # -- identity -----------------------------------------------------------
    message_id: str
    user_id: str
    conversation_type: str
    created_at: str

    # -- recipient ----------------------------------------------------------
    dnd_window: str = ""
    dnd_active: bool = False
    user_open_rate_30d: float = 0.0
    user_dismiss_rate_30d: float = 0.0
    user_engagement_denom_30d: int = 0
    user_report_count_30d: int = 0
    trailing_daily_load: float = 0.0
    trailing_daily_days: int = 0

    # -- group (conversation_type == "group") -------------------------------
    is_group: bool = False
    group_id: str = ""
    group_type: str = ""
    member_count: int = 0
    group_messages_30d: int = 0
    group_muted_by_user: bool = False
    user_role: str = ""
    sender_role: str = ""
    user_group_read_rate: float = 0.0
    user_group_reply_rate: float = 0.0
    group_rate_denom: int = 0
    is_direct_mention: bool = False

    # -- sender (sender_user_id present) ------------------------------------
    has_sender: bool = False
    sender_user_id: str = ""
    n_prior: int = 0
    open_rate: float = 0.0
    reply_rate: float = 0.0
    dismiss_rate: float = 0.0
    report_rate: float = 0.0
    mean_forwarded_count: float = 0.0
    sender_history_strength: str = "none"
    global_sender_n: int = 0
    global_sender_open_rate: float = 0.0
    global_sender_reply_rate: float = 0.0
    global_sender_dismiss_rate: float = 0.0
    global_sender_report_rate: float = 0.0
    global_sender_mean_forwarded_count: float = 0.0

    # -- business (conversation_type == "business") -------------------------
    is_business: bool = False
    business_id: str = ""
    brand_name: str = ""
    verified: bool = False
    official_domain: str = ""
    domain_used_by_sender: str = ""
    domain_match: bool = False
    official_domain_missing: bool = False
    account_age_days: int = 0
    domain_used_by_sender_age_days: int = 0
    report_rate_per_1k: float = 0.0
    business_messages_sent_30d: int = 0
    has_relationship: bool = False
    why_user_knows_account: str = ""
    allows_promotions: bool = False
    opted_out: bool = False
    promotions_opted_out_at: str = ""
    activity_count_180d: int = 0
    biz_open_rate: float = 0.0
    biz_dismiss_rate: float = 0.0
    biz_engagement_denom_30d: int = 0
    last_activity_recency_days: int | None = None

    # -- message ------------------------------------------------------------
    forwarded_count: int = 0
    forwarded_band: str = "0"
    media_type: str = ""
    media_id: str = ""
    has_link: bool = False
    has_url_shortener: bool = False

    # -- advisory (never used by decision logic) ----------------------------
    advisory: AdvisoryFlags = field(default_factory=AdvisoryFlags)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def is_known_counterparty(features: MessageFeatures) -> bool:
    """Has this recipient dealt with this counterparty before?

    Resolved per ``conversation_type`` rather than by one global field, because
    ``has_relationship`` is derived from ``user_business_history`` and is
    therefore False by construction on every group and personal row:

    * business -> ``has_relationship`` (a user_business_history row exists)
    * personal -> ``n_prior > 0`` (this recipient, this sender)
    * group    -> ``n_prior > 0`` (this recipient, this sender)

    Read by Layer 1's first-contact branch. ``has_relationship`` itself is never
    read outside business rows. Callers that need to distinguish a thin True
    (``n_prior == 1``) from a thick one should read ``n_prior`` or
    ``sender_history_strength``, both of which remain exposed.
    """
    if features.is_business:
        return features.has_relationship
    return features.n_prior > 0


def build_features(
    message_row: dict[str, str], indices: Indices
) -> MessageFeatures:
    """Join one message row against the dataset. Pure; no I/O, no model calls."""
    user_id = message_row["user_id"]
    text = message_row.get("message_text", "") or ""
    created_at = message_row.get("created_at", "") or ""

    out = MessageFeatures(
        message_id=message_row["message_id"],
        user_id=user_id,
        conversation_type=message_row.get("conversation_type", ""),
        created_at=created_at,
    )

    _fill_recipient(out, message_row, indices)
    _fill_group(out, message_row, indices, text)
    _fill_sender(out, message_row, indices)
    _fill_business(out, message_row, indices)
    _fill_message(out, message_row, text)
    return out


# -- recipient --------------------------------------------------------------


def _fill_recipient(out: MessageFeatures, row: dict[str, str], idx: Indices) -> None:
    user = idx.users.get(out.user_id, {})
    out.dnd_window = user.get("do_not_disturb_window", "")
    out.dnd_active = in_dnd_window(out.created_at, out.dnd_window)

    opened = _as_int(user.get("messages_opened_30d"))
    dismissed = _as_int(user.get("notifications_dismissed_30d"))
    # No "received" column exists. Opened + dismissed is the set of
    # notifications this user demonstrably acted on, and both come from the
    # same 30-day window, so the two rates are complementary and comparable.
    denom = opened + dismissed
    out.user_engagement_denom_30d = denom
    out.user_open_rate_30d = _rate(opened, denom)
    out.user_dismiss_rate_30d = _rate(dismissed, denom)
    out.user_report_count_30d = _as_int(user.get("messages_reported_30d"))

    days = idx.daily_summary.get(out.user_id, [])
    out.trailing_daily_days = len(days)
    out.trailing_daily_load = _rate(
        sum(_as_int(d.get("notifications_sent")) for d in days), len(days)
    )


# -- group ------------------------------------------------------------------


def _fill_group(
    out: MessageFeatures, row: dict[str, str], idx: Indices, text: str
) -> None:
    if row.get("conversation_type") != "group":
        return
    group_id = row.get("group_id", "")
    if not group_id:
        return

    out.is_group = True
    out.group_id = group_id
    group = idx.groups.get(group_id, {})
    out.group_type = group.get("group_type", "")
    out.member_count = _as_int(group.get("member_count"))
    out.group_messages_30d = _as_int(group.get("messages_30d"))

    membership = idx.group_members.get((group_id, out.user_id), {})
    out.group_muted_by_user = membership.get("group_muted_by_user") == "1"
    out.user_role = membership.get("role", "")

    sender_id = row.get("sender_user_id", "")
    if sender_id:
        sender_membership = idx.group_members.get((group_id, sender_id), {})
        out.sender_role = sender_membership.get("role", "")

    # Group traffic is the denominator for both rates so they stay comparable.
    denom = out.group_messages_30d
    out.group_rate_denom = denom
    out.user_group_read_rate = _rate(_as_int(membership.get("messages_read_30d")), denom)
    out.user_group_reply_rate = _rate(_as_int(membership.get("replies_sent_30d")), denom)

    out.is_direct_mention = bool(
        re.search(rf"@{re.escape(out.user_id)}\b", text)
    )


# -- sender -----------------------------------------------------------------


def _fill_sender(out: MessageFeatures, row: dict[str, str], idx: Indices) -> None:
    sender_id = row.get("sender_user_id", "")
    if not sender_id:
        return

    out.has_sender = True
    out.sender_user_id = sender_id

    pair = idx.sender_pair_stats.get((out.user_id, sender_id))
    if pair:
        n = int(pair["n"])
        out.n_prior = n
        out.open_rate = _rate(pair["opened"], n)
        out.reply_rate = _rate(pair["replied"], n)
        out.dismiss_rate = _rate(pair["dismissed"], n)
        out.report_rate = _rate(pair["reported"], n)
        out.mean_forwarded_count = _rate(pair["fwd_sum"], n)
    out.sender_history_strength = history_strength(out.n_prior)

    glob = idx.sender_global_stats.get(sender_id)
    if glob:
        n = int(glob["n"])
        out.global_sender_n = n
        out.global_sender_open_rate = _rate(glob["opened"], n)
        out.global_sender_reply_rate = _rate(glob["replied"], n)
        out.global_sender_dismiss_rate = _rate(glob["dismissed"], n)
        out.global_sender_report_rate = _rate(glob["reported"], n)
        out.global_sender_mean_forwarded_count = _rate(glob["fwd_sum"], n)


# -- business ---------------------------------------------------------------


def _fill_business(out: MessageFeatures, row: dict[str, str], idx: Indices) -> None:
    if row.get("conversation_type") != "business":
        return
    business_id = row.get("business_id", "")
    if not business_id:
        return

    out.is_business = True
    out.business_id = business_id

    biz = idx.businesses.get(business_id, {})
    out.brand_name = biz.get("brand_name", "")
    out.verified = biz.get("verified") == "1"
    official = (biz.get("official_domain") or "").strip()
    used = (biz.get("domain_used_by_sender") or "").strip()
    out.official_domain = official
    out.domain_used_by_sender = used
    # An absent official_domain is a distinct state, not a mismatch: it means
    # there is nothing to compare against.
    out.official_domain_missing = not official
    out.domain_match = bool(official) and official.casefold() == used.casefold()

    out.account_age_days = _as_int(biz.get("account_age_days"))
    out.domain_used_by_sender_age_days = _as_int(biz.get("domain_used_by_sender_age_days"))
    sent = _as_int(biz.get("messages_sent_30d"))
    out.business_messages_sent_30d = sent
    out.report_rate_per_1k = _rate(_as_int(biz.get("user_reports_30d")) * 1000, sent)

    rel = idx.user_business.get((out.user_id, business_id))
    out.has_relationship = rel is not None
    if rel is None:
        return

    out.why_user_knows_account = rel.get("why_user_knows_account", "")
    out.allows_promotions = rel.get("allows_promotions") == "1"
    # There is no `opted_out` column. `allows_promotions == 0` is the default
    # never-opted-in state; an explicit opt-out is a populated timestamp.
    out.promotions_opted_out_at = rel.get("promotions_opted_out_at", "")
    out.opted_out = bool(out.promotions_opted_out_at)
    out.activity_count_180d = _as_int(rel.get("activity_count_180d"))

    opened = _as_int(rel.get("messages_opened_30d"))
    dismissed = _as_int(rel.get("messages_dismissed_30d"))
    denom = opened + dismissed
    out.biz_engagement_denom_30d = denom
    out.biz_open_rate = _rate(opened, denom)
    out.biz_dismiss_rate = _rate(dismissed, denom)

    last = _parse_dt(rel.get("last_activity_at", ""))
    now = _parse_dt(out.created_at)
    if last and now:
        out.last_activity_recency_days = (now - last).days


# -- message ----------------------------------------------------------------


def _fill_message(out: MessageFeatures, row: dict[str, str], text: str) -> None:
    out.forwarded_count = _as_int(row.get("forwarded_count"))
    out.forwarded_band = forwarded_band(out.forwarded_count)
    out.media_type = row.get("media_type", "")
    out.media_id = row.get("media_id", "")
    out.has_link = bool(_LINK_RE.search(text))
    out.has_url_shortener = bool(_SHORTENER_RE.search(text))
    out.advisory = AdvisoryFlags(
        credential_language_flag=bool(_CREDENTIAL_RE.search(text)),
        injection_pattern_flag=bool(_INJECTION_RE.search(text)),
    )
