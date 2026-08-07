"""Print the full feature block for the labeled rows and the edge cases.

    python code/inspect_features.py            # everything
    python code/inspect_features.py muted      # muted-group rows only
    python code/inspect_features.py injection  # msg_095 / msg_107 / msg_110
    python code/inspect_features.py samples    # the 30 labeled rows

Reporting only. Reads nothing it does not print, decides nothing.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_layer  # noqa: E402
import features as feat  # noqa: E402

INJECTION_IDS = ("msg_095", "msg_107", "msg_110")
WIDTH = 78

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "RECIPIENT",
        (
            "dnd_window",
            "dnd_active",
            "user_open_rate_30d",
            "user_dismiss_rate_30d",
            "user_engagement_denom_30d",
            "user_report_count_30d",
            "trailing_daily_load",
            "trailing_daily_days",
        ),
    ),
    (
        "GROUP",
        (
            "is_group",
            "group_id",
            "group_type",
            "member_count",
            "group_messages_30d",
            "group_muted_by_user",
            "user_role",
            "sender_role",
            "user_group_read_rate",
            "user_group_reply_rate",
            "group_rate_denom",
            "is_direct_mention",
        ),
    ),
    (
        "SENDER",
        (
            "has_sender",
            "sender_user_id",
            "n_prior",
            "open_rate",
            "reply_rate",
            "report_rate",
            "mean_forwarded_count",
            "sender_history_strength",
            "global_sender_n",
            "global_sender_open_rate",
            "global_sender_reply_rate",
            "global_sender_report_rate",
            "global_sender_mean_forwarded_count",
        ),
    ),
    (
        "BUSINESS",
        (
            "is_business",
            "business_id",
            "brand_name",
            "verified",
            "official_domain",
            "domain_used_by_sender",
            "domain_match",
            "official_domain_missing",
            "account_age_days",
            "domain_used_by_sender_age_days",
            "report_rate_per_1k",
            "business_messages_sent_30d",
            "has_relationship",
            "why_user_knows_account",
            "allows_promotions",
            "opted_out",
            "promotions_opted_out_at",
            "activity_count_180d",
            "biz_open_rate",
            "biz_dismiss_rate",
            "biz_engagement_denom_30d",
            "last_activity_recency_days",
        ),
    ),
    (
        "MESSAGE",
        (
            "forwarded_count",
            "forwarded_band",
            "media_type",
            "media_id",
            "has_link",
            "has_url_shortener",
        ),
    ),
)


def wrap(text: str, indent: int) -> str:
    pad = " " * indent
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > WIDTH - indent:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return f"\n{pad}".join(out) if out else "(empty)"


def print_block(row: dict[str, str], f: feat.MessageFeatures) -> None:
    print("\n" + "=" * WIDTH)
    header = f"{f.message_id}  ·  {f.conversation_type}  ·  recipient {f.user_id}  ·  {f.created_at}"
    print(header)
    if "action" in row:
        print(
            f"GOLD: action={row['action']}  message_type={row['message_type']}  "
            f"confidence={row['confidence']}  evidence={row['evidence_message_ids']}"
        )
    print("=" * WIDTH)
    print(f"  text: {wrap(row.get('message_text', '') or '(none)', 8)}")

    relevant = {"RECIPIENT", "MESSAGE"}
    if f.is_group:
        relevant.add("GROUP")
    if f.has_sender:
        relevant.add("SENDER")
    if f.is_business:
        relevant.add("BUSINESS")

    for name, keys in GROUPS:
        if name not in relevant:
            continue
        print(f"\n  [{name}]")
        for key in keys:
            print(f"    {key:<36} {getattr(f, key)!r}")

    print("\n  [ADVISORY — hints for the model, never read by gates.py]")
    print(f"    {'credential_language_flag':<36} {f.advisory.credential_language_flag!r}")
    print(f"    {'injection_pattern_flag':<36} {f.advisory.injection_pattern_flag!r}")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    indices = feat.Indices.load()
    messages = data_layer.load_messages()
    samples = data_layer.load_sample_messages()

    built = [(r, feat.build_features(r, indices)) for r in messages]
    muted = [(r, f) for r, f in built if f.group_muted_by_user]
    injection = [(r, f) for r, f in built if r["message_id"] in INJECTION_IDS]

    if which in ("all", "samples"):
        print("\n" + "#" * WIDTH)
        print(f"# THE {len(samples)} LABELED ROWS")
        print("#" * WIDTH)
        for row in samples:
            print_block(row, feat.build_features(row, indices))

    if which in ("all", "injection"):
        print("\n" + "#" * WIDTH)
        print(f"# INJECTION ROWS ({len(injection)} of {len(INJECTION_IDS)} requested)")
        print("#" * WIDTH)
        for row, f in injection:
            print_block(row, f)

    if which in ("all", "muted"):
        print("\n" + "#" * WIDTH)
        print(f"# MUTED-GROUP ROWS ({len(muted)})")
        print("#" * WIDTH)
        for row, f in muted:
            print_block(row, f)

    print("\n" + "#" * WIDTH)
    print("# SUMMARY")
    print("#" * WIDTH)
    print(f"  rows built                     : {len(built)}")
    print(f"  muted-group rows               : {len(muted)}")
    print(f"  ...of those, direct mention    : {sum(1 for _, f in muted if f.is_direct_mention)}")
    print(f"  dnd_active rows                : {sum(1 for _, f in built if f.dnd_active)}")
    print(f"  direct-mention rows (all)      : {sum(1 for _, f in built if f.is_direct_mention)}")
    print(f"  advisory injection_pattern_flag: {sum(1 for _, f in built if f.advisory.injection_pattern_flag)}")
    print(f"  advisory credential_lang_flag  : {sum(1 for _, f in built if f.advisory.credential_language_flag)}")
    print(f"  has_link / has_url_shortener   : "
          f"{sum(1 for _, f in built if f.has_link)} / {sum(1 for _, f in built if f.has_url_shortener)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
