"""Print gold reason groupings and raw joined records. stdout only.

    python code/inspect_gold_reasons.py

Reporting only: this script reads the labeled rows and prints what is in them.
It draws no conclusions and writes no files.

Note on ``opted_out``: there is no such column. ``user_business_history`` has
``allows_promotions`` and ``promotions_opted_out_at``. Both raw values are
printed, plus a derived flag defined here as
``promotions_opted_out_at != ""`` and labeled as derived.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_layer  # noqa: E402

OFFER_RELEVANT = (
    "The offer is potentially relevant, but it does not need immediate attention."
)
PROMO_OPTED_IN = (
    "The message is promotional but matches a topic or business the user has "
    "opted into."
)

WIDTH = 78


def wrap(text: str, indent: int, width: int = WIDTH) -> str:
    """Wrap without importing textwrap's paragraph semantics."""
    pad = " " * indent
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width - indent:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return f"\n{pad}".join(out)


def section(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def dump_rows(rows: list[dict[str, str]], businesses, user_business) -> None:
    if not rows:
        print("\n  (no sample rows carry this gold reason)")
        return
    for row in rows:
        biz_id = row["business_id"]
        biz = businesses.get(biz_id, {}) if biz_id else {}
        rel_key = (row["user_id"], biz_id)
        rel = user_business.get(rel_key) if biz_id else None
        opted_out_at = rel.get("promotions_opted_out_at", "") if rel else ""

        print(f"\n  {'-' * (WIDTH - 2)}")
        print(f"  message_id             : {row['message_id']}")
        print(f"  user_id                : {row['user_id']}")
        print(f"  conversation_type      : {row['conversation_type']}")
        print(f"  business_id            : {biz_id or '—'}")
        print(f"  brand_name             : {biz.get('brand_name', '—')}")
        print(f"  verified               : {biz.get('verified', '—')}")
        print(f"  user_business_history  : {'present' if rel else 'ABSENT'}")
        print(f"  allows_promotions      : {rel.get('allows_promotions', '—') if rel else '—'}")
        print(f"  promotions_opted_out_at: {opted_out_at or '(empty)'}")
        print(f"  opted_out  [derived]   : {bool(opted_out_at)}")
        print(f"  why_user_knows_account : {rel.get('why_user_knows_account', '—') if rel else '—'}")
        print(f"  activity_count_180d    : {rel.get('activity_count_180d', '—') if rel else '—'}")
        print(f"  media_type / media_id  : {row['media_type'] or '—'} / {row['media_id'] or '—'}")
        print(f"  forwarded_count        : {row['forwarded_count']}")
        print(f"  GOLD action            : {row['action']}")
        print(f"  GOLD message_type      : {row['message_type']}")
        print(f"  GOLD confidence        : {row['confidence']}")
        print(f"  GOLD evidence          : {row['evidence_message_ids']}")
        text = row["message_text"] or "(empty — media-only row)"
        print(f"  message_text           : {wrap(text, 27)}")


def main() -> int:
    samples = data_layer.load_sample_messages()
    businesses = data_layer.index_businesses()
    user_business = data_layer.index_user_business()

    # ---- (a) ---------------------------------------------------------------
    section(f"(a) GOLD REASON FREQUENCY — {len(samples)} labeled rows")

    by_reason: dict[str, list[str]] = defaultdict(list)
    action_of: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        by_reason[row["reason"]].append(row["message_id"])
        action_of[row["reason"]].add(row["action"])

    counts = Counter({reason: len(ids) for reason, ids in by_reason.items()})
    print(f"\n  distinct gold reason strings: {len(by_reason)}")
    print(f"  repeated (count > 1)        : {sum(1 for c in counts.values() if c > 1)}")
    print(f"  used exactly once           : {sum(1 for c in counts.values() if c == 1)}")

    print(f"\n  {'n':>2}  {'action':<7}  {'message_ids':<32}  reason")
    print(f"  {'-' * 2}  {'-' * 7}  {'-' * 32}  {'-' * 28}")
    for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        ids = ", ".join(by_reason[reason])
        actions = "/".join(sorted(action_of[reason]))
        print(f"  {count:>2}  {actions:<7}  {ids:<32}  {reason}")

    # ---- (b) ---------------------------------------------------------------
    section("(b) FULL RECORDS — gold reason: offer potentially relevant")
    print(f"\n  {OFFER_RELEVANT}")
    dump_rows(
        [r for r in samples if r["reason"] == OFFER_RELEVANT],
        businesses,
        user_business,
    )

    # ---- (c) ---------------------------------------------------------------
    section("(c) FULL RECORDS — gold reason: promotional but opted into")
    print(f"\n  {PROMO_OPTED_IN}")
    dump_rows(
        [r for r in samples if r["reason"] == PROMO_OPTED_IN],
        businesses,
        user_business,
    )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
