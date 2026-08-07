"""Standalone dataset integrity check. Exits non-zero on any failure.

Run before anything else touches the data:

    python code/check_dataset.py

Every assertion here is about the *provided* files, not about our output. It
answers one question: can the pipeline rely on these joins resolving?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_layer  # noqa: E402

EXPECTED_MESSAGE_ROWS = 110
EXPECTED_SAMPLE_ROWS = 30
EXPECTED_HISTORY_ROWS = 412
EXPECTED_EVENT_ROWS = 412

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one assertion without aborting, so every failure is visible."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    if not ok:
        _failures.append(label)


def main() -> int:
    messages = data_layer.load_messages()
    samples = data_layer.load_sample_messages()
    history = data_layer.load_message_history()
    events = data_layer.index_message_events()
    users = data_layer.index_users()
    groups = data_layer.index_groups()
    businesses = data_layer.index_businesses()
    group_members = data_layer.index_group_members()
    user_business = data_layer.index_user_business()
    daily = data_layer.index_daily_summary()

    print("ROW COUNTS")
    check(
        f"messages.csv == {EXPECTED_MESSAGE_ROWS}",
        len(messages) == EXPECTED_MESSAGE_ROWS,
        f"got {len(messages)}",
    )
    check(
        f"sample_messages.csv == {EXPECTED_SAMPLE_ROWS}",
        len(samples) == EXPECTED_SAMPLE_ROWS,
        f"got {len(samples)}",
    )
    check(
        f"message_history.csv == {EXPECTED_HISTORY_ROWS}",
        len(history) == EXPECTED_HISTORY_ROWS,
        f"got {len(history)}",
    )
    check(
        f"message_events.csv == {EXPECTED_EVENT_ROWS}",
        len(events) == EXPECTED_EVENT_ROWS,
        f"got {len(events)} unique (user_id, message_id) keys",
    )

    print("\nIDENTITY")
    ids = [row["message_id"] for row in messages]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    check("no duplicate message_id in messages.csv", not dupes, f"dupes: {dupes}")

    sample_ids = [row["message_id"] for row in samples]
    sdupes = sorted({i for i in sample_ids if sample_ids.count(i) > 1})
    check("no duplicate message_id in sample_messages.csv", not sdupes, f"dupes: {sdupes}")

    overlap = set(ids) & set(sample_ids)
    check(
        "sample_messages ids are a separate namespace",
        not overlap,
        f"{len(overlap)} shared ids" if overlap else "zero overlap, as expected",
    )

    hist_ids = [row["message_id"] for row in history]
    hdupes = sorted({i for i in hist_ids if hist_ids.count(i) > 1})
    check("no duplicate message_id in message_history.csv", not hdupes, f"dupes: {hdupes}")

    print("\nREFERENTIAL INTEGRITY (messages.csv)")
    for label, field, index in (
        ("user_id", "user_id", users),
        ("group_id", "group_id", groups),
        ("business_id", "business_id", businesses),
        ("sender_user_id", "sender_user_id", users),
    ):
        missing = sorted(
            {r[field] for r in messages if r[field] and r[field] not in index}
        )
        check(
            f"every {label} resolves",
            not missing,
            f"unresolved: {missing}" if missing else f"{len(index)} in index",
        )

    gm_missing = sorted(
        {
            (r["group_id"], r["user_id"])
            for r in messages
            if r["conversation_type"] == "group"
            and r["group_id"]
            and (r["group_id"], r["user_id"]) not in group_members
        }
    )
    check(
        "every (group_id, recipient) has a group_members row",
        not gm_missing,
        f"unresolved: {gm_missing}" if gm_missing else "",
    )

    du_missing = sorted({r["user_id"] for r in messages if r["user_id"] not in daily})
    check(
        "every recipient has daily_notification_summary rows",
        not du_missing,
        f"unresolved: {du_missing}" if du_missing else f"{len(daily)} users covered",
    )

    print("\nMEDIA RESOLUTION")
    for label, rows in (
        ("messages.csv", messages),
        ("sample_messages.csv", samples),
        ("message_history.csv", history),
    ):
        refs = [(r["media_type"], r["media_id"]) for r in rows if r["media_id"]]
        broken: list[str] = []
        for media_type, media_id in refs:
            try:
                data_layer.resolve_media_path(media_type, media_id)
            except (FileNotFoundError, ValueError) as exc:
                broken.append(f"{media_id}: {exc}")
        check(
            f"{label}: all {len(refs)} media refs resolve to a file",
            not broken,
            "; ".join(broken) if broken else "",
        )

    print("\nCOVERAGE (reported, not asserted)")
    biz_rows = [r for r in messages if r["conversation_type"] == "business"]
    no_rel = [
        r["message_id"]
        for r in biz_rows
        if (r["user_id"], r["business_id"]) not in user_business
    ]
    print(f"  business rows in messages.csv          : {len(biz_rows)}")
    print(f"  ...with NO user_business_history entry : {len(no_rel)}")
    print(f"     {no_rel}")

    sender_rows = [r for r in messages if r["sender_user_id"]]
    print(f"  rows with a sender_user_id             : {len(sender_rows)}")
    print(f"  rows with media                        : "
          f"{sum(1 for r in messages if r['media_id'])}")

    print("\nLOADER STABILITY")
    print("  memoized loaders hand out shared objects; this proves reading")
    print("  them again does not mutate them")
    before = data_layer.frames_checksum()
    for loader in data_layer._LOADERS.values():
        loader()
    after = data_layer.frames_checksum()
    for name in sorted(before):
        same = before[name] == after[name]
        check(
            f"{name:<16} {before[name]}",
            same,
            f"changed to {after[name]}" if not same else "",
        )
    check("checksum dicts identical", before == after)

    print()
    if _failures:
        print(f"FAILED — {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
