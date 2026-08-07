"""Standalone pre-submission validator. Exits non-zero on any failure.

    python code/validate_output.py

Checks the submission contract only. It never repairs anything — a file that
fails here is a file that must be regenerated, not patched.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import data_layer  # noqa: E402
import schema  # noqa: E402

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    if not ok:
        _failures.append(label)


def main() -> int:
    path = config.OUTPUT_CSV
    print("=" * 74)
    print(f"VALIDATE {path}")
    print("=" * 74)

    if not path.exists():
        print(f"\nFAIL: {path} does not exist")
        return 1

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)

    messages = data_layer.load_messages()
    expected_ids = [row["message_id"] for row in messages]

    print("\nSTRUCTURE")
    check(
        "header equals schema.OUTPUT_COLUMNS in order",
        header == list(schema.OUTPUT_COLUMNS),
        f"got {header}",
    )
    check(
        f"row count == messages.csv ({len(expected_ids)})",
        len(rows) == len(expected_ids),
        f"got {len(rows)}",
    )

    print("\nIDENTITY")
    ids = [row["message_id"] for row in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    check("no duplicate message_id", not dupes, f"dupes: {dupes}")
    missing = sorted(set(expected_ids) - set(ids))
    extra = sorted(set(ids) - set(expected_ids))
    check(
        "message_id set matches messages.csv exactly",
        not missing and not extra,
        f"missing: {missing[:5]} extra: {extra[:5]}",
    )

    print("\nFIELD VALUES")
    bad_action = sorted({r["action"] for r in rows if r["action"] not in schema.ACTIONS})
    check("every action in ACTIONS", not bad_action, f"bad: {bad_action}")

    bad_type = sorted(
        {r["message_type"] for r in rows if r["message_type"] not in schema.MESSAGE_TYPES}
    )
    check("every message_type in MESSAGE_TYPES", not bad_type, f"bad: {bad_type}")

    bad_conf = []
    for row in rows:
        try:
            value = float(row["confidence"])
        except (TypeError, ValueError):
            bad_conf.append(f"{row['message_id']}={row['confidence']!r}")
            continue
        if not 0.0 <= value <= 1.0:
            bad_conf.append(f"{row['message_id']}={value}")
    check("confidence parses as float in [0,1]", not bad_conf, f"bad: {bad_conf[:5]}")

    history_ids = {row["message_id"] for row in data_layer.load_message_history()}
    bad_evidence = []
    for row in rows:
        raw = row["evidence_message_ids"]
        if raw == schema.EVIDENCE_NONE:
            continue
        if not raw:
            bad_evidence.append(f"{row['message_id']}=<empty string>")
            continue
        parts = raw.split(schema.EVIDENCE_SEPARATOR)
        unknown = [p for p in parts if p not in history_ids]
        if unknown:
            bad_evidence.append(f"{row['message_id']}={unknown}")
        elif len(parts) > schema.MAX_EVIDENCE_IDS:
            bad_evidence.append(f"{row['message_id']}={len(parts)} ids")
    check(
        "evidence is 'none' or ids that exist in message_history.csv",
        not bad_evidence,
        f"bad: {bad_evidence[:5]}",
    )

    catalogue = set(schema.REASONS.values())
    bad_reason = sorted(
        {r["message_id"] for r in rows if not r["reason"] or r["reason"] not in catalogue}
    )
    check(
        "reason is non-empty and appears in schema.REASONS",
        not bad_reason,
        f"bad: {bad_reason[:5]}",
    )

    print("\nDISTRIBUTIONS")
    actions = Counter(r["action"] for r in rows)
    types = Counter(r["message_type"] for r in rows)
    confs = Counter(r["confidence"] for r in rows)
    evid = Counter(
        "none" if r["evidence_message_ids"] == schema.EVIDENCE_NONE
        else str(len(r["evidence_message_ids"].split(schema.EVIDENCE_SEPARATOR)))
        for r in rows
    )
    total = len(rows) or 1

    print(f"\n  {'action':<18}{'n':>5}{'pct':>8}")
    print("  " + "-" * 31)
    for key in schema.ACTIONS:
        print(f"  {key:<18}{actions[key]:>5}{actions[key] / total * 100:>7.1f}%")

    print(f"\n  {'message_type':<18}{'n':>5}{'pct':>8}")
    print("  " + "-" * 31)
    for key in schema.MESSAGE_TYPES:
        if types[key]:
            print(f"  {key:<18}{types[key]:>5}{types[key] / total * 100:>7.1f}%")
    unused = [k for k in schema.MESSAGE_TYPES if not types[k]]
    if unused:
        print(f"  (never predicted: {', '.join(unused)})")

    print(f"\n  {'confidence':<18}{'n':>5}{'pct':>8}")
    print("  " + "-" * 31)
    for key in sorted(confs):
        print(f"  {key:<18}{confs[key]:>5}{confs[key] / total * 100:>7.1f}%")

    print(f"\n  {'evidence ids':<18}{'n':>5}{'pct':>8}")
    print("  " + "-" * 31)
    for key in sorted(evid):
        print(f"  {key:<18}{evid[key]:>5}{evid[key] / total * 100:>7.1f}%")

    print()
    if _failures:
        print(f"FAILED — {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
