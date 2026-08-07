"""Strict CSV emission. Exact column order, resumable, nothing else.

Rows are appended as they complete, so a crash at row 90 keeps rows 1-89. On a
restart, ``already_written()`` reports which ids are present and the caller
skips them. ``finalize_order()`` rewrites the file once at the end in
``messages.csv`` input order.

Only ``schema.OUTPUT_COLUMNS`` is written. Diagnostic keys that ``finalize``
attaches (``_tier``, ``_rule_key``) are dropped here — they belong in
``run_stats.json``, not in the submission.
"""

from __future__ import annotations

import csv
from pathlib import Path

import config
import data_layer
import schema


def already_written(path: Path | None = None) -> set[str]:
    """message_ids already present in the output file. Empty if absent."""
    path = path or config.OUTPUT_CSV
    if not path.exists():
        return set()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(schema.OUTPUT_COLUMNS):
                return set()  # a file with the wrong header is not resumable
            return {row["message_id"] for row in reader if row.get("message_id")}
    except (OSError, csv.Error):
        return set()


def _open_for_append(path: Path):
    """Open for append, writing the header only when the file is new."""
    is_new = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(schema.OUTPUT_COLUMNS),
        quoting=csv.QUOTE_ALL,
        extrasaction="ignore",
    )
    if is_new:
        writer.writeheader()
    return handle, writer


def append_row(row: dict, path: Path | None = None) -> None:
    """Append one row immediately and flush, so a crash cannot lose it."""
    path = path or config.OUTPUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, writer = _open_for_append(path)
    try:
        writer.writerow({column: row[column] for column in schema.OUTPUT_COLUMNS})
        handle.flush()
    finally:
        handle.close()


def append_rows(rows, path: Path | None = None) -> int:
    """Append many rows in one open. Returns the count written."""
    path = path or config.OUTPUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, writer = _open_for_append(path)
    count = 0
    try:
        for row in rows:
            writer.writerow({column: row[column] for column in schema.OUTPUT_COLUMNS})
            count += 1
        handle.flush()
    finally:
        handle.close()
    return count


def finalize_order(path: Path | None = None) -> int:
    """Rewrite the file in messages.csv input order. Returns the row count.

    Rows are appended in completion order, which under concurrency is not input
    order. The submission contract does not require a particular order, but a
    diff against the input should line up, and a reader comparing the two files
    side by side should not have to sort first.
    """
    path = path or config.OUTPUT_CSV
    if not path.exists():
        return 0

    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["message_id"]: row for row in csv.DictReader(handle)}

    order = [row["message_id"] for row in data_layer.load_messages()]
    ordered = [rows[mid] for mid in order if mid in rows]
    # Anything not in messages.csv is kept at the end rather than silently
    # dropped, so an unexpected id is visible to the validator.
    ordered += [rows[mid] for mid in rows if mid not in set(order)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(schema.OUTPUT_COLUMNS),
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ordered)
    return len(ordered)
