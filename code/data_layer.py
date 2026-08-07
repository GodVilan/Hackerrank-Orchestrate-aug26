"""Loading and indexing for the provided CSVs. No decisions, no derivations.

This module does exactly three things: read the dataset files, key them for
lookup, and resolve media IDs to paths on disk. It computes no rates, applies no
thresholds, joins nothing, and calls no model. Anything derived belongs in
``features.py``.

Two guarantees every loader upholds:

* **Every value is a ``str``.** Files are read with ``dtype=str`` and
  ``keep_default_na=False``, so an empty cell is ``""`` and never ``NaN`` or
  ``None``. Downstream code can use plain truthiness on any field without a
  pandas-specific null check.
* **Input order is preserved.** ``load_messages()`` returns rows in the order
  they appear in ``messages.csv``, which is the order ``output.csv`` must be
  written back in.

Results are memoized, so the 110-row main loop reads each file once.

``dataset/output.csv`` is the blank submission template, not an input. It is
deliberately not loaded here; ``writer.py`` owns output, and ``schema.py`` owns
the column contract.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

import pandas as pd

import config

# --------------------------------------------------------------------------
# Primitive reader
# --------------------------------------------------------------------------


def _read(path: Path) -> list[dict[str, str]]:
    """Read one CSV as a list of all-string dicts, order preserved.

    ``dtype=str`` plus ``keep_default_na=False`` is the combination that keeps
    empty cells as ``""``. Without the second argument pandas turns them into
    ``NaN`` floats even under ``dtype=str``, and every downstream truthiness
    check silently inverts.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"dataset file missing: {path}. Expected it under "
            f"{config.DATASET_DIR}; check that the dataset is present."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.to_dict(orient="records")


# --------------------------------------------------------------------------
# Row loaders
# --------------------------------------------------------------------------


@cache
def load_messages() -> list[dict[str, str]]:
    """The rows to route, in ``messages.csv`` input order."""
    return _read(config.MESSAGES_CSV)


@cache
def load_sample_messages() -> list[dict[str, str]]:
    """The labeled example rows.

    These carry the same input columns as ``messages.csv`` plus the five gold
    output columns. Note their ``message_id`` values live in a separate
    namespace (``sample_msg_*``) and do not appear in ``messages.csv``.
    """
    return _read(config.SAMPLE_MESSAGES_CSV)


@cache
def load_message_history() -> list[dict[str, str]]:
    """Past messages received by users, in file order."""
    return _read(config.MESSAGE_HISTORY_CSV)


# --------------------------------------------------------------------------
# Indexes
# --------------------------------------------------------------------------


@cache
def index_users() -> dict[str, dict[str, str]]:
    return {row["user_id"]: row for row in _read(config.USERS_CSV)}


@cache
def index_groups() -> dict[str, dict[str, str]]:
    return {row["group_id"]: row for row in _read(config.GROUPS_CSV)}


@cache
def index_group_members() -> dict[tuple[str, str], dict[str, str]]:
    """Keyed ``(group_id, user_id)``."""
    return {
        (row["group_id"], row["user_id"]): row
        for row in _read(config.GROUP_MEMBERS_CSV)
    }


@cache
def index_businesses() -> dict[str, dict[str, str]]:
    return {
        row["business_id"]: row for row in _read(config.BUSINESS_ACCOUNTS_CSV)
    }


@cache
def index_user_business() -> dict[tuple[str, str], dict[str, str]]:
    """Keyed ``(user_id, business_id)``.

    A missing key means this user has no recorded relationship with that
    business — a meaningful state, not an error.
    """
    return {
        (row["user_id"], row["business_id"]): row
        for row in _read(config.USER_BUSINESS_HISTORY_CSV)
    }


@cache
def index_message_events() -> dict[tuple[str, str], dict[str, str]]:
    """Keyed ``(user_id, message_id)`` — how one user reacted to one message."""
    return {
        (row["user_id"], row["message_id"]): row
        for row in _read(config.MESSAGE_EVENTS_CSV)
    }


@cache
def index_daily_summary() -> dict[str, list[dict[str, str]]]:
    """Keyed ``user_id`` -> that user's daily rows, in file order."""
    out: dict[str, list[dict[str, str]]] = {}
    for row in _read(config.DAILY_NOTIFICATION_SUMMARY_CSV):
        out.setdefault(row["user_id"], []).append(row)
    return out


@cache
def index_images() -> dict[str, str]:
    """``image_id`` -> ``file_path``, relative to the dataset directory."""
    return {row["image_id"]: row["file_path"] for row in _read(config.IMAGES_CSV)}


@cache
def index_voice_notes() -> dict[str, str]:
    """``voice_note_id`` -> ``file_path``, relative to the dataset directory."""
    return {
        row["voice_note_id"]: row["file_path"]
        for row in _read(config.VOICE_NOTES_CSV)
    }


# --------------------------------------------------------------------------
# Media resolution
# --------------------------------------------------------------------------

#: ``media_type`` value in messages.csv -> the index that maps its IDs.
_MEDIA_INDEX = {"image": index_images, "voice": index_voice_notes}


def resolve_media_path(media_type: str, media_id: str) -> Path:
    """Resolve a ``(media_type, media_id)`` pair to a file on disk.

    The ``file_path`` column in ``images.csv`` and ``voice_notes.csv`` is
    relative to :data:`config.DATASET_DIR`, not to the repo root, so the join
    goes through :func:`config.resolve_media_path` — the single place that
    knows that rule.

    Raises:
        ValueError: ``media_type`` is not ``"image"`` or ``"voice"``.
        FileNotFoundError: the ID is absent from its index, or the file it
            names does not exist on disk. The message always names the
            ``media_id``.
    """
    if media_type not in _MEDIA_INDEX:
        raise ValueError(
            f"unknown media_type {media_type!r} for media_id {media_id!r}; "
            f"expected one of {sorted(_MEDIA_INDEX)}"
        )

    index = _MEDIA_INDEX[media_type]()
    file_path = index.get(media_id)
    if file_path is None:
        raise FileNotFoundError(
            f"media_id {media_id!r} is not listed in the {media_type} index "
            f"({len(index)} entries)"
        )

    path = config.resolve_media_path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"media_id {media_id!r} maps to {file_path!r}, which does not "
            f"exist at {path}"
        )
    return path


#: Every memoized loader, by the name used in checksums and diagnostics.
_LOADERS = {
    "messages": load_messages,
    "sample_messages": load_sample_messages,
    "message_history": load_message_history,
    "users": index_users,
    "groups": index_groups,
    "group_members": index_group_members,
    "businesses": index_businesses,
    "user_business": index_user_business,
    "message_events": index_message_events,
    "daily_summary": index_daily_summary,
    "images": index_images,
    "voice_notes": index_voice_notes,
}


def clear_caches() -> None:
    """Drop every memoized load. Only needed by tests that swap datasets."""
    for fn in _LOADERS.values():
        fn.cache_clear()


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------


def _canonical(obj: object) -> object:
    """Reduce a loader's return value to an order-stable JSON-safe structure.

    Mapping keys may be tuples, so they are flattened to a single string.
    Mappings are emitted as sorted key/value pairs, which makes the result
    independent of insertion order.
    """
    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            flat = "".join(key) if isinstance(key, tuple) else str(key)
            items.append([flat, _canonical(value)])
        items.sort(key=lambda pair: pair[0])
        return items
    if isinstance(obj, list):
        return [_canonical(item) for item in obj]
    return obj


def frames_checksum() -> dict[str, str]:
    """A stable content hash per memoized loader.

    Deterministic within a process: the same underlying rows always produce the
    same digest, regardless of how many times a loader has been called. Used to
    prove that reading the data never mutates it — the memoized loaders hand out
    shared objects, so a caller that mutates a row in place would silently
    corrupt every later read.
    """
    digests: dict[str, str] = {}
    for name, loader in _LOADERS.items():
        payload = json.dumps(
            _canonical(loader()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digests[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return digests
