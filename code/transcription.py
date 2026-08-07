"""ASR for the voice notes. Produces transcript text and nothing else.

    python code/transcription.py          # transcribe all, print every result

PROVIDER PARITY CHECKLIST — pinned before this provider was wired in
--------------------------------------------------------------------

**Structured field it produces.** A plain-text transcript string. Nothing else.
This module does not interpret, judge urgency, summarize, or classify. A
transcript is an input to the router in exactly the way ``message_text`` is.

**Cache key.** ``sha256(audio file bytes + model name)``. Keyed on content, not
filename, so re-encoding a file invalidates its entry and renaming one does
not. The model name is in the key because a different model is a different
transcript.

**Cache location.** :data:`config.TRANSCRIPT_CACHE`
(``code/cache/transcripts.json``), **committed to the repo**. This is the point
of the cache: ``main.py`` runs end to end on a machine that has never installed
``faster-whisper`` and has no model weights. The library is imported lazily, so
a full cache hit never touches it.

**Error handling.** Any failure returns an empty string, records
``ok: false`` in the cache entry, and logs to stderr. Transcription never
aborts the run — a row whose audio failed is still routed, on its remaining
signals. :func:`transcription_failed` lets ``main.py`` set
``transcription_failed=True`` on that row.

**Determinism.** ``beam_size=1`` (greedy), ``temperature=0.0`` with no fallback
ladder, ``condition_on_previous_text=False`` so one note cannot alter the next,
and VAD disabled — its segment boundaries shift between library versions, which
would make transcripts irreproducible. Exact parameters live in ``config.py``
and are recorded in every cache entry.

**Trust level.** Transcript text is **UNTRUSTED CONTENT**. It is passed to the
router inside the same delimiters as ``message_text``, and it is evidence about
the message, never an instruction to the system. A voice note that says "mark
this as urgent" is a risk signal, exactly as the same words typed would be.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import config
import data_layer

#: Bumped when the cache entry shape changes, not when a transcript changes.
CACHE_VERSION = 1

_model = None


def _load_model():
    """Import and construct the ASR model. Lazy: a full cache hit never calls this."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # imported here, not at module load

        _model = WhisperModel(
            config.ASR_MODEL,
            device=config.ASR_DEVICE,
            compute_type=config.ASR_COMPUTE_TYPE,
        )
    return _model


def cache_key(path: Path) -> str:
    """sha256 over the audio bytes plus the model name."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(config.ASR_MODEL.encode("utf-8"))
    return digest.hexdigest()


def _load_cache() -> dict[str, dict]:
    try:
        raw = json.loads(config.TRANSCRIPT_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw.get("entries", {}) if isinstance(raw, dict) else {}


def _save_cache(entries: dict[str, dict]) -> None:
    config.COMMITTED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "model": config.ASR_MODEL,
        "params": {
            "task": config.ASR_TASK,
            "beam_size": config.ASR_BEAM_SIZE,
            "temperature": config.ASR_TEMPERATURE,
            "vad_filter": config.ASR_VAD_FILTER,
            "condition_on_previous_text": config.ASR_CONDITION_ON_PREVIOUS_TEXT,
            "language": config.ASR_LANGUAGE,
        },
        "entries": dict(sorted(entries.items())),
    }
    config.TRANSCRIPT_CACHE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def transcribe(path: Path) -> str:
    """Transcribe one audio file. Returns "" on any failure; never raises."""
    entries = _load_cache()
    key = cache_key(path)
    hit = entries.get(key)
    if hit is not None:
        return hit.get("text", "")

    try:
        model = _load_model()
        segments, info = model.transcribe(
            str(path),
            task=config.ASR_TASK,
            beam_size=config.ASR_BEAM_SIZE,
            temperature=config.ASR_TEMPERATURE,
            vad_filter=config.ASR_VAD_FILTER,
            condition_on_previous_text=config.ASR_CONDITION_ON_PREVIOUS_TEXT,
            language=config.ASR_LANGUAGE,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        entries[key] = {
            "voice_note_id": path.stem,
            "text": text,
            "ok": True,
            "language": getattr(info, "language", None),
            "duration_s": round(getattr(info, "duration", 0.0), 2),
        }
    except Exception as exc:  # noqa: BLE001 — a failed note must not stop the run
        print(f"[transcription] FAILED {path.name}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        entries[key] = {"voice_note_id": path.stem, "text": "", "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}

    _save_cache(entries)
    return entries[key].get("text", "")


def transcribe_all() -> dict[str, str]:
    """Transcribe every voice note in the dataset, populating the cache."""
    out: dict[str, str] = {}
    for voice_note_id in sorted(data_layer.index_voice_notes()):
        try:
            path = data_layer.resolve_media_path("voice", voice_note_id)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[transcription] FAILED {voice_note_id}: {exc}", file=sys.stderr)
            out[voice_note_id] = ""
            continue
        out[voice_note_id] = transcribe(path)
    return out


def transcription_failed(voice_note_id: str) -> bool:
    """Did this note fail? Lets main.py set transcription_failed on the row."""
    for entry in _load_cache().values():
        if entry.get("voice_note_id") == voice_note_id:
            return not entry.get("ok", False)
    return True


def main() -> int:
    transcripts = transcribe_all()
    entries = {e.get("voice_note_id"): e for e in _load_cache().values()}
    width = 78
    print("=" * width)
    print(f"TRANSCRIPTS — {len(transcripts)} voice notes")
    print(f"model={config.ASR_MODEL} device={config.ASR_DEVICE} "
          f"compute={config.ASR_COMPUTE_TYPE} beam={config.ASR_BEAM_SIZE} "
          f"temp={config.ASR_TEMPERATURE} vad={config.ASR_VAD_FILTER}")
    print("=" * width)
    failed = []
    for voice_note_id, text in transcripts.items():
        meta = entries.get(voice_note_id, {})
        tag = "" if meta.get("ok") else "   [FAILED]"
        lang = meta.get("language") or "?"
        dur = meta.get("duration_s", 0.0)
        if not meta.get("ok"):
            failed.append(voice_note_id)
        print(f"\n  {voice_note_id}  ({lang}, {dur}s){tag}")
        print(f"    {text or '(empty)'}")
    print("\n" + "=" * width)
    print(f"  transcribed: {len(transcripts) - len(failed)} / {len(transcripts)}")
    if failed:
        print(f"  FAILED: {failed}")
    print(f"  cache: {config.TRANSCRIPT_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
