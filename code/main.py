"""End-to-end pipeline. Load, feature, transcribe, pool, route, decide, write.

    python code/main.py             # all 110 rows
    python code/main.py --limit 5   # smoke test
    python code/main.py --fresh     # ignore an existing output.csv

Resumable: rows already present in ``output.csv`` are skipped, so a crash at row
90 costs nine rows rather than ninety.

Per-row failure handling
------------------------
Two exceptions are caught per row, and only these two — an unanticipated error
still surfaces and stops the run:

* ``ToolOutputMissing`` — the model did not answer in the required shape after
  its retries.
* ``UnsupportedImageFormat`` — the image could not be prepared. This is raised
  during media preparation, *before* the model call, so the row is first retried
  **text-only**. Only if that also fails does it degrade.

A degraded row is still a complete, schema-valid row. The ladder runs on
``finalize.NEUTRAL_READ``; a branch that fired on joined features alone keeps
its real verdict, and anything that would have been a content judgement becomes
``DIGEST_INSUFFICIENT_SIGNAL``. Confidence is pinned to the band floor and
evidence is the ``none`` sentinel. Never a missing row, never a null field.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import data_layer  # noqa: E402
import evidence  # noqa: E402
import features as feat  # noqa: E402
import finalize as fin  # noqa: E402
import gates  # noqa: E402
import router  # noqa: E402
import transcription  # noqa: E402
import vision  # noqa: E402
import writer  # noqa: E402
from router import ToolOutputMissing  # noqa: E402
from vision import UnsupportedImageFormat  # noqa: E402

LOG_PATH = Path.home() / "hackerrank_orchestrate_august26" / "log.txt"

#: Populated during the run, never reconstructed afterwards.
DEGRADED: list[dict] = []


def log_degraded(message_id: str, exc: BaseException, stage: str) -> None:
    """Append one degraded row to the shared transcript log."""
    line = (
        f"[{datetime.now().astimezone().isoformat()}] DEGRADED ROW "
        f"message_id={message_id} exception={type(exc).__name__} stage={stage} "
        f"detail={str(exc)[:300]}\n"
    )
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass  # a log we cannot write must not stop the run
    print(f"  DEGRADED {message_id}: {type(exc).__name__} at {stage}", file=sys.stderr)


def prepare_row(row: dict, indices):
    """Build everything the router needs. Returns ``(row, features, pool, media, note)``.

    An image that cannot be prepared does **not** stop the row. The failure
    happens before the model call, so the message text is still routable and
    usually carries the substantive content. The image is dropped, the failure
    is logged against the media stage, and the row proceeds text-only —
    ``note`` records that it was recovered rather than routed intact.

    Only if the subsequent routing call also fails does the row degrade.
    """
    row = dict(row)
    features = feat.build_features(row, indices)
    pool = evidence.build_candidate_pool(row, indices)
    media = None
    note = None

    if row.get("media_type") == "voice" and row.get("media_id"):
        try:
            row["_transcript"] = transcription.transcribe(
                data_layer.resolve_media_path("voice", row["media_id"])
            )
        except (FileNotFoundError, ValueError):
            row["_transcript"] = ""
    elif row.get("media_type") == "image" and row.get("media_id"):
        try:
            media = vision.image_block(row["media_id"])
        except UnsupportedImageFormat as exc:
            log_degraded(row["message_id"], exc, "media")
            media, note = None, "recovered_text_only"

    return row, features, pool, media, note


def process(row: dict, indices, *, ablate_layers_1_2: bool = False) -> dict:
    """One row, start to finished output dict. Never raises for the two known modes.

    ``ablate_layers_1_2`` is the Config B switch, threaded through to
    ``gates.decide``. It defaults to False, so ``main()`` and every other caller
    get Config A; only ``evaluation/main.py --config b`` sets it. The ablation is
    a flag on this one path rather than a forked pipeline, so the two configs
    cannot drift in loading, features, media, pooling, or evidence enforcement —
    the only difference between them is the ladder.
    """
    message_id = row["message_id"]
    stage = "features"
    try:
        row, features, pool, media, recovery = prepare_row(row, indices)
        stage = "route"
        model_read = router.route(row, features, pool, media)
        stage = "evidence"
        evidence_ids, violation = evidence.enforce_evidence(
            model_read.evidence_message_ids, pool
        )
        stage = "decide"
        decision = gates.decide(
            features, model_read, ablate_layers_1_2=ablate_layers_1_2
        )
        out = fin.finalize(
            decision, features, model_read, evidence_ids, pool_size=len(pool)
        )
        out["_degraded"] = False
        out["_evidence_violation"] = violation
        # In-pool ids the model selected and enforcement kept, that the emission
        # cap then discarded. Distinct from `_evidence_violation`, which counts
        # ids that were never admissible. Write-only: recorded for run_stats and
        # read by no decision.
        out["_evidence_truncated"] = max(
            0, len(evidence_ids) - fin.EMITTED_EVIDENCE_IDS
        )
        out["_recovery"] = recovery
        return out

    except (ToolOutputMissing, UnsupportedImageFormat) as exc:
        log_degraded(message_id, exc, stage)
        DEGRADED.append(
            {
                "message_id": message_id,
                "exception_class": type(exc).__name__,
                "stage": stage,
            }
        )
        features = feat.build_features(row, indices)
        decision = fin.degrade(gates.decide(features, fin.NEUTRAL_READ))
        out = fin.finalize(
            decision, features, fin.NEUTRAL_READ, (), pool_size=0, degraded=True
        )
        out["_degraded"] = True
        out["_evidence_violation"] = False
        out["_evidence_truncated"] = 0
        out["_recovery"] = None
        return out


def _counts(rows, key) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Route every message in messages.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    indices = feat.Indices.load()
    messages = data_layer.load_messages()
    if args.limit:
        messages = messages[: args.limit]

    if args.fresh and config.OUTPUT_CSV.exists():
        config.OUTPUT_CSV.unlink()

    done = writer.already_written()
    todo = [row for row in messages if row["message_id"] not in done]
    print(f"rows={len(messages)}  already written={len(done)}  to process={len(todo)}")

    finished: list[dict] = []
    for index, row in enumerate(todo, 1):
        out = process(row, indices)
        writer.append_row(out)
        finished.append(out)
        if index % 10 == 0 or index == len(todo):
            print(f"  {index}/{len(todo)}")

    total = writer.finalize_order()

    by_exception: dict[str, int] = {}
    for entry in DEGRADED:
        by_exception[entry["exception_class"]] = (
            by_exception.get(entry["exception_class"], 0) + 1
        )
    for name in ("ToolOutputMissing", "UnsupportedImageFormat"):
        by_exception.setdefault(name, 0)

    router.STATS.write(
        {
            "degraded_rows": DEGRADED,
            "degraded_count": len(DEGRADED),
            "degraded_by_exception": by_exception,
            "evidence_violations": sum(
                1 for r in finished if r.get("_evidence_violation")
            ),
            # Two different things, counted separately on purpose.
            # `evidence_violations` counts rows where the model returned an id
            # that was never admissible — an out-of-pool, duplicate, or
            # over-length selection. `evidence_truncated` counts valid, in-pool
            # ids that enforcement kept and the emission cap
            # (finalize.EMITTED_EVIDENCE_IDS) then dropped. A run can have zero
            # violations and still discard many ids; reporting only the former
            # would read as "nothing was discarded".
            "evidence_truncated": sum(
                r.get("_evidence_truncated", 0) for r in finished
            ),
            "evidence_truncated_rows": sum(
                1 for r in finished if r.get("_evidence_truncated", 0)
            ),
            "emitted_evidence_ids_cap": fin.EMITTED_EVIDENCE_IDS,
            "recovered_text_only": sum(
                1 for r in finished if r.get("_recovery") == "recovered_text_only"
            ),
            "rule_key_counts": _counts(finished, "_rule_key"),
            "tier_counts": _counts(finished, "_tier"),
            "run_wall_clock_s": round(time.monotonic() - started, 2),
        }
    )

    print(f"\nwrote {total} rows to {config.OUTPUT_CSV}")
    print(f"degraded: {len(DEGRADED)}  by exception: {by_exception}")
    print(f"stats:    {config.RUN_STATS}")
    print(f"elapsed:  {time.monotonic() - started:.1f}s")

    # The invariant: 110 rows with zero exceptions and 110 with any number.
    expected = len(data_layer.load_messages())
    assert total == expected, (
        f"row-count invariant broken: wrote {total}, messages.csv has {expected}. "
        f"A degraded row must still be a row."
    )
    print(f"row-count invariant: {total} == {expected}  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
