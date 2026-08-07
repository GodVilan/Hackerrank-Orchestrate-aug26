"""Degraded-row contract: a failure produces a worse row, never a missing one.

    python code/test_degraded.py

Both known failure modes are injected for real — the router and the image
encoder are monkeypatched to raise — and the assertion is on the row the
pipeline emits, not on the exception path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_layer  # noqa: E402
import features as feat  # noqa: E402
import finalize as fin  # noqa: E402
import gates  # noqa: E402
import main as pipeline  # noqa: E402
import router  # noqa: E402
import schema  # noqa: E402
import vision  # noqa: E402
from schema import RuleKey  # noqa: E402


def _row(message_id: str) -> dict:
    return next(r for r in data_layer.load_messages() if r["message_id"] == message_id)


def _assert_valid(out: dict, label: str) -> None:
    """Every degraded row must still satisfy the submission contract."""
    for column in schema.OUTPUT_COLUMNS:
        assert column in out, f"{label}: missing column {column}"
        assert out[column] not in (None, ""), f"{label}: {column} is null/empty"
    assert out["action"] in schema.ACTIONS, f"{label}: {out['action']}"
    assert out["message_type"] in schema.MESSAGE_TYPES, f"{label}: {out['message_type']}"
    assert out["reason"] in set(schema.REASONS.values()), f"{label}: reason not in catalogue"
    assert 0.0 <= float(out["confidence"]) <= 1.0, f"{label}: {out['confidence']}"
    assert out["evidence_message_ids"] == schema.EVIDENCE_NONE, (
        f"{label}: degraded evidence must be the {schema.EVIDENCE_NONE!r} sentinel, "
        f"got {out['evidence_message_ids']!r}"
    )
    assert out["_tier"] == 0, f"{label}: tier must be pinned to 0, got {out['_tier']}"
    assert schema.RULE_ACTION[RuleKey(out["_rule_key"])] == out["action"], (
        f"{label}: rule_key {out['_rule_key']} disagrees with action {out['action']}"
    )


def _run_with(monkey_target, attr, boom, message_id):
    indices = feat.Indices.load()
    original = getattr(monkey_target, attr)
    before = len(pipeline.DEGRADED)
    setattr(monkey_target, attr, boom)
    try:
        return pipeline.process(_row(message_id), indices), before
    finally:
        setattr(monkey_target, attr, original)


def test_tool_output_missing_emits_a_valid_row() -> None:
    def boom(*a, **k):
        raise router.ToolOutputMissing("injected: no tool_use block")

    out, before = _run_with(router, "route", boom, "msg_048")
    _assert_valid(out, "ToolOutputMissing")
    assert out["_degraded"] is True
    assert len(pipeline.DEGRADED) == before + 1
    assert pipeline.DEGRADED[-1]["exception_class"] == "ToolOutputMissing"


def test_unsupported_image_format_emits_a_valid_row() -> None:
    """Both the image path and the text-only retry fail — no recovery possible."""

    def boom_image(*a, **k):
        raise vision.UnsupportedImageFormat("injected: unconvertible image")

    def boom_route(*a, **k):
        raise router.ToolOutputMissing("injected: retry also failed")

    indices = feat.Indices.load()
    orig_img, orig_route = vision.image_block, router.route
    before = len(pipeline.DEGRADED)
    vision.image_block, router.route = boom_image, boom_route
    try:
        out = pipeline.process(_row("msg_027"), indices)  # an image row
    finally:
        vision.image_block, router.route = orig_img, orig_route
    _assert_valid(out, "UnsupportedImageFormat")
    assert out["_degraded"] is True
    assert len(pipeline.DEGRADED) == before + 1


def test_image_failure_recovers_text_only_rather_than_degrading() -> None:
    """The image is unusable but the text is fine — the row must NOT degrade."""

    def boom_image(*a, **k):
        raise vision.UnsupportedImageFormat("injected: unconvertible image")

    captured = {}

    def fake_route(row, features, pool, media=None):
        captured["media_was"] = media
        return fin.NEUTRAL_READ

    indices = feat.Indices.load()
    orig_img, orig_route = vision.image_block, router.route
    before = len(pipeline.DEGRADED)
    vision.image_block, router.route = boom_image, fake_route
    try:
        out = pipeline.process(_row("msg_027"), indices)
    finally:
        vision.image_block, router.route = orig_img, orig_route

    assert out["_degraded"] is False, "recoverable image failure must not degrade"
    assert out["_recovery"] == "recovered_text_only", out["_recovery"]
    assert captured["media_was"] is None, "retry must drop the image block"
    assert len(pipeline.DEGRADED) == before, "a recovered row is not a degraded row"


def test_degrade_keeps_a_deterministic_verdict() -> None:
    """A branch decided on joined features alone survives degradation intact."""
    kept = gates.Decision("mute", RuleKey.MUTE_IMPERSONATION_DOMAIN, ["impersonation"])
    out = fin.degrade(kept)
    assert out.action == "mute", out
    assert out.rule_key is RuleKey.MUTE_IMPERSONATION_DOMAIN, out
    assert out.tier_signals == [], "tier signals must be cleared"


def test_degrade_discards_a_content_verdict() -> None:
    """A content-substance branch is not defensible without a reading."""
    for key in (
        RuleKey.DIGEST_GROUP_INFO,
        RuleKey.NOTIFY_CLOSE_CONTACT_URGENT,
        RuleKey.MUTE_SCAM_OTP,
        RuleKey.DIGEST_CASUAL,
    ):
        out = fin.degrade(gates.Decision(schema.RULE_ACTION[key], key, ["x"]))
        assert out.rule_key is RuleKey.DIGEST_INSUFFICIENT_SIGNAL, (key, out)
        assert out.action == "digest", (key, out)


def test_degraded_confidence_is_band_floor() -> None:
    assert fin.confidence_for("digest", 0) == schema.CONFIDENCE_MIN
    assert fin.confidence_for("mute", 0) == schema.CONFIDENCE_BASE["mute"]


def test_disagreement_pins_tier_to_zero() -> None:
    features = feat.build_features(_row("msg_048"), feat.Indices.load())
    decision = gates.Decision("mute", RuleKey.MUTE_HIGH_REPORT_SENDER, ["a", "b"])
    read = schema.ModelRead(
        message_type="personal",
        content_risk="none",
        urgency="none",
        promotional=False,
        is_router_injection_attempt=False,
        asks_user_for_action=False,
        proposed_action="notify",  # disagrees with the ladder's mute
    )
    tier, signals = fin.confidence_tier(decision, features, read, ["message_0151"], 7)
    assert tier == 0, (tier, signals)
    assert signals == ["disagreement_override"], signals


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
