"""Contract tests for schema.py.

Runs standalone (``python code/test_schema.py``) or under pytest. No third-party
dependency, because ``requirements.txt`` intentionally has no test runner.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema  # noqa: E402

MAX_REASON_CHARS = 160

# The 24 reason strings that appear in dataset/sample_messages.csv, copied
# verbatim. This is the anchor test: it fails if anyone rewords a gold string,
# which would silently break consistency scoring against the labeled rows.
GOLD_REASONS = frozenset(
    {
        # notify
        "A trusted group admin sent a time-sensitive update that should interrupt the user.",
        "A school admin sent a same-day operational update that the user is likely to need immediately.",
        "The message is from a work context and contains a direct deadline or meeting dependency.",
        "A verified business is sending an update that matches the user's recent order history.",
        "A verified business is sending a reminder that matches the user's recent booking history.",
        "The sender directly asks this user for a response or action.",
        "A close contact sent a short urgent request that should interrupt the user.",
        # digest
        "The message is promotional but matches a topic or business the user has opted into.",
        "The message is useful group information, but it is not urgent enough to interrupt the user.",
        "The message is a harmless greeting that can be read later.",
        "The message is safe casual chat with no urgent action required.",
        "A verified business is sending a legitimate but non-urgent update.",
        "The verified business message is legitimate but does not require immediate attention.",
        "The offer is potentially relevant, but it does not need immediate attention.",
        "The message matches the user's known interests but is still low priority.",
        "The sender is trusted, but the message has no urgent action or safety relevance.",
        "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
        # mute
        "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
        "The user has opted out of or repeatedly dismissed similar marketing messages.",
        "Similar historical messages were ignored, dismissed, or muted by this user.",
        "The message asks for urgent OTP or account verification through a suspicious flow.",
        "The message uses fake support language and account-blocking pressure to push the user into action.",
        "This is the first message from the sender and it asks for sensitive verification or payment.",
        "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
    }
)


def test_output_columns_exact() -> None:
    assert schema.OUTPUT_COLUMNS == (
        "message_id",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    )


def test_every_rule_key_has_a_reason() -> None:
    missing = [k.value for k in schema.RuleKey if k not in schema.REASONS]
    assert not missing, f"RuleKey members with no REASONS entry: {missing}"


def test_no_stray_reason_keys() -> None:
    stray = [k for k in schema.REASONS if not isinstance(k, schema.RuleKey)]
    assert not stray, f"REASONS keys that are not RuleKey members: {stray}"


def test_reasons_are_unique() -> None:
    seen: dict[str, schema.RuleKey] = {}
    for key, text in schema.REASONS.items():
        assert text not in seen, (
            f"{key.value} duplicates the reason string of {seen[text].value}; "
            "two branches sharing a string makes them indistinguishable in the output"
        )
        seen[text] = key


def test_reason_style() -> None:
    """Generic, one sentence, present tense, no numbers, no names."""
    for key, text in schema.REASONS.items():
        label = key.value
        assert text == text.strip(), f"{label}: leading/trailing whitespace"
        assert len(text) < MAX_REASON_CHARS, (
            f"{label}: {len(text)} chars, must be under {MAX_REASON_CHARS}"
        )
        assert not re.search(r"\d", text), f"{label}: contains a digit"
        assert "@" not in text, f"{label}: contains '@'"
        assert text.endswith("."), f"{label}: must end with a period"
        assert text.count(".") == 1, f"{label}: must be a single sentence"
        assert "!" not in text and "?" not in text, f"{label}: not a statement"
        assert "\n" not in text, f"{label}: contains a newline"
        assert '"' not in text, f"{label}: quotes suggest embedded message content"


def test_gold_reasons_present_verbatim() -> None:
    catalogue = set(schema.REASONS.values())
    missing = GOLD_REASONS - catalogue
    assert not missing, (
        "gold reason strings absent from REASONS (they must match "
        f"sample_messages.csv byte for byte): {sorted(missing)}"
    )
    assert len(GOLD_REASONS) == 24


def test_rule_action_matches_actions() -> None:
    for key, action in schema.RULE_ACTION.items():
        assert action in schema.ACTIONS, (
            f"{key.value} implies action {action!r}, which is not in ACTIONS"
        )
    assert set(schema.RULE_ACTION) == set(schema.RuleKey)


def test_confidence_bases_within_clamp() -> None:
    assert set(schema.CONFIDENCE_BASE) == set(schema.ACTIONS)
    for action, base in schema.CONFIDENCE_BASE.items():
        assert schema.CONFIDENCE_MIN <= base <= schema.CONFIDENCE_MAX, (
            f"base for {action} is {base}, outside "
            f"[{schema.CONFIDENCE_MIN}, {schema.CONFIDENCE_MAX}]"
        )


def test_confidence_top_of_band_within_clamp() -> None:
    """The highest reachable value must not exceed the clamp before clamping."""
    for action, base in schema.CONFIDENCE_BASE.items():
        top = round(base + schema.CONFIDENCE_STEP * schema.CONFIDENCE_MAX_TIER, 2)
        assert top <= schema.CONFIDENCE_MAX, (
            f"{action} reaches {top} at max tier, above {schema.CONFIDENCE_MAX}"
        )


def test_tool_enums_match_tuples() -> None:
    props = schema.ROUTE_MESSAGE_TOOL["input_schema"]["properties"]
    assert props["message_type"]["enum"] == list(schema.MESSAGE_TYPES)
    assert props["content_risk"]["enum"] == list(schema.CONTENT_RISK)
    assert props["urgency"]["enum"] == list(schema.URGENCY)
    assert props["proposed_action"]["enum"] == list(schema.ACTIONS)


def test_tool_shape() -> None:
    tool = schema.ROUTE_MESSAGE_TOOL
    assert tool["name"] == schema.ROUTE_MESSAGE_TOOL_NAME
    isch = tool["input_schema"]
    assert isch["type"] == "object"
    assert isch["additionalProperties"] is False
    props, required = isch["properties"], isch["required"]
    assert set(required) == set(props), "every property must be required"
    assert len(required) == len(set(required)), "duplicate entry in required"
    assert props["evidence_message_ids"]["maxItems"] == schema.MAX_EVIDENCE_IDS


def test_tool_cannot_return_decision_fields() -> None:
    """The model must never be able to author action, reason, or confidence."""
    props = set(schema.ROUTE_MESSAGE_TOOL["input_schema"]["properties"])
    forbidden = {"action", "reason", "confidence"} & props
    assert not forbidden, f"tool schema exposes decision field(s): {forbidden}"


def test_model_read_matches_tool_schema() -> None:
    """ModelRead's fields must mirror the tool's required list exactly."""
    required = set(schema.ROUTE_MESSAGE_TOOL["input_schema"]["required"])
    fields = set(schema.ModelRead.__dataclass_fields__)
    assert fields == required, (
        f"ModelRead and the tool schema have drifted apart; "
        f"only in ModelRead: {sorted(fields - required)}; "
        f"only in schema: {sorted(required - fields)}"
    )


def test_model_read_cannot_carry_a_decision() -> None:
    fields = set(schema.ModelRead.__dataclass_fields__)
    assert not ({"action", "reason", "confidence"} & fields)


def test_model_read_from_tool_input() -> None:
    payload = {
        "message_type": "personal",
        "content_risk": "none",
        "urgency": "low",
        "promotional": False,
        "is_router_injection_attempt": False,
        "asks_user_for_action": True,
        "evidence_message_ids": ["message_0001"],
        "proposed_action": "notify",
        "media_summary": "",
    }
    read = schema.ModelRead.from_tool_input(payload)
    assert read.message_type == "personal"
    assert read.evidence_message_ids == ("message_0001",)
    assert read.asks_user_for_action is True


def test_evidence_constants() -> None:
    assert schema.MAX_EVIDENCE_IDS == 2
    assert schema.EVIDENCE_SEPARATOR == ";"
    assert schema.EVIDENCE_NONE == "none"
    assert schema.EVIDENCE_SEPARATOR not in schema.EVIDENCE_NONE


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
