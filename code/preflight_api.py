"""Pre-run API contract check. Standalone; nothing imports this.

Run before every full pipeline run:

    python code/preflight_api.py

It answers four questions with evidence rather than memory:
  * is a key present and usable,
  * what models does this key actually see,
  * is the model named in config.py among them,
  * does one real forced-tool call come back in the exact shape schema.py
    declares.

Exit codes:
  0  every assertion passed
  2  no API key resolved
  3  config.PRIMARY_MODEL absent from models.list()
  4  the API call itself failed
  5  the response violated the route_message contract

The key is read through config.py and is never printed, echoed, or written.
The probe message is synthetic and is not drawn from any dataset file; this
script never reads dataset/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic  # noqa: E402

import config  # noqa: E402
import schema  # noqa: E402

# Synthetic, WhatsApp-shaped, and deliberately not from any dataset file.
PROBE_TEXT = (
    "Hi all, the lift in B wing is getting serviced tomorrow between 10 and 1. "
    "Please use the stairs in that window. Sorry for the trouble!"
)

EXIT_OK = 0
EXIT_NO_KEY = 2
EXIT_MODEL_MISSING = 3
EXIT_API_ERROR = 4
EXIT_CONTRACT = 5


def fail(code: int, message: str) -> int:
    print(f"\nFAIL: {message}")
    return code


def validate_tool_input(payload: dict[str, Any]) -> list[str]:
    """Validate a tool_use input against ROUTE_MESSAGE_TOOL. Returns problems."""
    isch = schema.ROUTE_MESSAGE_TOOL["input_schema"]
    props: dict[str, Any] = isch["properties"]
    required: list[str] = isch["required"]
    problems: list[str] = []

    missing = [name for name in required if name not in payload]
    if missing:
        problems.append(f"missing required propert(ies): {missing}")

    extra = [name for name in payload if name not in props]
    if extra:
        problems.append(f"extra propert(ies) not in schema: {extra}")

    for name, value in payload.items():
        spec = props.get(name)
        if spec is None:
            continue
        expected = spec["type"]
        if expected == "string" and not isinstance(value, str):
            problems.append(f"{name}: expected string, got {type(value).__name__}")
        elif expected == "boolean" and not isinstance(value, bool):
            problems.append(f"{name}: expected boolean, got {type(value).__name__}")
        elif expected == "array" and not isinstance(value, list):
            problems.append(f"{name}: expected array, got {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{name}: {value!r} is not one of {spec['enum']}")

    evidence = payload.get("evidence_message_ids")
    if isinstance(evidence, list):
        if len(evidence) > schema.MAX_EVIDENCE_IDS:
            problems.append(
                f"evidence_message_ids has {len(evidence)} entries, "
                f"max is {schema.MAX_EVIDENCE_IDS}"
            )
        bad = [item for item in evidence if not isinstance(item, str)]
        if bad:
            problems.append(f"evidence_message_ids contains non-strings: {bad}")

    return problems


def main() -> int:
    print("=" * 72)
    print("PREFLIGHT — API contract check")
    print("=" * 72)

    # ---- 1. key -----------------------------------------------------------
    print("\n[1] Credential")
    if not config.ANTHROPIC_API_KEY:
        print("  no API key resolved")
        return fail(
            EXIT_NO_KEY,
            "ANTHROPIC_API_KEY is not set. Export it, or copy "
            "code/.env.example to code/.env and fill it in.",
        )
    print("  ANTHROPIC_API_KEY resolved (value not shown)")
    print(f"  anthropic SDK: {anthropic.__version__}")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # ---- 2. model lineup --------------------------------------------------
    print("\n[2] Models visible to this key")
    try:
        models = list(client.models.list())
    except Exception as exc:  # noqa: BLE001 — report any transport/auth failure
        return fail(EXIT_API_ERROR, f"models.list() failed: {type(exc).__name__}: {exc}")

    print(f"  {'model id':<34} {'display name':<26} created")
    print(f"  {'-' * 34} {'-' * 26} {'-' * 10}")
    for model in models:
        created = getattr(model, "created_at", None)
        created_str = str(created)[:10] if created else "—"
        print(f"  {model.id:<34} {model.display_name:<26} {created_str}")
    print(f"  ({len(models)} models)")

    # ---- 3. config model present -----------------------------------------
    print("\n[3] config.PRIMARY_MODEL present in that list")
    available = {model.id for model in models}
    if config.PRIMARY_MODEL not in available:
        print(f"  config.PRIMARY_MODEL = {config.PRIMARY_MODEL!r}")
        print("  NOT among the ids above.")
        return fail(
            EXIT_MODEL_MISSING,
            f"{config.PRIMARY_MODEL!r} is not available to this key. "
            "Update config.PRIMARY_MODEL to one of the ids listed above.",
        )
    print(f"  PASS  {config.PRIMARY_MODEL}")

    # ---- 4. one forced-tool call -----------------------------------------
    print("\n[4] One forced-tool messages.create call")
    print(f"  model      : {config.PRIMARY_MODEL}")
    print(f"  max_tokens : 256")
    print(f"  tool       : {schema.ROUTE_MESSAGE_TOOL_NAME} (imported from schema.py)")
    print(f"  probe text : {PROBE_TEXT[:60]}...")
    try:
        response = client.messages.create(
            model=config.PRIMARY_MODEL,
            max_tokens=256,
            tools=[schema.ROUTE_MESSAGE_TOOL],
            tool_choice={"type": "tool", "name": schema.ROUTE_MESSAGE_TOOL_NAME},
            messages=[{"role": "user", "content": PROBE_TEXT}],
        )
    except Exception as exc:  # noqa: BLE001
        return fail(
            EXIT_API_ERROR, f"messages.create failed: {type(exc).__name__}: {exc}"
        )

    # ---- 5. report --------------------------------------------------------
    print("\n[5] Response")
    print(f"  stop_reason         : {response.stop_reason}")
    print(f"  usage.input_tokens  : {response.usage.input_tokens}")
    print(f"  usage.output_tokens : {response.usage.output_tokens}")
    print(f"  content block types : {[block.type for block in response.content]}")

    tool_blocks = [block for block in response.content if block.type == "tool_use"]
    if len(tool_blocks) != 1:
        return fail(
            EXIT_CONTRACT,
            f"expected exactly 1 tool_use block, got {len(tool_blocks)}",
        )

    payload = tool_blocks[0].input
    print(f"\n  tool_use.name       : {tool_blocks[0].name}")
    print("  tool_use.input      :")
    for line in json.dumps(payload, indent=2, sort_keys=True).splitlines():
        print(f"    {line}")

    # ---- 6. contract assertions ------------------------------------------
    print("\n[6] Contract assertions")
    checks: list[tuple[str, bool, str]] = []

    checks.append(("exactly one tool_use block", True, ""))
    checks.append(
        (
            "tool name matches schema",
            tool_blocks[0].name == schema.ROUTE_MESSAGE_TOOL_NAME,
            f"got {tool_blocks[0].name!r}",
        )
    )
    checks.append(("tool_use.input is an object", isinstance(payload, dict), ""))

    problems = validate_tool_input(payload) if isinstance(payload, dict) else ["not a dict"]
    checks.append(
        ("input validates against route_message schema", not problems, "; ".join(problems))
    )

    evidence = payload.get("evidence_message_ids") if isinstance(payload, dict) else None
    checks.append(
        (
            f"evidence_message_ids length <= {schema.MAX_EVIDENCE_IDS}",
            isinstance(evidence, list) and len(evidence) <= schema.MAX_EVIDENCE_IDS,
            f"got {evidence!r}",
        )
    )
    checks.append(
        (
            "no decision field returned (action/reason/confidence)",
            isinstance(payload, dict)
            and not ({"action", "reason", "confidence"} & set(payload)),
            "",
        )
    )

    failed = 0
    for label, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail and not ok else ''}")
        failed += not ok

    if failed:
        return fail(EXIT_CONTRACT, f"{failed} contract assertion(s) failed")

    print("\n" + "=" * 72)
    print("PREFLIGHT OK — safe to run the pipeline.")
    print("=" * 72)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
