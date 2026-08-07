"""Run the ladder over all 110 rows without a model call. stdout only.

    python code/dryrun_gates.py

This is the check that catches branch-ordering mistakes before any tokens are
spent — the failure mode Design Tradeoff 1 names as the ladder's main cost. It
answers three questions:

1. Does every row produce a valid ``(action, rule_key)`` pair, with
   ``RULE_ACTION[rule_key] == action``?
2. Under a neutral reading, how much of the corpus lands on the terminal
   default? A large share means Layer 4's mapping is under-specified, not that
   ``DIGEST_GROUP_INFO`` is genuinely popular.
3. Which rule keys are unreachable — never produced by ANY model reading, on
   any row? An unreachable branch is dead code or a mis-ordered guard.

The model readings here are **stubs**, not predictions. Section 2's
distribution describes the stub as much as the ladder; section 3 is the part
that says something about the ladder alone.
"""

from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_layer  # noqa: E402
import features as feat  # noqa: E402
import gates  # noqa: E402
import schema  # noqa: E402

WIDTH = 74


def neutral_read() -> schema.ModelRead:
    """The least opinionated reading the schema permits."""
    return schema.ModelRead(
        message_type="unknown",
        content_risk="none",
        urgency="none",
        promotional=False,
        is_router_injection_attempt=False,
        asks_user_for_action=False,
        proposed_action="digest",
    )


def main() -> int:
    idx = feat.Indices.load()
    rows = data_layer.load_messages()
    built = [(r, feat.build_features(r, idx)) for r in rows]
    total = len(built)
    failures: list[str] = []

    # ---- 1. validity -----------------------------------------------------
    print("=" * WIDTH)
    print("1. VALIDITY — every row yields a coherent decision")
    print("=" * WIDTH)
    stub = neutral_read()
    decisions = []
    for row, f in built:
        d = gates.decide(f, stub)
        decisions.append((row, f, d))
        if d.action not in schema.ACTIONS:
            failures.append(f"{row['message_id']}: bad action {d.action!r}")
        if not isinstance(d.rule_key, schema.RuleKey):
            failures.append(f"{row['message_id']}: bad rule_key {d.rule_key!r}")
        elif schema.RULE_ACTION[d.rule_key] != d.action:
            failures.append(
                f"{row['message_id']}: {d.rule_key.value} implies "
                f"{schema.RULE_ACTION[d.rule_key]}, got {d.action}"
            )
        if d.rule_key not in schema.REASONS:
            failures.append(f"{row['message_id']}: {d.rule_key} has no reason string")
    print(f"  rows evaluated                : {total}")
    print(f"  action in ACTIONS             : {'PASS' if not failures else 'FAIL'}")
    print(f"  RULE_ACTION[key] == action    : {'PASS' if not failures else 'FAIL'}")
    print(f"  every key has a reason string : {'PASS' if not failures else 'FAIL'}")
    for problem in failures[:10]:
        print(f"    {problem}")

    # ---- 2. distribution under a neutral reading -------------------------
    print("\n" + "=" * WIDTH)
    print("2. DISTRIBUTION under a NEUTRAL stub reading")
    print("   (describes the stub as much as the ladder — see section 3)")
    print("=" * WIDTH)
    counts = Counter(d.rule_key for _, _, d in decisions)
    actions = Counter(d.action for _, _, d in decisions)
    print(f"\n  {'rule_key':<32}{'n':>5}{'pct':>8}")
    print("  " + "-" * (WIDTH - 4))
    for key, n in counts.most_common():
        print(f"  {key.value:<32}{n:>5}{n / total * 100:>7.1f}%")
    print("  " + "-" * (WIDTH - 4))
    print(f"  {'ACTIONS':<32}", end="")
    print("  ".join(f"{a}={n}" for a, n in actions.most_common()))

    default_share = counts[schema.RuleKey.DIGEST_GROUP_INFO] / total * 100
    print(f"\n  terminal default DIGEST_GROUP_INFO: {default_share:.1f}% of rows")
    if default_share > 40:
        print("    NOTE: under a neutral stub the default absorbing a large share is")
        print("    expected — 'unknown' message_type matches no Layer 4 digest arm.")
        print("    Section 3 is the test of whether the mapping is reachable.")

    downgraded = [d for _, _, d in decisions if d.downgraded_from]
    print(f"  quiet-hours downgrades fired      : {len(downgraded)}")

    # ---- 3. reachability across the reading space ------------------------
    print("\n" + "=" * WIDTH)
    print("3. REACHABILITY — rule keys produced by ANY reading, on any row")
    print("=" * WIDTH)
    grid = [
        schema.ModelRead(
            message_type=mt,
            content_risk=cr,
            urgency=ur,
            promotional=promo,
            is_router_injection_attempt=inj,
            asks_user_for_action=asks,
            proposed_action="digest",
        )
        for mt, cr, ur, promo, inj, asks in product(
            schema.MESSAGE_TYPES,
            schema.CONTENT_RISK,
            schema.URGENCY,
            (True, False),
            (True, False),
            (True, False),
        )
    ]
    reached: Counter[schema.RuleKey] = Counter()
    for _, f in built:
        for read in grid:
            reached[gates.decide(f, read).rule_key] += 1
    print(f"\n  {len(grid)} readings x {total} rows = {len(grid) * total:,} decisions")

    never = [k for k in schema.RuleKey if k not in reached]
    print(f"\n  rule keys reached : {len(reached)} of {len(list(schema.RuleKey))}")
    print(f"  NEVER reached     : {len(never)}")
    for key in never:
        print(f"    {key.value}")
    if not never:
        print("    (none — every branch is reachable)")

    print("\n" + "=" * WIDTH)
    if failures:
        print(f"FAILED — {len(failures)} validity problem(s)")
        return 1
    print("Dry run clean: all decisions coherent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
