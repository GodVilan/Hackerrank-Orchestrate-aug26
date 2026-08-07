"""Print candidate pool sizes and three fully rendered pools. stdout only.

    python code/inspect_pools.py

Reporting only. Shows what the model will actually be handed as its evidence
menu, in the exact rendering ``prompts.py`` will use.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_layer  # noqa: E402
import evidence  # noqa: E402
import features as feat  # noqa: E402

WIDTH = 78


def main() -> int:
    idx = feat.Indices.load()
    rows = data_layer.load_messages()
    pools = [(r, evidence.build_candidate_pool(r, idx)) for r in rows]
    sizes = [len(p) for _, p in pools]

    print("=" * WIDTH)
    print(f"CANDIDATE POOL SIZE — {len(rows)} rows")
    print("=" * WIDTH)
    print(f"\n  mean={statistics.mean(sizes):.2f}  median={statistics.median(sizes)}  "
          f"min={min(sizes)}  max={max(sizes)}  total candidates={sum(sizes)}")
    print("\n  size  rows")
    for size, count in sorted(Counter(sizes).items()):
        print(f"  {size:>4}  {count:>4}  {'#' * count}")

    print("\n  by conversation_type:")
    for ctype in ("business", "group", "personal"):
        vals = [len(p) for r, p in pools if r["conversation_type"] == ctype]
        print(f"    {ctype:<9} rows={len(vals):>3}  mean={statistics.mean(vals):5.2f}  "
              f"min={min(vals)}  max={max(vals)}  empty={vals.count(0)}")

    empty = [(r["message_id"], r["conversation_type"]) for r, p in pools if not p]
    print(f"\n  EMPTY POOLS: {len(empty)}")
    for mid, ctype in empty:
        print(f"    {mid}  [{ctype}]")
    print("\n  An empty pool is answered with evidence = none, enforced in code.")
    print("  The pool is never widened to the same-user pool: the rubric asks")
    print("  whether evidence points to relevant history, and for these rows")
    print("  there is none.")

    print("\n" + "=" * WIDTH)
    print("FULLY RENDERED POOLS — one per conversation_type")
    print("=" * WIDTH)
    for ctype in ("business", "group", "personal"):
        pick = next(
            (
                (r, p)
                for r, p in pools
                if r["conversation_type"] == ctype and 3 <= len(p) <= 8
            ),
            next((r, p) for r, p in pools if r["conversation_type"] == ctype),
        )
        row, pool = pick
        print(f"\n  {'-' * (WIDTH - 4)}")
        print(f"  {ctype.upper()}  {row['message_id']}  recipient={row['user_id']}  "
              f"pool={len(pool)}")
        counterparty = (
            row["business_id"] or row["group_id"] or row["sender_user_id"] or "-"
        )
        print(f"  counterparty: {counterparty}")
        text = (row["message_text"] or "").replace("\n", " ")[:150]
        print(f"  incoming: {text or '(media-only row)'}")
        print(f"\n  candidate pool as the model sees it:")
        print(f"    message_id | created_at | text[:120] | media_type | "
              f"forwarded_count | reaction")
        for candidate in pool:
            print(f"    {candidate.render()}")

    print("\n" + "=" * WIDTH)
    reactions = Counter(c.reaction for _, p in pools for c in p)
    print("REACTION LABEL DISTRIBUTION across every candidate")
    print("=" * WIDTH)
    total = sum(reactions.values())
    for label, count in reactions.most_common():
        print(f"  {label:<12}{count:>6}  {count / total * 100:5.1f}%")
    print(f"  {'total':<12}{total:>6}")
    print("\n  Precedence when several event flags are set on one row:")
    print("  reported > dismissed > replied > opened > no_record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
