"""Two-tier scorer. Runs the full pipeline on the labeled rows and compares.

    cd code && python -m evaluation.main                       # Config A
    cd code && python -m evaluation.main --config b --out evaluation/metrics_b.json

Scores against ``dataset/sample_messages.csv``. Those 30 rows are a **disjoint
id namespace** from ``messages.csv`` — they are extra labeled examples, not a
labeled subset of the rows we predict — so this is held-out scoring, and no
threshold in ``config.py`` was fitted to them.

Every number printed is also written to the ``--out`` path, which defaults to
``code/evaluation/metrics.json``. The report is written separately; this module
only measures.

``--config b`` selects the Decision 1 ablation (``gates.decide`` with Layers 1
and 2 removed and the model's ``proposed_action`` written straight through).
Both configs share one pipeline; the flag is threaded through ``main.process``.

``--skip-judge`` runs every section except the LLM judge in Tier 2b. The judge
is **not cached**, so re-running it costs 30 model calls and its verdict counts
move by 1-2 between invocations at temperature 0. Use the flag whenever the
judge section is not the thing being measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import anthropic  # noqa: E402

import config  # noqa: E402
import data_layer  # noqa: E402
import evidence as ev  # noqa: E402
import features as feat  # noqa: E402
import main as pipeline  # noqa: E402
import router  # noqa: E402
import schema  # noqa: E402

METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"
WIDTH = 78
SHARED_MEDIA = ("img_003", "img_008", "img_010")

CONFIG_DESCRIPTIONS = {
    "a": "full precedence ladder (shipping configuration)",
    "b": "ablation — Layers 1 and 2 removed, model proposed_action written through",
}


def section(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def run_pipeline_on_samples(ablate: bool = False) -> list[dict]:
    indices = feat.Indices.load()
    rows = data_layer.load_sample_messages()
    out = []
    print(f"routing {len(rows)} labeled rows through the full pipeline...")
    for index, gold in enumerate(rows, 1):
        predicted = pipeline.process(gold, indices, ablate_layers_1_2=ablate)
        pool = ev.build_candidate_pool(gold, indices)
        out.append(
            {
                "gold": gold,
                "pred": predicted,
                "pool_ids": [c.message_id for c in pool],
                "pool_size": len(pool),
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}")
    return out


def routing_cache_stats() -> dict:
    """Cache hits vs live model calls for the routing step of this run.

    Read out of ``router.STATS`` in memory. Deliberately does **not** call
    ``router.STATS.write``: ``run_stats.json`` is the record of the 110-row
    production run, and a scoring pass over 30 different rows must not
    overwrite it.
    """
    rows = router.STATS.rows
    hits = sum(1 for r in rows.values() if r.cache_hit)
    return {
        "rows": len(rows),
        "cache_hits": hits,
        "model_calls": len(rows) - hits,
        "hit_rate": round(hits / len(rows), 4) if rows else None,
    }


# --------------------------------------------------------------------------
# TIER 1 — field accuracy
# --------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def tier1(results) -> dict:
    section("TIER 1 — FIELD ACCURACY")
    n = len(results)
    action_hits = sum(1 for r in results if r["pred"]["action"] == r["gold"]["action"])
    type_hits = sum(
        1 for r in results if r["pred"]["message_type"] == r["gold"]["message_type"]
    )
    both = sum(
        1
        for r in results
        if r["pred"]["action"] == r["gold"]["action"]
        and r["pred"]["message_type"] == r["gold"]["message_type"]
    )

    print(f"\n  rows scored          : {n}")
    print(f"  action accuracy      : {action_hits}/{n} = {action_hits / n:.3f}")
    print(f"  message_type accuracy: {type_hits}/{n} = {type_hits / n:.3f}")
    print(f"  both correct         : {both}/{n} = {both / n:.3f}")

    print("\n  CONFUSION MATRIX on action (rows = gold, cols = predicted)")
    header = "".join(f"{a:>9}" for a in schema.ACTIONS)
    print(f"    {'gold \\ pred':<14}{header}{'total':>9}")
    matrix = {g: {p: 0 for p in schema.ACTIONS} for g in schema.ACTIONS}
    for r in results:
        matrix[r["gold"]["action"]][r["pred"]["action"]] += 1
    for g in schema.ACTIONS:
        row_total = sum(matrix[g].values())
        cells = "".join(f"{matrix[g][p]:>9}" for p in schema.ACTIONS)
        print(f"    {g:<14}{cells}{row_total:>9}")
    col = {p: sum(matrix[g][p] for g in schema.ACTIONS) for p in schema.ACTIONS}
    cells = "".join(f"{col[p]:>9}" for p in schema.ACTIONS)
    print(f"    {'total':<14}{cells}{n:>9}")

    print("\n  PER-ACTION precision / recall / F1")
    print(f"    {'action':<10}{'support':>9}{'prec':>8}{'recall':>8}{'F1':>8}")
    per_action = {}
    f1s = []
    for a in schema.ACTIONS:
        tp = matrix[a][a]
        fp = sum(matrix[g][a] for g in schema.ACTIONS if g != a)
        fn = sum(matrix[a][p] for p in schema.ACTIONS if p != a)
        support = sum(matrix[a].values())
        p, rc, f1 = _prf(tp, fp, fn)
        per_action[a] = {"support": support, "precision": p, "recall": rc, "f1": f1}
        f1s.append(f1)
        print(f"    {a:<10}{support:>9}{p:>8.3f}{rc:>8.3f}{f1:>8.3f}")
    macro_f1 = round(sum(f1s) / len(f1s), 4)
    print(f"\n  MACRO-F1 on action: {macro_f1:.4f}")

    print("\n  PER-MESSAGE_TYPE breakdown (grouped by gold type)")
    print(f"    {'gold type':<18}{'n':>4}{'action ok':>11}{'type ok':>9}")
    by_type = defaultdict(lambda: {"n": 0, "action_ok": 0, "type_ok": 0})
    for r in results:
        bucket = by_type[r["gold"]["message_type"]]
        bucket["n"] += 1
        bucket["action_ok"] += r["pred"]["action"] == r["gold"]["action"]
        bucket["type_ok"] += r["pred"]["message_type"] == r["gold"]["message_type"]
    for key in sorted(by_type, key=lambda k: -by_type[k]["n"]):
        b = by_type[key]
        print(f"    {key:<18}{b['n']:>4}{b['action_ok']:>11}{b['type_ok']:>9}")

    print("\n  MISSES (gold != predicted on action)")
    misses = []
    for r in results:
        if r["pred"]["action"] != r["gold"]["action"]:
            entry = {
                "message_id": r["gold"]["message_id"],
                "gold_action": r["gold"]["action"],
                "pred_action": r["pred"]["action"],
                "gold_type": r["gold"]["message_type"],
                "pred_type": r["pred"]["message_type"],
                "rule_key": r["pred"]["_rule_key"],
                "confidence": r["pred"]["confidence"],
            }
            misses.append(entry)
            print(
                f"    {entry['message_id']:<16}{entry['gold_action']:>7} -> "
                f"{entry['pred_action']:<7} [{entry['rule_key']}] "
                f"conf={entry['confidence']}  type {entry['gold_type']}->{entry['pred_type']}"
            )
    if not misses:
        print("    (none)")

    # Every scored row, not just the misses. The report needs the rule_key of
    # rows the config got *right* in order to attribute a decision to a ladder
    # layer; reading it back out of the misses list can only ever describe
    # failures.
    per_row = [
        {
            "message_id": r["gold"]["message_id"],
            "gold_action": r["gold"]["action"],
            "pred_action": r["pred"]["action"],
            "rule_key": r["pred"]["_rule_key"],
            "confidence": r["pred"]["confidence"],
            "tier": r["pred"]["_tier"],
            "tier_signals": r["pred"]["_tier_signals"],
            "correct": r["pred"]["action"] == r["gold"]["action"],
        }
        for r in results
    ]

    return {
        "n": n,
        "action_accuracy": round(action_hits / n, 4),
        "message_type_accuracy": round(type_hits / n, 4),
        "both_correct": round(both / n, 4),
        "confusion_matrix": matrix,
        "per_action": per_action,
        "macro_f1_action": macro_f1,
        "per_message_type": {k: dict(v) for k, v in by_type.items()},
        "misses": misses,
        "per_row": per_row,
    }


# --------------------------------------------------------------------------
# TIER 2 — evidence grounding
# --------------------------------------------------------------------------


def _split(raw: str) -> list[str]:
    if not raw or raw == schema.EVIDENCE_NONE:
        return []
    return [p for p in raw.split(schema.EVIDENCE_SEPARATOR) if p]


def tier2(results) -> dict:
    section("TIER 2 — EVIDENCE GROUNDING")

    print("\n  (a) INVARIANT — every emitted id came from that row's candidate pool")
    emitted = in_pool = 0
    violations = []
    for r in results:
        ids = _split(r["pred"]["evidence_message_ids"])
        emitted += len(ids)
        for i in ids:
            if i in r["pool_ids"]:
                in_pool += 1
            else:
                violations.append({"message_id": r["gold"]["message_id"], "id": i})
    fraction = (in_pool / emitted) if emitted else 1.0
    print(f"      emitted ids: {emitted}   in pool: {in_pool}   fraction: {fraction:.4f}")
    print(f"      violations : {len(violations)} "
          f"{violations if violations else '(none — enforced in code, not requested in the prompt)'}")

    print("\n  (b) PRECISION / RECALL / F1 against gold evidence ids")
    tp = fp = fn = exact = 0
    for r in results:
        pred = set(_split(r["pred"]["evidence_message_ids"]))
        gold = set(_split(r["gold"]["evidence_message_ids"]))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += pred == gold
    p, rc, f1 = _prf(tp, fp, fn)
    print(f"      tp={tp}  fp={fp}  fn={fn}")
    print(f"      precision={p:.3f}  recall={rc:.3f}  F1={f1:.3f}")
    print(f"      exact set match: {exact}/{len(results)} = {exact / len(results):.3f}")

    print("\n  (c) AGREEMENT on rows where gold says 'none'")
    gold_none = [r for r in results if not _split(r["gold"]["evidence_message_ids"])]
    agree = sum(1 for r in gold_none if not _split(r["pred"]["evidence_message_ids"]))
    pred_none = [r for r in results if not _split(r["pred"]["evidence_message_ids"])]
    print(f"      gold 'none' rows : {len(gold_none)} "
          f"{[r['gold']['message_id'] for r in gold_none]}")
    rate = f" = {agree / len(gold_none):.3f}" if gold_none else ""
    print(f"      we also said none: {agree}/{len(gold_none)}{rate}")
    print(f"      we said none     : {len(pred_none)} "
          f"{[r['gold']['message_id'] for r in pred_none]}")
    for r in gold_none:
        if _split(r["pred"]["evidence_message_ids"]):
            print(f"        DISAGREE {r['gold']['message_id']}: gold none, we emitted "
                  f"{r['pred']['evidence_message_ids']} (pool_size={r['pool_size']})")

    return {
        "emitted_ids": emitted,
        "ids_in_pool": in_pool,
        "in_pool_fraction": round(fraction, 4),
        "pool_violations": violations,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": rc,
        "f1": f1,
        "exact_set_match": round(exact / len(results), 4),
        "gold_none_rows": [r["gold"]["message_id"] for r in gold_none],
        "gold_none_agreement": round(agree / len(gold_none), 4) if gold_none else None,
        "pred_none_rows": [r["gold"]["message_id"] for r in pred_none],
    }


# --------------------------------------------------------------------------
# TIER 2b — reason quality
# --------------------------------------------------------------------------

JUDGE_TOOL = {
    "name": "record_verdict",
    "description": "Record whether two routing explanations express the same rule.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["same_rule", "compatible", "contradictory"],
                "description": (
                    "same_rule: both cite the same underlying reason. "
                    "compatible: different reasons, both plausible, pointing the "
                    "same way. contradictory: they cannot both be the reason, or "
                    "they imply opposite routing."
                ),
            },
            "justification": {
                "type": "string",
                "description": "One short sentence.",
            },
        },
        "required": ["verdict", "justification"],
        "additionalProperties": False,
    },
}

JUDGE_SYSTEM = """\
You compare two short explanations for why a WhatsApp message was routed a
particular way. One is the reference explanation, one is a system's output.

Judge only whether they express the SAME UNDERLYING RULE. Do not judge writing
quality, tone, length, or which one you prefer.

- same_rule: both cite the same reason — e.g. both say the sender is trusted and
  the message is not urgent, or both say the message asks for credentials.
- compatible: different reasons, both plausibly true of the same message, and
  both pointing toward the same routing outcome.
- contradictory: the two cannot both be the operative reason, or they point
  toward different routing outcomes.

Neither explanation is authoritative about the message itself. You are comparing
the two strings, not adjudicating the message.
"""


def tier2b(results) -> dict:
    section("TIER 2b — REASON QUALITY")

    print("\n  (a) RULE-KEY STABILITY — do rows sharing (gold action, gold type)")
    print("      receive a single rule_key?")
    groups = defaultdict(list)
    for r in results:
        groups[(r["gold"]["action"], r["gold"]["message_type"])].append(r)
    print(f"\n    {'gold (action, type)':<28}{'n':>3}  rule_keys")
    unstable = []
    stability = {}
    for key in sorted(groups):
        members = groups[key]
        keys = Counter(m["pred"]["_rule_key"] for m in members)
        label = f"{key[0]}/{key[1]}"
        stable = len(keys) == 1
        stability[label] = {"n": len(members), "rule_keys": dict(keys), "stable": stable}
        if len(members) > 1 and not stable:
            unstable.append(label)
        marker = "" if stable else "   <-- multiple"
        joined = ", ".join(f"{k} x{v}" for k, v in keys.items())
        print(f"    {label:<28}{len(members):>3}  {joined}{marker}")
    multi = [k for k in groups if len(groups[k]) > 1]
    print(f"\n    groups with more than one row: {len(multi)}")
    print(f"    of those, single rule_key     : {len(multi) - len(unstable)}")
    print(f"    unstable                      : {unstable if unstable else '(none)'}")

    print("\n  (b) LLM-AS-JUDGE — reference reason vs emitted reason")
    print("      Rubric: same_rule / compatible / contradictory")
    client = anthropic.Anthropic(api_key=config.require_api_key())
    verdicts = []
    for r in results:
        gold_reason = r["gold"]["reason"]
        pred_reason = r["pred"]["reason"]
        try:
            response = client.messages.create(
                model=config.PRIMARY_MODEL,
                max_tokens=300,
                temperature=0.0,
                system=[
                    {
                        "type": "text",
                        "text": JUDGE_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "record_verdict"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"<reference>{gold_reason}</reference>\n"
                            f"<system_output>{pred_reason}</system_output>"
                        ),
                    }
                ],
            )
            block = next(b for b in response.content if b.type == "tool_use")
            verdict = block.input["verdict"]
            justification = block.input.get("justification", "")
        except Exception as exc:  # noqa: BLE001 — a judge failure is not a scoring failure
            verdict, justification = "judge_error", f"{type(exc).__name__}: {exc}"
        verdicts.append(
            {
                "message_id": r["gold"]["message_id"],
                "gold_action": r["gold"]["action"],
                "pred_action": r["pred"]["action"],
                "rule_key": r["pred"]["_rule_key"],
                "verdict": verdict,
                "justification": justification,
                "gold_reason": gold_reason,
                "pred_reason": pred_reason,
            }
        )

    print("\n      ALL VERDICTS — hand-check these before trusting the aggregate\n")
    for v in verdicts:
        flag = "   <-- action also wrong" if v["gold_action"] != v["pred_action"] else ""
        print(f"      {v['message_id']}  [{v['verdict']}]  {v['rule_key']}{flag}")
        print(f"        reference : {v['gold_reason']}")
        print(f"        emitted   : {v['pred_reason']}")
        print(f"        judge     : {v['justification']}")
        print()

    counts = Counter(v["verdict"] for v in verdicts)
    total = len(verdicts) or 1
    print(f"      {'verdict':<18}{'n':>4}{'pct':>8}")
    print("      " + "-" * 30)
    for key in ("same_rule", "compatible", "contradictory", "judge_error"):
        if counts[key]:
            print(f"      {key:<18}{counts[key]:>4}{counts[key] / total * 100:>7.1f}%")

    return {
        "rule_key_stability": stability,
        "groups_with_multiple_rows": len(multi),
        "unstable_groups": unstable,
        "judge_counts": dict(counts),
        "judge_verdicts": verdicts,
    }


# --------------------------------------------------------------------------
# TIER 3 — consistency
# --------------------------------------------------------------------------

COMPARE_FIELDS = (
    "user_open_rate_30d",
    "user_dismiss_rate_30d",
    "user_report_count_30d",
    "trailing_daily_load",
    "dnd_active",
    "is_group",
    "group_type",
    "group_muted_by_user",
    "user_role",
    "sender_role",
    "is_direct_mention",
    "n_prior",
    "open_rate",
    "dismiss_rate",
    "report_rate",
    "sender_history_strength",
    "is_business",
    "verified",
    "has_relationship",
    "allows_promotions",
    "opted_out",
    "forwarded_count",
)


def tier3(results) -> dict:
    section("TIER 3 — CONSISTENCY ON SHARED MEDIA")
    print("\n  Identical bytes reaching different recipients. If every row gets the")
    print("  same action, personalization is not working. If actions differ, the")
    print("  differing user features should be the explanation.")

    indices = feat.Indices.load()
    predicted = {}
    if config.OUTPUT_CSV.exists():
        with config.OUTPUT_CSV.open(newline="", encoding="utf-8") as handle:
            predicted = {r["message_id"]: r for r in csv.DictReader(handle)}

    clusters = {}
    flags = []
    for media_id in SHARED_MEDIA:
        members = [r for r in data_layer.load_messages() if r.get("media_id") == media_id]
        members += [
            r for r in data_layer.load_sample_messages() if r.get("media_id") == media_id
        ]
        if len(members) < 2:
            continue
        print(f"\n  --- {media_id}  ({len(members)} rows) ---")
        rows_info = []
        for row in members:
            f = feat.build_features(row, indices)
            mid = row["message_id"]
            if mid in predicted:
                action, source = predicted[mid]["action"], "output.csv"
            else:
                match = next((r for r in results if r["gold"]["message_id"] == mid), None)
                action = match["pred"]["action"] if match else "?"
                source = "labeled run" if match else "not run"
            gold = row.get("action", "")
            rows_info.append((mid, f, action, gold, source))
            gold_note = f"  gold={gold}" if gold else ""
            print(f"    {mid:<16} recipient={f.user_id:<7} action={action:<7}"
                  f"{gold_note}  ({source})")

        differing = [
            field
            for field in COMPARE_FIELDS
            if len({getattr(r[1], field) for r in rows_info}) > 1
        ]
        print(f"    features that DIFFER across the cluster ({len(differing)}):")
        for field in differing:
            vals = ", ".join(f"{r[0]}={getattr(r[1], field)!r}" for r in rows_info)
            print(f"      {field:<26}{vals}")

        actions = {r[2] for r in rows_info}
        if len(actions) == 1 and differing:
            flag = (
                f"{media_id}: identical action {sorted(actions)[0]!r} across "
                f"{len(rows_info)} rows despite {len(differing)} differing features"
            )
            flags.append(flag)
            print(f"    FLAG: {flag}")
        elif len(actions) > 1 and not differing:
            flag = (
                f"{media_id}: differing actions {sorted(actions)} with identical "
                "user features"
            )
            flags.append(flag)
            print(f"    FLAG: {flag}")
        else:
            print(f"    OK: actions {sorted(actions)} with {len(differing)} differing features")

        clusters[media_id] = {
            "rows": [
                {
                    "message_id": r[0],
                    "recipient": r[1].user_id,
                    "action": r[2],
                    "gold_action": r[3],
                    "source": r[4],
                }
                for r in rows_info
            ],
            "differing_features": differing,
            "distinct_actions": sorted(actions),
        }

    return {"clusters": clusters, "flags": flags}


# --------------------------------------------------------------------------
# CALIBRATION
# --------------------------------------------------------------------------


def calibration(results) -> dict:
    section("CALIBRATION — accuracy per emitted confidence value")
    buckets = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        bucket = buckets[r["pred"]["confidence"]]
        bucket["n"] += 1
        bucket["correct"] += r["pred"]["action"] == r["gold"]["action"]

    print(f"\n  {'confidence':<12}{'n':>5}{'correct':>9}{'accuracy':>10}")
    print("  " + "-" * 36)
    table = {}
    for key in sorted(buckets):
        b = buckets[key]
        accuracy = b["correct"] / b["n"]
        table[key] = {"n": b["n"], "correct": b["correct"], "accuracy": round(accuracy, 4)}
        print(f"  {key:<12}{b['n']:>5}{b['correct']:>9}{accuracy:>10.3f}")

    ordered = sorted((float(k), v["accuracy"], v["n"]) for k, v in table.items())
    print("\n  Does accuracy rise with confidence?")
    monotonic = None
    separation = None
    if len(ordered) < 2:
        print("    too few distinct values to say")
    else:
        monotonic = all(
            ordered[i][1] <= ordered[i + 1][1] for i in range(len(ordered) - 1)
        )
        lo = [o for o in ordered if o[0] <= 0.84]
        hi = [o for o in ordered if o[0] > 0.84]
        lo_acc = sum(o[1] * o[2] for o in lo) / sum(o[2] for o in lo) if lo else 0.0
        hi_acc = sum(o[1] * o[2] for o in hi) / sum(o[2] for o in hi) if hi else 0.0
        separation = round(hi_acc - lo_acc, 4)
        print(f"    strictly monotonic across every value: {monotonic}")
        print(f"    low band  (<= 0.84): n={sum(o[2] for o in lo):>3}  accuracy={lo_acc:.3f}")
        print(f"    high band ( > 0.84): n={sum(o[2] for o in hi):>3}  accuracy={hi_acc:.3f}")
        print(f"    separation: {separation:+.3f}")
        if hi_acc <= lo_acc:
            print("    WARNING: confidence does not track correctness — the tier")
            print("             definition is wrong and the field is decoration.")

    return {"buckets": table, "monotonic": monotonic, "band_separation": separation}


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("a", "b"), default="a")
    parser.add_argument("--out", default=None, help="metrics path (default metrics.json)")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve() if args.out else METRICS_PATH
    ablate = args.config == "b"

    started = time.monotonic()
    print(f"CONFIG {args.config.upper()} — {CONFIG_DESCRIPTIONS[args.config]}")
    results = run_pipeline_on_samples(ablate)
    routing = routing_cache_stats()
    print(
        f"routing: {routing['cache_hits']}/{routing['rows']} cache hits, "
        f"{routing['model_calls']} live model calls"
    )

    metrics = {
        "config": args.config,
        "config_description": CONFIG_DESCRIPTIONS[args.config],
        "model": config.PRIMARY_MODEL,
        "prompt_version": config.PROMPT_VERSION,
        "scored_rows": len(results),
        "routing_cache": routing,
        "tier1_field_accuracy": tier1(results),
        "tier2_evidence_grounding": tier2(results),
        "tier2b_reason_quality": (
            {"skipped": True} if args.skip_judge else tier2b(results)
        ),
        "tier3_consistency": tier3(results),
        "calibration": calibration(results),
        "degraded_during_scoring": len(pipeline.DEGRADED),
        "elapsed_s": round(time.monotonic() - started, 2),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    section("SUMMARY")
    t1 = metrics["tier1_field_accuracy"]
    t2 = metrics["tier2_evidence_grounding"]
    print(f"\n  action accuracy      : {t1['action_accuracy']:.3f}")
    print(f"  message_type accuracy: {t1['message_type_accuracy']:.3f}")
    print(f"  macro-F1 (action)    : {t1['macro_f1_action']:.4f}")
    print(f"  evidence F1          : {t2['f1']:.3f}")
    print(f"  in-pool invariant    : {t2['in_pool_fraction']:.4f}")
    print(f"  degraded rows        : {metrics['degraded_during_scoring']}")
    print(f"  routing cache hits   : {routing['cache_hits']}/{routing['rows']}")
    print(f"\n  metrics -> {out_path}")
    print(f"  elapsed {metrics['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
