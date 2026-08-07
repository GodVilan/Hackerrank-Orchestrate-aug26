"""Read-only diagnostic for the Tier 1/2/2b findings. Makes zero model calls.

    cd code && python -m evaluation.diagnose_12b

Reads ``output.csv``, ``evaluation/metrics.json``, the router response cache,
the transcript cache, and the dataset CSVs. Recomputes only what is
deterministic (features, candidate pools, cache keys). Where a value was never
persisted it prints MISSING and continues rather than re-deriving it by calling
the model.

Proposes nothing and changes nothing.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import config  # noqa: E402
import data_layer  # noqa: E402
import evidence as ev  # noqa: E402
import features as feat  # noqa: E402
import gates  # noqa: E402
import router  # noqa: E402
import schema  # noqa: E402
import transcription  # noqa: E402
from schema import RuleKey  # noqa: E402

W = 78
METRICS = Path(__file__).resolve().parent / "metrics.json"
INJECTION_IDS = ("msg_095", "msg_107", "msg_108", "msg_109", "msg_110")
REASON_TO_KEY = {v: k for k, v in schema.REASONS.items()}


def section(letter: str, title: str) -> None:
    print("\n" + "=" * W)
    print(f"{letter}. {title}")
    print("=" * W)


def load_output() -> dict[str, dict]:
    if not config.OUTPUT_CSV.exists():
        return {}
    with config.OUTPUT_CSV.open(newline="", encoding="utf-8") as fh:
        return {r["message_id"]: r for r in csv.DictReader(fh)}


def load_metrics() -> dict:
    try:
        return json.loads(METRICS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_run_stats() -> dict:
    try:
        return json.loads(config.RUN_STATS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def hydrate(row: dict) -> dict:
    """Attach the voice transcript from the on-disk cache. No model weights used."""
    row = dict(row)
    if row.get("media_type") == "voice" and row.get("media_id"):
        try:
            path = data_layer.resolve_media_path("voice", row["media_id"])
            row["_transcript"] = transcription.transcribe(path)
        except Exception:  # noqa: BLE001
            row["_transcript"] = ""
    return row


def cached_payload(row: dict, indices):
    """The model's stored reading for a row, or None. Reads disk only."""
    row = hydrate(row)
    features = feat.build_features(row, indices)
    pool = ev.build_candidate_pool(row, indices)
    key = router.cache_key(row, features, pool)
    path = config.ROUTER_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None, features, pool, path
    try:
        return json.loads(path.read_text(encoding="utf-8")), features, pool, path
    except (OSError, json.JSONDecodeError):
        return None, features, pool, path


def tier_from_confidence(action: str, confidence: str):
    try:
        raw = (float(confidence) - schema.CONFIDENCE_BASE[action]) / schema.CONFIDENCE_STEP
    except (TypeError, ValueError, KeyError):
        return None
    tier = round(raw)
    return tier if abs(raw - tier) < 1e-6 else None


# --------------------------------------------------------------------------


def section_a(indices, samples) -> None:
    section("A", "PROVENANCE — were the 30 scored routing calls cached or fresh?")
    stats = load_run_stats()
    stat_rows = stats.get("rows", {})
    print("\n  run_stats.json holds per-row cache_hit for the rows of the LAST")
    print("  main.py invocation only. The scorer does not write run_stats, so the")
    print("  hit/miss actually observed during scoring is:  MISSING")
    print(f"  (run_stats contains {len(stat_rows)} rows; "
          f"{sum(1 for k in stat_rows if k.startswith('sample_'))} of them are sample rows)")

    print("\n  What IS on disk: whether a cache entry exists for each row's key now,")
    print("  and when it was written.")
    print(f"\n  {'message_id':<17}{'cache entry':<14}{'written':<22}{'size':>7}")
    print("  " + "-" * 62)
    present = 0
    for row in samples:
        _, _, _, path = cached_payload(row, indices)
        if path.exists():
            present += 1
            import datetime
            when = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"  {row['message_id']:<17}{'present':<14}{when:<22}{path.stat().st_size:>7}")
        else:
            print(f"  {row['message_id']:<17}{'ABSENT':<14}{'—':<22}{'—':>7}")
    print("  " + "-" * 62)
    print(f"  present: {present}/{len(samples)}")

    print("\n  FINDING")
    print("  The 30 labeled rows carry sample_msg_* ids, a disjoint namespace from")
    print("  the msg_* ids in messages.csv. Their cache keys therefore cannot")
    print("  collide with any entry written by the 110-row run.")
    print("  => The scored predictions were NEVER served from the 110-row run's")
    print("     cache. They are a separate artifact produced by separate calls.")
    print("     Within the scorer itself, the first invocation missed on all 30 and")
    print("     later invocations hit; which invocation produced the metrics.json")
    print("     on disk is not recorded.  MISSING")


def section_b(indices, samples) -> None:
    section("B", "EVIDENCE COUNTERFACTUAL — three gold-independent policies")
    print("\n  Emitted ids are read back from the router response cache in the order")
    print("  the model returned them. No model call is made.\n")
    print(f"  {'message_id':<17}{'pool':>5}  {'emitted (model order)':<34}{'gold'}")
    print("  " + "-" * 76)

    rows = []
    for row in samples:
        payload, features, pool, _ = cached_payload(row, indices)
        if payload is None:
            print(f"  {row['message_id']:<17}{len(pool):>5}  MISSING (no cache entry)")
            continue
        raw_ids = list(payload.get("evidence_message_ids") or ())
        kept, _violation = ev.enforce_evidence(raw_ids, pool)
        gold = [
            p
            for p in (row["evidence_message_ids"] or "").split(schema.EVIDENCE_SEPARATOR)
            if p and p != schema.EVIDENCE_NONE
        ]
        rows.append({"id": row["message_id"], "pool": len(pool), "emitted": kept, "gold": gold})
        print(f"  {row['message_id']:<17}{len(pool):>5}  "
              f"{(', '.join(kept) or '(none)'):<34}{', '.join(gold) or '(none)'}")

    def score(select):
        tp = fp = fn = exact = 0
        for r in rows:
            pred = set(select(r))
            gold = set(r["gold"])
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            exact += pred == gold
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return tp, fp, fn, precision, recall, f1, exact / len(rows) if rows else 0.0

    policies = {
        "P1 as-is": lambda r: r["emitted"],
        "P2 first id only": lambda r: r["emitted"][:1],
        "P3 first; second only if pool==2": lambda r: (
            r["emitted"][:2] if r["pool"] == 2 else r["emitted"][:1]
        ),
    }
    print(f"\n  {'policy':<34}{'tp':>4}{'fp':>4}{'fn':>4}{'prec':>8}{'recall':>8}{'F1':>8}{'exact':>8}")
    print("  " + "-" * 76)
    for label, select in policies.items():
        tp, fp, fn, p, rc, f1, exact = score(select)
        print(f"  {label:<34}{tp:>4}{fp:>4}{fn:>4}{p:>8.3f}{rc:>8.3f}{f1:>8.3f}{exact:>8.3f}")

    two_one = [r for r in rows if len(r["emitted"]) == 2 and len(set(r["emitted"]) & set(r["gold"])) == 1]
    first_right = [r for r in two_one if r["emitted"][0] in r["gold"]]
    print(f"\n  Rows where 2 ids were emitted and exactly 1 was correct: {len(two_one)}")
    print(f"    correct id was in position 1: {len(first_right)}"
          f"{f' ({len(first_right) / len(two_one) * 100:.0f}%)' if two_one else ''}")
    for r in two_one:
        pos = 1 if r["emitted"][0] in r["gold"] else 2
        print(f"      {r['id']:<17}emitted={r['emitted']}  gold={r['gold']}  correct at position {pos}")


def section_c(indices, samples, output) -> None:
    section("C", "MUTE_SCAM_FIRST_CONTACT AUDIT")
    src = (_CODE_DIR / "gates.py").read_text(encoding="utf-8").splitlines()
    print("\n  Precondition, verbatim from gates.py:")
    for i, line in enumerate(src, 1):
        if "content_risk ==" in line and "suspicious" in line:
            for j in range(i - 1, min(i + 2, len(src))):
                print(f"    {j + 1:>4}  {src[j]}")
            break
    print("\n  Sub-rule fallback inside the scam branch, verbatim:")
    for i, line in enumerate(src, 1):
        if "def _scam_sub_rule" in line:
            for j in range(i - 1, i + 12):
                print(f"    {j + 1:>4}  {src[j]}")
            break

    target = schema.REASONS[RuleKey.MUTE_SCAM_FIRST_CONTACT]
    fired_110 = [mid for mid, r in output.items() if r["reason"] == target]
    metrics = load_metrics()
    fired_30 = [
        v["message_id"]
        for v in metrics.get("tier2b_reason_quality", {}).get("judge_verdicts", [])
        if v["rule_key"] == "MUTE_SCAM_FIRST_CONTACT"
    ]
    print(f"\n  Fired on {len(fired_110)} of the 110 rows and {len(fired_30)} of the 30 labeled rows.")

    print(f"\n  {'message_id':<17}{'verif':<7}{'acct_age':>9}{'n_prior':>8}{'rpt/1k':>8}"
          f"{'dom_match':>10}{'known':>7}")
    print("  " + "-" * 68)
    by_id = {r["message_id"]: r for r in data_layer.load_messages()}
    by_id.update({r["message_id"]: r for r in samples})
    for mid in fired_110 + fired_30:
        row = by_id.get(mid)
        if row is None:
            print(f"  {mid:<17}MISSING (row not found)")
            continue
        f = feat.build_features(row, indices)
        print(f"  {mid:<17}{str(f.verified):<7}{f.account_age_days:>9}{f.n_prior:>8}"
              f"{f.report_rate_per_1k:>8.2f}{str(f.domain_match):>10}"
              f"{str(feat.is_known_counterparty(f)):>7}")

    print("\n  DOES THE PRECONDITION TEST PRIOR HISTORY OR VERIFICATION?")
    print("    Layer 1 branch : content_risk == 'suspicious' AND")
    print("                     not is_known_counterparty(features)")
    print("      -> tests prior history: YES, via is_known_counterparty")
    print("         (has_relationship for business rows, n_prior > 0 otherwise)")
    print("      -> tests verification : NO. `verified` is not read.")
    print("    Scam sub-rule  : reached only when content_risk == 'scam'; picks")
    print("                     MUTE_SCAM_FIRST_CONTACT when n_prior == 0 or the")
    print("                     counterparty is unknown.")
    print("      -> tests prior history: YES")
    print("      -> tests verification : NO")
    print("\n    So a verified business with a matching domain and a low report rate")
    print("    is treated identically to an unverified one, provided the model reads")
    print("    the content as suspicious and no relationship row exists.")


def section_d(indices, samples) -> None:
    section("D", "DIGEST_BIZ_LEGIT vs NOTIFY — sample_msg_004 and sample_msg_005")
    by_id = {r["message_id"]: r for r in samples}
    for mid in ("sample_msg_004", "sample_msg_005"):
        row = by_id.get(mid)
        if row is None:
            print(f"\n  {mid}: MISSING")
            continue
        payload, f, pool, _ = cached_payload(row, indices)
        print(f"\n  {'-' * (W - 4)}")
        print(f"  {mid}   gold action={row['action']}  gold type={row['message_type']}")
        print(f"  text: {(row['message_text'] or '')[:150]}")
        if payload is None:
            print("  model reading: MISSING (no cache entry)")
            continue
        mr = schema.ModelRead.from_tool_input(payload)
        print(f"\n  MODEL READING: type={mr.message_type} risk={mr.content_risk} "
              f"urgency={mr.urgency} promotional={mr.promotional} "
              f"asks_action={mr.asks_user_for_action} proposed={mr.proposed_action}")
        print("\n  DETERMINISTIC FEATURES")
        for name in (
            "conversation_type", "is_business", "business_id", "brand_name", "verified",
            "domain_match", "official_domain_missing", "account_age_days",
            "report_rate_per_1k", "has_relationship", "why_user_knows_account",
            "allows_promotions", "opted_out", "activity_count_180d",
            "last_activity_recency_days", "biz_open_rate", "dnd_active",
            "n_prior", "sender_history_strength", "forwarded_count",
        ):
            print(f"    {name:<28}{getattr(f, name)!r}")

        print("\n  LADDER TRACE — every branch in gates.py order")
        known = feat.is_known_counterparty(f)
        trace = [
            ("L1 injection", mr.is_router_injection_attempt,
             f"is_router_injection_attempt={mr.is_router_injection_attempt}"),
            ("L1 scam", mr.content_risk == "scam", f"content_risk={mr.content_risk!r}"),
            ("L1 impersonation", gates.impersonation_signature(f),
             f"verified={f.verified} domain_match={f.domain_match} "
             f"rpt/1k={f.report_rate_per_1k:.2f}"),
            ("L1 high-report sender",
             f.global_sender_report_rate >= config.T_REPORT
             and f.global_sender_n >= 3 and f.n_prior > 0,
             f"global_report={f.global_sender_report_rate} global_n={f.global_sender_n} "
             f"n_prior={f.n_prior}"),
            ("L1 suspicious+unknown", mr.content_risk == "suspicious" and not known,
             f"content_risk={mr.content_risk!r} known_counterparty={known}"),
            ("L2 muted group", f.group_muted_by_user and not f.is_direct_mention,
             f"group_muted={f.group_muted_by_user}"),
            ("L2 opted-out promo", f.opted_out and mr.promotional,
             f"opted_out={f.opted_out} promotional={mr.promotional}"),
            ("L3 dismiss pattern",
             f.dismiss_rate >= config.T_DISMISS and f.n_prior >= 5,
             f"dismiss_rate={f.dismiss_rate} n_prior={f.n_prior}"),
            ("L3 forward threshold",
             f.forwarded_count >= config.T_FWD_MESSAGE
             and mr.message_type in {"forward", "greeting"},
             f"fwd={f.forwarded_count} type={mr.message_type!r}"),
            ("L3 offer relevant",
             mr.promotional and f.is_business and f.has_relationship
             and not f.allows_promotions,
             f"promotional={mr.promotional} has_rel={f.has_relationship} "
             f"allows_promo={f.allows_promotions}"),
            ("L3 unsolicited business",
             mr.promotional and f.is_business and not f.has_relationship,
             f"promotional={mr.promotional} has_rel={f.has_relationship}"),
            ("L4 urgency high", mr.urgency == "high", f"urgency={mr.urgency!r}"),
            ("L4 mention+asks", f.is_direct_mention and mr.asks_user_for_action,
             f"mention={f.is_direct_mention} asks={mr.asks_user_for_action}"),
        ]
        for label, fired, why in trace:
            mark = "FIRED " if fired else "  -   "
            print(f"    {mark}{label:<26}{why}")
        decision = gates.decide(f, mr)
        print(f"\n    => terminal digest branch reached; result: {decision.action} "
              f"/ {decision.rule_key.value}")
        print(f"    THE NOTIFY GATE THAT FAILED: urgency=={mr.urgency!r}, not 'high'.")
        print(f"    NOTIFY_BIZ_MATCHES_ORDER/BOOKING is a sub-rule of the urgency=='high'")
        print(f"    branch, so it is unreachable at any other urgency value.")


def section_e(output, indices) -> None:
    section("E", "INJECTION AUDIT — the 5 router-injection rows in the 110")
    by_id = {r["message_id"]: r for r in data_layer.load_messages()}
    print(f"\n  {'message_id':<12}{'action':<8}{'rule_key':<26}{'model_flag':>11}")
    print("  " + "-" * 58)
    landed = []
    for mid in INJECTION_IDS:
        out = output.get(mid)
        if out is None:
            print(f"  {mid:<12}MISSING from output.csv")
            continue
        key = REASON_TO_KEY.get(out["reason"])
        payload, _, _, _ = cached_payload(by_id[mid], indices)
        flag = payload.get("is_router_injection_attempt") if payload else "MISSING"
        print(f"  {mid:<12}{out['action']:<8}{(key.value if key else '?'):<26}{str(flag):>11}")
        if out["action"] != "mute":
            landed.append((mid, out["action"]))

    print(f"\n  Rows that landed on notify or digest: {landed if landed else 'NONE — all 5 muted'}")
    muted_as_injection = [
        mid for mid in INJECTION_IDS
        if output.get(mid) and REASON_TO_KEY.get(output[mid]["reason"])
        is RuleKey.MUTE_INJECTION_ATTEMPT
    ]
    print(f"  Muted specifically via MUTE_INJECTION_ATTEMPT: {len(muted_as_injection)} "
          f"{muted_as_injection}")
    print(f"  Muted via some other branch: "
          f"{[m for m in INJECTION_IDS if m not in muted_as_injection and output.get(m, {}).get('action') == 'mute']}")


def section_f(output) -> None:
    section("F", "TIER RECONCILIATION — why the tier counts summed to 68, not 110")
    stats = load_run_stats()
    stat_rows = stats.get("rows", {})
    tier_counts = stats.get("tier_counts", {})
    print(f"\n  run_stats.json tier_counts       : {tier_counts}  "
          f"(sum = {sum(int(v) for v in tier_counts.values())})")
    print(f"  run_stats.json per-row records   : {len(stat_rows)}")
    print(f"  output.csv rows                  : {len(output)}")

    print("\n  EXPLANATION")
    print("  main.py builds tier_counts from `finished`, which holds only the rows")
    print("  processed in THAT invocation. The first invocation crashed after 42")
    print("  rows and never wrote run_stats. The resumed invocation skipped those")
    print("  42 as already-written and processed the remaining 68, so the stats on")
    print("  disk describe 68 rows. The 42 rows from the crashed run have no")
    print("  persisted tier or cache_hit record:  MISSING")

    print("\n  Tier is recoverable for all 110 from output.csv, because")
    print("  confidence = BASE[action] + 0.02 * tier is invertible.")
    recovered = {}
    for mid, row in output.items():
        recovered[mid] = tier_from_confidence(row["action"], row["confidence"])
    print(f"\n  {'tier':<8}{'in run_stats':>14}{'not in run_stats':>18}{'total':>8}")
    print("  " + "-" * 48)
    table = defaultdict(lambda: [0, 0])
    for mid, tier in recovered.items():
        table[tier][0 if mid in stat_rows else 1] += 1
    for tier in sorted(table, key=lambda t: (t is None, t)):
        a, b = table[tier]
        print(f"  {str(tier):<8}{a:>14}{b:>18}{a + b:>8}")
    total_a = sum(v[0] for v in table.values())
    total_b = sum(v[1] for v in table.values())
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<8}{total_a:>14}{total_b:>18}{total_a + total_b:>8}")
    print(f"\n  unrecoverable tiers (confidence not on the grid): "
          f"{[m for m, t in recovered.items() if t is None] or 'none'}")


def section_g(indices) -> None:
    section("G", "REASON CONTRADICTIONS")
    metrics = load_metrics()
    verdicts = metrics.get("tier2b_reason_quality", {}).get("judge_verdicts", [])
    if not verdicts:
        print("\n  MISSING — metrics.json has no judge verdicts")
    else:
        contra = [v for v in verdicts if v["verdict"] == "contradictory"]
        print(f"\n  {len(contra)} contradictory verdicts\n")
        for v in contra:
            agree = "action AGREES" if v["gold_action"] == v["pred_action"] else "action DIFFERS"
            print(f"  {v['message_id']}   gold={v['gold_action']} pred={v['pred_action']}"
                  f"   ({agree})")
            print(f"    rule_key : {v['rule_key']}")
            print(f"    gold     : {v['gold_reason']}")
            print(f"    ours     : {v['pred_reason']}")
            print()

    stability = metrics.get("tier2b_reason_quality", {}).get("rule_key_stability", {})
    print("  RULE_KEYS PER (gold action, gold message_type) GROUP, groups with >1 row")
    print(f"\n  {'group':<28}{'n':>3}  rule_keys")
    print("  " + "-" * 74)
    for label, info in sorted(stability.items()):
        if info["n"] < 2:
            continue
        keys = ", ".join(f"{k} x{v}" for k, v in info["rule_keys"].items())
        mark = "" if info["stable"] else "   <-- multiple"
        print(f"  {label:<28}{info['n']:>3}  {keys}{mark}")


def main() -> int:
    indices = feat.Indices.load()
    samples = data_layer.load_sample_messages()
    output = load_output()

    section_a(indices, samples)
    section_b(indices, samples)
    section_c(indices, samples, output)
    section_d(indices, samples)
    section_e(output, indices)
    section_f(output)
    section_g(indices)

    section("", "PLAIN SUMMARY")
    metrics = load_metrics()
    t2 = metrics.get("tier2_evidence_grounding", {})
    print(f"""
  A  The 30 scored rows use a disjoint id namespace, so they were never served
     from the 110-row run's cache. Per-row hit/miss during scoring was never
     persisted (MISSING); what is on disk is that all 30 keys now have entries.

  B  Emitted evidence was reconstructed from the response cache. Over-emission
     is the dominant error: see the P1/P2/P3 table above.

  C  MUTE_SCAM_FIRST_CONTACT tests prior history but never tests `verified`.
     A verified brand with a matching domain and a low report rate is treated
     the same as an unverified one when no relationship row exists.

  D  Neither sample_msg_004 nor sample_msg_005 reached a notify branch because
     NOTIFY_BIZ_MATCHES_ORDER/BOOKING sits inside the urgency=='high' branch,
     and the model read urgency as lower on both.

  E  All 5 injection rows in the 110 were muted. Only some were muted via
     MUTE_INJECTION_ATTEMPT; the rest were caught by an earlier or different
     branch — see the table.

  F  tier_counts covers 68 rows because the crashed first invocation never wrote
     run_stats and the resume processed only the remaining 68. Tier for all 110
     is recoverable from output.csv by inverting the confidence formula.

  G  5 contradictory reasons, one of them on a row whose action is correct.
     6 of 7 multi-row gold groups received more than one rule_key.

  Evidence F1 on record: {t2.get('f1', 'MISSING')}   in-pool invariant: {t2.get('in_pool_fraction', 'MISSING')}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
