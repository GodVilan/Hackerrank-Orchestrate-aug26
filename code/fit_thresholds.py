"""Print the distributions the ladder thresholds are chosen from. stdout only.

    python code/fit_thresholds.py

Read-only. This script does not choose anything — it prints the evidence so the
constants in ``config.py`` can be reproduced from the repository rather than
taken on trust. Each section ends with the empty band in the observed
distribution: the interval containing no data, inside which every threshold
value behaves identically on this corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import data_layer  # noqa: E402
import features as feat  # noqa: E402

WIDTH = 104


def section(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def empty_band(values: list[float], threshold: float) -> str:
    """Describe the gap in `values` that `threshold` sits inside."""
    uniq = sorted(set(values))
    below = [v for v in uniq if v < threshold]
    above = [v for v in uniq if v >= threshold]
    lo = max(below) if below else float("-inf")
    hi = min(above) if above else float("inf")
    return f"empty band ({lo:g}, {hi:g}) — every value inside behaves identically"


def senders(idx: feat.Indices) -> None:
    section("TABLE 1 — senders in message_history")
    msgs = data_layer.load_messages()
    in_msgs: dict[str, int] = {}
    for row in msgs:
        if row["sender_user_id"]:
            in_msgs[row["sender_user_id"]] = in_msgs.get(row["sender_user_id"], 0) + 1

    print("\nGLOBAL (one row per sender, across every recipient)")
    print(f"  {'sender':<9}{'n':>5}{'open':>7}{'reply':>7}{'dismiss':>9}{'report':>8}"
          f"{'mean_fwd':>10}{'in msgs':>9}")
    print("  " + "-" * (WIDTH - 4))
    rows = []
    for sender, st in idx.sender_global_stats.items():
        n = int(st["n"])
        rows.append(
            (sender, n, st["opened"] / n, st["replied"] / n, st["dismissed"] / n,
             st["reported"] / n, st["fwd_sum"] / n, in_msgs.get(sender, 0))
        )
    for s, n, o, rp, dm, re_, fw, cnt in sorted(rows, key=lambda r: (-r[5], -r[6])):
        print(f"  {s:<9}{n:>5}{o:>7.2f}{rp:>7.2f}{dm:>9.2f}{re_:>8.2f}{fw:>10.2f}{cnt:>9}")

    built = [feat.build_features(r, idx) for r in msgs]

    print("\nPER-PAIR (recipient, sender) — what the ladder actually reads")
    rep = [f.report_rate for f in built if f.n_prior >= 3]
    dis = [f.dismiss_rate for f in built if f.n_prior >= 5]
    fwd = [st["fwd_sum"] / st["n"] for st in idx.sender_global_stats.values()]
    print(f"  report_rate  at n_prior >= 3 : {sorted(set(round(v, 4) for v in rep))}")
    print(f"      T_REPORT = {config.T_REPORT}   fires on "
          f"{sum(1 for f in built if f.report_rate >= config.T_REPORT and f.n_prior >= 3)}"
          f" of {len(built)} rows")
    print(f"      {empty_band(rep, config.T_REPORT)}")
    print(f"  dismiss_rate at n_prior >= 5 : {sorted(set(round(v, 4) for v in dis))}")
    print(f"      T_DISMISS = {config.T_DISMISS}   fires on "
          f"{sum(1 for f in built if f.dismiss_rate >= config.T_DISMISS and f.n_prior >= 5)}"
          f" of {len(built)} rows")
    print(f"      {empty_band(dis, config.T_DISMISS)}")
    print(f"  sender mean_forwarded_count  : {sorted(set(round(v, 2) for v in fwd))}")
    print(f"      T_FWD_SENDER_MEAN = {config.T_FWD_SENDER_MEAN} (sub-rule selection only)")
    print(f"      {empty_band(fwd, config.T_FWD_SENDER_MEAN)}")

    counts: dict[int, int] = {}
    for f in built:
        counts[f.forwarded_count] = counts.get(f.forwarded_count, 0) + 1
    print(f"\n  per-message forwarded_count  : {sorted(counts.items())}")
    print(f"      T_FWD_MESSAGE = {config.T_FWD_MESSAGE}   "
          f"{sum(1 for f in built if f.forwarded_count >= config.T_FWD_MESSAGE)} rows at or above")
    samp = data_layer.load_sample_messages()
    for action in ("mute", "digest", "notify"):
        vals = sorted(int(r["forwarded_count"]) for r in samp if r["action"] == action)
        print(f"      gold {action:<7}: {vals}")


def businesses() -> None:
    section("TABLE 2 — all businesses, sorted by report_rate_per_1k desc")
    biz = data_layer.index_businesses()
    used = {r["business_id"] for r in data_layer.load_messages() if r["business_id"]}
    rows = []
    for bid, b in biz.items():
        official = (b["official_domain"] or "").strip()
        used_dom = (b["domain_used_by_sender"] or "").strip()
        sent = int(b["messages_sent_30d"] or 0)
        reports = int(b["user_reports_30d"] or 0)
        rows.append(
            {
                "bid": bid,
                "brand": b["brand_name"][:19],
                "ver": b["verified"],
                "match": int(bool(official) and official.casefold() == used_dom.casefold()),
                "miss": int(not official),
                "age": int(b["account_age_days"] or 0),
                "dage": int(b["domain_used_by_sender_age_days"] or 0),
                "r1k": (reports / sent * 1000) if sent else 0.0,
                "dom": used_dom[:24],
                "used": bid in used,
            }
        )
    rows.sort(key=lambda r: -r["r1k"])
    print(f"\n  {'business_id':<14}{'brand':<20}{'ver':>4}{'match':>6}{'miss':>5}"
          f"{'acct_age':>9}{'dom_age':>8}{'rpt/1k':>8}  domain_used_by_sender")
    print("  " + "-" * (WIDTH - 4))
    for r in rows[:30]:
        star = " *" if r["used"] else ""
        print(f"  {r['bid']:<14}{r['brand']:<20}{r['ver']:>4}{r['match']:>6}{r['miss']:>5}"
              f"{r['age']:>9}{r['dage']:>8}{r['r1k']:>8.2f}  {r['dom']}{star}")
    print(f"  ... {len(rows) - 30} more, all at or below {rows[30]['r1k']:.2f}")
    print("  (* = this business appears in messages.csv)")

    r1k = [r["r1k"] for r in rows]
    print(f"\n  T_REPORT_PER_1K = {config.T_REPORT_PER_1K}")
    print(f"      {empty_band(r1k, config.T_REPORT_PER_1K)}")
    print(f"  verified == 0                      : {sum(1 for r in rows if r['ver'] == '0')}")
    print(f"  domain mismatch (official non-empty): "
          f"{sum(1 for r in rows if not r['match'] and not r['miss'])}")
    print(f"  official_domain empty              : {sum(1 for r in rows if r['miss'])}")


def signature_probe(idx: feat.Indices) -> None:
    section("PROBE — rows the impersonation signature fires on")
    import gates

    msgs = data_layer.load_messages()
    hits = [(r, feat.build_features(r, idx)) for r in msgs]
    hits = [(r, f) for r, f in hits if gates.impersonation_signature(f)]
    print(f"\n  {len(hits)} of {len(msgs)} rows, "
          f"{len({f.business_id for _, f in hits})} distinct businesses\n")
    for row, f in hits:
        print(f"  {row['message_id']}  recipient={f.user_id}  {f.brand_name!r}")
        print(f"      verified={f.verified}  official={f.official_domain!r}  "
              f"used={f.domain_used_by_sender!r}")
        print(f"      acct_age={f.account_age_days}d  dom_age={f.domain_used_by_sender_age_days}d"
              f"  rpt/1k={f.report_rate_per_1k:.2f}  has_relationship={f.has_relationship}")
        text = (row["message_text"] or "").replace("\n", " ")[:100]
        print(f"      text: {text or '(empty — media-only row)'}")
    biz = data_layer.index_businesses()
    controls = {"business_092": "Thrillophilia, verified, must NOT fire",
                "business_032": "empty official_domain, must NOT fire"}
    print("\n  controls:")
    for bid, note in controls.items():
        stub = feat.MessageFeatures(
            message_id="", user_id="", conversation_type="business", created_at="",
            is_business=True, business_id=bid,
            verified=biz[bid]["verified"] == "1",
            official_domain=(biz[bid]["official_domain"] or "").strip(),
            domain_used_by_sender=(biz[bid]["domain_used_by_sender"] or "").strip(),
            official_domain_missing=not (biz[bid]["official_domain"] or "").strip(),
            domain_match=bool((biz[bid]["official_domain"] or "").strip())
            and (biz[bid]["official_domain"] or "").strip().casefold()
            == (biz[bid]["domain_used_by_sender"] or "").strip().casefold(),
            account_age_days=int(biz[bid]["account_age_days"] or 0),
            domain_used_by_sender_age_days=int(biz[bid]["domain_used_by_sender_age_days"] or 0),
            report_rate_per_1k=int(biz[bid]["user_reports_30d"] or 0)
            / max(int(biz[bid]["messages_sent_30d"] or 1), 1) * 1000,
        )
        print(f"    {bid}: fires={gates.impersonation_signature(stub)}   ({note})")


def fwd_probe(idx: feat.Indices) -> None:
    section("PROBE — gold mute rows with forwarded_count > 0")
    for row in data_layer.load_sample_messages():
        if row["action"] != "mute" or int(row["forwarded_count"]) == 0:
            continue
        f = feat.build_features(row, idx)
        print(f"\n  {row['message_id']}  fwd={f.forwarded_count}  [{f.conversation_type}]"
              f"  gold_type={row['message_type']}")
        print(f"      reason: {row['reason']}")
        print(f"      n_prior={f.n_prior}  dismiss_rate={f.dismiss_rate}  "
              f"mean_fwd={f.mean_forwarded_count}  opted_out={f.opted_out}")
        print(f"      Layer 3 forward branch requires message_type in "
              f"{{forward, greeting}}: "
              f"{'ELIGIBLE' if row['message_type'] in ('forward', 'greeting') else 'EXCLUDED'}")


def main() -> int:
    idx = feat.Indices.load()
    senders(idx)
    businesses()
    signature_probe(idx)
    fwd_probe(idx)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
