# Errata

Defects found by reading the submitted artifact cold **after** it was graded.

**Scope of this file.** It lists only what the shipped documentation does *not* already say.
The submission's own [`code/README.md`](code/README.md) and
[`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md) carry a
measured known-limitations section, a "what was not tested" list, a held-out-vs-fitted table,
and several passages that correct earlier revisions of themselves. Those disclosures are part
of what was graded and are not repeated here. Where an item below overlaps one of them, the
overlap is stated so this file cannot be read as revealing something that was concealed.

**Nothing here has been fixed in `code/`.** That directory is the submission as zipped.
Patching it would make this repository a record of something that was never evaluated.

Three of the four are documentation defects — the code did one thing and a prose claim,
docstring, or filename said another. That is the pattern, and it is worth naming: in a system
whose premise is that code decides rather than the model, the claims *about* the code are the
part that goes unverified. The shipped artifact already caught three instances of this class
itself (`unresolved`, `evidence_violation`, `transcription_failed()`); these are the ones it
missed.

The first entry adds a second pattern the others do not: a crosscheck that renders every
figure from a metrics file cannot catch a figure that never passes through one. The
verification has to cover the renderer, not only what it renders.

---

## 1. The concurrency passage describes a run that did not happen

**Where:** `code/evaluation/evaluation_report.md`, operational section

**What the report shows.** The table reports wall-clock runtime 463.0 s, summed request
latency 462.71 s, and 110 of 110 rows routed live with zero cache hits. The prose beneath it
explains that summed latency exceeds wall clock "whenever the four workers overlap, and falls
below it when rows are served from cache without a request at all."

**Why that is wrong.** Neither disjunct applies. No row was served from cache, and summed
latency is *below* wall clock — which is what sequential execution looks like, not
overlapping workers. The passage offers an account of concurrency for a table whose own
arithmetic shows the run was serial.

**The machinery exists and is never reached.** `route_many` is defined at `router.py:324`
and opens a `ThreadPoolExecutor(max_workers=config.ROUTER_MAX_WORKERS)` at `router.py:342`.
It has **zero callers** anywhere in the codebase. `router.py:21` advertises concurrency in
the module docstring. Every row was routed one at a time.

**Three sources appear to corroborate it, and all three read the same unused constant.** The
report table is labelled `concurrency (ROUTER_MAX_WORKERS) | 4`, naming the config variable
rather than claiming a measurement; `router.py:96` writes `"concurrency":
config.ROUTER_MAX_WORKERS` into the run stats; and `code/README.md` lists the same setting
under "Concurrency and backoff." None of them is evidence that the setting took effect.

**The compounding defect.** `write_report.py:1206` emits that table row as
`["concurrency (\`ROUTER_MAX_WORKERS\`)", 4]` — a hardcoded integer literal, not a lookup
against `run_stats.json`. Both documents state that no number in the report is transcribed
by hand: `code/README.md` says a stale figure is fixed by re-running the scorer rather than
by editing prose, and `evaluation_report.md` opens its metrics with the same guarantee. The
single number in the report describing something that did not happen is also the single
number typed in by hand. That is not a coincidence — a value read from a stats file would
have been a value someone had to produce, and producing it would have required the code path
to run.

**Effect on results:** None on correctness. The 463 s figure is accurate as a sequential
runtime. What is false is the account of how it was produced, and the claim that no figure in
the report was hand-entered.

**Recommended check, not a claim:** other literals in `write_report.py`'s table
constructors have not been audited against `run_stats.json`. One confirmed exception to a
stated invariant is reason to check the rest, not reason to assume they are all broken.

**Lesson:** Dead code with zero callers is a claim about the system. Grep for callers before
describing any component as part of the run. A config value echoed into a stats file is not
evidence that the config took effect. And a crosscheck script that renders numbers from a
metrics file cannot catch a number that bypasses the metrics file — the audit has to cover
the renderer, not only the rendered output.

---

## 2. Phantom retry in an `evidence.py` docstring

**Where:** `code/evidence.py`, `enforce_evidence` docstring

**Claim:** An ID that is not in the candidate pool is described as *the hard failure the
caller retries once, naming the violation, before falling back to `none`.* The retry is
attributed to the caller, not to `enforce_evidence` itself.

**Reality:** No caller retries. **Enforcement is real** — `enforce_evidence` detects
out-of-pool IDs and drops them, the row keeps whatever admissible IDs remain or `none`, and
the count surfaces as the run-level `evidence_violations`. The retry-and-re-ask step
described as living one level up does not exist; violations resolve straight to the fallback
on the first pass.

The rest of the same docstring is accurate: the empty-pool short-circuit does return
`([], False)` without reading `selected_ids` at all, and the over-count and duplicate-ID
cases are handled as described. It is one clause in an otherwise correct docstring, which is
why it survived review.

**Effect on results:** None measurable. The anti-hallucination guarantee holds; the described
recovery mechanism does not exist.

**Relation to shipped disclosure:** `code/README.md` already flags three name-vs-reality
mismatches of exactly this kind — `unresolved: true` is never written, `evidence_violation`
is not a per-row key, and `transcription_failed()` has no caller. This is a fourth instance
that audit did not reach.

**Lesson:** Docstrings are claims and must be verified like code.

---

## 3. `MUTE_UNSOLICITED_BUSINESS` does not carry the verified-brand exemption

**Where:** `code/gates.py`

**Defect:** The verified-brand exemption is present in the Layer 1 branches and in
`_scam_sub_rule`, but not in the `MUTE_UNSOLICITED_BUSINESS` branch. It was wired into two of
the five places that needed it.

**Relation to shipped disclosure:** the evaluation report records the *outcome* — the
exemption releases zero rows across the 110 shipped rows and 2 of the 30 labeled rows, and
both of those still land on `mute` via `MUTE_UNSOLICITED_BUSINESS`, so its measured effect on
action accuracy is zero and it changes the reason string only. What the report does not say is
**why** those two rows still mute: the branch that catches them has no exemption to apply.
The measurement was published; the mechanism behind it was not identified until afterward.

**Effect on results:** Two verified brands messaging from their own official domain were
muted by a rule the exemption, applied consistently, would likely have released. Not
corrected in the shipped artifact.

**Fix:** Write the exemption once as a named helper, then grep every call site.

**Lesson:** An exemption that exists in *some* branches is more dangerous than one that
exists nowhere, because its presence in the codebase reads as coverage. Audit every branch
for exemptions that should apply — an inline predicate copied by hand will be copied into
some places and not others, and the gap is invisible from the outside.

---

## 4. `fit_thresholds.py` is misnamed

**Where:** `code/fit_thresholds.py`

**Defect:** The script fits nothing. It imports `config` and reads the constants *out* — it
never writes them, never searches a parameter space, never optimizes an objective. It is a
read-only diagnostic that prints the distributions the constants were chosen from, including
the empty band each threshold sits inside, so that the choice can be reproduced from the
repository rather than taken on trust. The filename describes a job the file does not do, on
a rubric that grades whether filenames reflect role.

**On the substantive question it raises.** A reader who sees this name will reasonably ask
whether thresholds were fitted to the only labeled data available and then scored on that
same data. The shipped evaluation report answers this directly, in a table headed *"What is
held out, and what is not,"* which itself corrects an earlier draft that had claimed nothing
was fitted:

| quantity | fitted to the labeled rows? |
|---|---|
| `CONFIDENCE_BASE`, `CONFIDENCE_STEP` | **Yes** — read off the 30 gold confidence values |
| `T_FWD_MESSAGE` | **Partly** — bounded above only, by `sample_msg_014`; every value in 3…11 behaves identically on this corpus |
| `T_REPORT`, `T_DISMISS`, `T_FWD_SENDER_MEAN` | No — empty bands in the 110-row distributions |
| `T_REPORT_PER_1K`, impersonation age bounds | No — same method |
| Ladder branch conditions and their ordering | No — not tuned per-row against gold |

The stated consequence — **action accuracy is held out, calibration is not** — is accurate
and stands. The docstring's claim that the constants "can be reproduced from the repository
rather than taken on trust" is true for the empty-band thresholds and incomplete for
`T_FWD_MESSAGE`, which the report qualifies but the docstring does not.

**Also worth naming:** `fwd_probe()` iterates gold `mute` rows and evaluates each against the
Layer 3 forward branch's `message_type in {forward, greeting}` restriction, printing ELIGIBLE
or EXCLUDED. That is a branch condition checked against gold labels, and it is not covered by
the table above. `signature_probe()` hardcodes `business_092` and `business_032` as control
cases — assertions that the impersonation signature does not fire on a verified brand or on a
brand with an empty official domain. They are not in the decision path and affect no output
row, but they are dataset-specific identifiers in committed code.

**Effect on results:** None. The file is read-only and no output row depends on it.

---

## Not errata — already disclosed in the shipped artifact

Recorded here so this file's silence on them is not read as an omission. Each is documented
in `code/README.md` or `evaluation_report.md`, with measured counts:

- **The pipeline is not reproducible at `temperature=0.0`.** 43 of 140 readings changed on a
  cache-cleared re-run; 8 of 110 output rows differed; 1 action changed. Measured and
  published, and named there as the largest known threat to every number in the report.
- **Calibration is not held out**, and the calibration curve is **not strictly monotonic** —
  it dips at 0.84. Both stated, with the second correcting an earlier revision that had
  claimed the ablation bought a tidier curve.
- **The confidence formula was read off the gold data**, not invented: `BASE[action] + 0.02 ×
  tier` reproduces all 30 gold values exactly. Its ordering property — a `notify` at tier 0
  (0.85) landing above a `digest` at tier 3 (0.84) — follows from that formula rather than
  from a design choice, and did not occur on any shipped row.
- **Quiet hours are a downgrade, not a Layer 2 branch**, and bind on 0 of 110 rows. Recorded
  as Design Decision 3, including the period when a since-reverted change made that claim
  briefly false.
- **The Layer 3 `_biz_variant` hoist was applied and reverted**, on the logic rather than on
  a metric: it made `DIGEST_PROMO_OPTED_IN` structurally unreachable — the defect class it
  was meant to remove, inverted — while being worth one row on the labeled set. Documented,
  along with the earlier report revisions that had described it as shipped.
- **The Config A/B comparison is not a result.** Config B scored 0.9000 against Config A's
  0.8333 — 2 rows on n=30, within the declared noise band. The report states plainly that
  **the ladder is not validated by these 30 rows and the case for it is structural rather
  than empirical**, and that Config A ships because Config B deletes the safety and consent
  layers, independently of the score.
- **Evidence in-pool fraction is a regression check, not a quality metric.** Enforcement
  drops out-of-pool IDs before measurement, so it can only report 1.0.
- **`sample_msg_048` is in `sample_messages.csv` only** — not in `messages.csv`, not in
  `results/output.csv`. Scope it accordingly when counting misses.