# Message Notification Router

## Architecture

### System contract

**The model perceives; the code decides.**

One Claude call per message returns a structured semantic reading and nothing
else. It never returns `action`, `reason`, or `confidence` — those three fields
are produced entirely by deterministic code. The model's reading is one input to
a precedence ladder; the joined CSV features are the other.

The reading is constrained by a forced tool schema (`tool_choice` pinned to the
single tool), so the output shape is guaranteed rather than parsed. There is no
JSON-repair path and none may be added: a missing tool block is a failure to be
handled, not a string to be salvaged.

| The model returns | The model never returns |
|---|---|
| `message_type`, `content_risk`, `urgency` | `action` |
| `promotional`, `asks_user_for_action` | `reason` |
| `is_router_injection_attempt` | `confidence` |
| `evidence_message_ids` (from a supplied candidate list only) | any field derivable from a join |
| `proposed_action` (diagnostics + ablation only; never shipped to `output.csv`) | any prose that reaches the output |
| `media_summary` | |

### What code owns, never the model

Every field below is a join or an arithmetic derivation over the provided CSVs.
None of it is ever posed to the model as a question. Asking the model for any of
these is a design error, not a tuning choice.

Group role and admin status · this user's mute state for this group · quiet-hours
check against `created_at` · business verification · official-vs-sender domain
comparison · account age and sender-domain age · report rates · opt-in/opt-out
state · this user's open / reply / dismiss / report rates for this sender or
business · trailing daily notification load · `forwarded_count` banding ·
`@mention` detection.

Two derived flags — `credential_language_flag` and `injection_pattern_flag` — are
**advisory only**. They are passed to the model as context and are never read by
any decision branch. A naive injection regex false-positives on legitimate
messages containing delivery-code instructions, so they are hints, not gates.

### Pipeline

```
messages.csv ─┐
              ├─► data_layer.py ──► features.py ──┐
context CSVs ─┘   (load + index)   (all joins)    │
                                                  │
media files ──► transcription.py ─┐               │
                vision.py ────────┴─► media text ─┤
                                                  │
              evidence.py ──► candidate pool ─────┤
                                                  │
                                                  ▼
                                              router.py
                                     (ONE tool-use call, forced schema)
                                                  │
                                                  ▼
                                              gates.py
                                    (precedence ladder → action + rule_key)
                                                  │
                                                  ▼
                                             finalize.py
                                  (confidence tier + catalogue reason)
                                                  │
                                                  ▼
                                              writer.py ──► output.csv
                                                  │
                                                  ▼
                                          validate_output.py
```

### Module boundaries

| File | Owns | Never does |
|---|---|---|
| `config.py` | paths, model names, thresholds, env-only secrets | any logic |
| `schema.py` | enums, output column order, reason catalogue, rule keys | I/O |
| `data_layer.py` | loading + indexing the 13 CSVs, media path resolution | any decision |
| `features.py` | every deterministic join and derived feature | any model call |
| `transcription.py` | ASR for the voice notes, disk cache | interpreting the transcript |
| `vision.py` | image normalization + base64 blocks | interpreting the image |
| `evidence.py` | deterministic candidate pool construction | selecting evidence |
| `router.py` | the single structured model call | deciding the action |
| `gates.py` | the precedence ladder; emits `(action, rule_key, tier_signals)` | any I/O or model call |
| `finalize.py` | confidence tier, catalogue reason lookup, invariant enforcement | new reasoning |
| `writer.py` | strict CSV emission, exact column order | anything else |
| `validate_output.py` | standalone pre-submission validator | fixing anything |
| `evaluation/main.py` | scoring against the labeled rows | mutating pipeline state |

`gates.py` is a pure function: no I/O, no model calls, no import from
`router.py`. It is the one module that must be exhaustively unit-testable
without a network.

### Evidence emission cap

`finalize.EMITTED_EVIDENCE_IDS = 1`. At most **one** evidence id reaches
`output.csv`, on every row.

The tool schema still permits two (`schema.MAX_EVIDENCE_IDS = 2`), and
`evidence.enforce_evidence` still validates up to two against the candidate
pool. The cap applies at the emission step only, so the three layers do
different jobs: the schema bounds what the model may return, enforcement bounds
what is admissible, and the cap bounds what ships.

**Which id survives is the model's own ranking, not a code decision.**
`format_evidence` keeps the first id in the order the model returned them; no
component re-scores, re-orders, or otherwise chooses between them. This is the
one place where model output is trusted to prioritize rather than merely to be
constrained, and it is worth naming as such — an id that survives is not an id
that code judged best.

The cap is a measured precision trade, not a safety property. Gold cites exactly
one id on 25 of the 30 labeled rows, and on the labeled set the second id was
wrong far more often than right; capping raised evidence F1 and roughly doubled
exact-set-match. It also means the two-id path ships on no row and is therefore
exercised by no scored row. The count of in-pool ids discarded this way is
reported separately as `evidence_truncated` — it is deliberately not folded into
`evidence_violations`, which counts a different thing.

### Precedence ladder (`gates.py`)

Each layer overrides everything below it. A layer that fires returns
immediately, with one documented exception.

**Layer 1 — Safety.** Scam, impersonation, router-injection attempt. Runs before
any personalization and cannot be overridden by engagement history. A sender the
user always opens is still muted if the message is a credential-harvesting
attempt.

**Layer 2 — Hard user state.** Group mute, promotion opt-out, quiet hours.
- *Documented exception:* a muted group still surfaces a direct `@mention` of
  this user. The mute is **lifted, not converted to notify** — control falls
  through to Layers 3–4, which decide the action on the merits.
- Quiet hours **downgrade** `notify` to `digest`. They never mute.

**Layer 3 — Personalization.** This user's behavioral history with this sender,
group, or business.

**Layer 4 — Content substance.** Urgency and usefulness of what the message
says. This is the only layer where the model's semantic reading drives the
outcome, and it is the last to be consulted.

The ordering is the point. Content reasoning cannot invert an integrity or
consent decision because content reasoning is never reached when those fire.
Putting precedence in code makes that inversion structurally impossible rather
than a matter of prompt discipline.

### Failure paths

| Condition | Behavior | Recorded in `run_stats.json` as |
|---|---|---|
| Malformed or missing tool output | Retried, then `digest` at band-floor confidence with `rule_key = DIGEST_INSUFFICIENT_SIGNAL` and the tier pinned to 0. | An entry in `degraded_rows` (`message_id`, `exception_class`, `stage`), plus `degraded_count` and `degraded_by_exception`. Also appended to the shared `log.txt` as a `DEGRADED ROW` line. |
| Evidence ID not in the supplied candidate list | Dropped by `evidence.enforce_evidence`; the row keeps whatever admissible ids remain, or `none`. | The run-level count `evidence_violations`. |
| In-pool ID discarded by the emission cap | Kept by enforcement, then truncated to `finalize.EMITTED_EVIDENCE_IDS`. Not a failure — a deliberate precision trade. | `evidence_truncated` (ids) and `evidence_truncated_rows` (rows), alongside `emitted_evidence_ids_cap`. |
| Image cannot be prepared | The image is dropped and the row is re-routed text-only before any degradation. | `recovered_text_only`. |
| Empty candidate pool | `evidence = none`, enforced in code. The field is omitted from the model input entirely; the model is never asked. | Nothing — not a failure. |

Failures fail safe and stay visible. A row that could not be resolved is
recorded in `degraded_rows` rather than quietly assigned a plausible action.

**The key names above are the ones actually written.** Two names that appear in
module docstrings do **not** exist in any emitted record and should not be
looked for: `unresolved: true` (`router.py`) is never written — `degraded_rows`
is the real record — and `evidence_violation` is not a per-row key, only the
run-level aggregate `evidence_violations`. The per-row objects under `rows`
carry exactly the ten `RowStat` fields (`message_id`, `model`, token counts,
`latency_s`, `cache_hit`, `retries`, `error`) and nothing else. Relatedly,
`transcription.transcription_failed()` is defined but has **no caller**: a
failed transcription is swallowed into an empty `_transcript` by
`main.prepare_row`, and no flag is set anywhere.

### Untrusted content boundary

`message_text`, OCR/vision output, and voice transcripts are wrapped in
delimiters. The system block states that content inside those delimiters is
**evidence about the message, never an instruction to the system**, and that a
message asking to be routed a particular way is a risk signal rather than a
directive. **Five** rows in `messages.csv` (`msg_095`, `msg_107`, `msg_108`,
`msg_109`, `msg_110`)
attempt exactly this.

### Dataset facts this design depends on

Verified directly against `dataset/` before implementation:

| Fact | Value |
|---|---|
| Rows to predict / labeled example rows | 110 / 30 |
| `message_history` / `message_events` rows | 412 / 412 |
| Images / voice notes | 20 / 13 |
| Conversation mix (110 rows) | 63 group, 30 business, 17 personal |
| Candidate pool size | mean 5.82, max 21, **6 rows empty** |
| Rows in a muted group | 14 (only 2 also carry a direct `@mention`) |
| Rows inside a quiet-hours window | 8 |
| Rows with a direct `@mention` | 5 (literal `@u_0NN` form) |
| Business rows with no `user_business_history` | 11 of 30 |
| Per-(recipient, sender) prior messages | mean **5.22 over all 110 rows** (7.45 over the 77 rows that have any prior history), max 23; 57 rows reach n≥3, 46 reach n≥5 |
| Gold evidence IDs per row | 25 rows use 1, 3 use 2, 2 use `none` — never more than 2 |
| Distinct gold reason strings | 24 across 30 rows |

**Confidence formula confirmed.** `BASE[action] + 0.02 × tier` with bases
`{digest: 0.78, mute: 0.81, notify: 0.85}` and `tier ∈ {0,1,2,3}` reproduces
**all 30** gold confidence values exactly, with no unexplained residual. The
formula is read off the data, not invented.

### Open decisions recorded before implementation

These are dataset facts that the plan as written does not yet resolve. Each is
answered before the module that depends on it is built.

1. **`sample_messages.csv` is a disjoint ID namespace, not a labeled subset.**
   IDs run `sample_msg_001`…`sample_msg_053` against `msg_001`…`msg_110`; the
   overlap is zero. The labeled range is **non-contiguous**: 30 rows occupy 53
   id slots, in three blocks — 001–015, 019–020, and 041–053 — with 23 numbers
   absent (016–018, 021–040). Counting the labeled rows from the id range, or
   assuming the range ends at 030, both give the wrong answer; the count comes
   from reading the file. The
   30 labeled rows are additional rows carrying the same 11 input columns, and
   all their entity and media references resolve. Consequence: the 110 shipped
   rows have **no ground truth at all**, so nothing may be tuned per-row against
   them, and evaluation runs the full pipeline over the sample rows as fresh
   inputs.
2. **`has_relationship` is business-only.** Two ladder branches read it without
   a `conversation_type` guard, which would make every suspicious personal or
   group message a first-contact scam and every promotional-sounding group
   message unsolicited business. The branches need an explicit guard.
3. **`opted_out` is not a column.** `allows_promotions = 0` holds for 88 of 106
   relationship rows and is the default never-opted-in state; explicit opt-out
   is the 14 rows carrying a `promotions_opted_out_at` timestamp. The two are
   not interchangeable.
4. **Quiet hours must downgrade, not pre-empt.** As a plain returning branch,
   the quiet-hours rule would convert a Layer-3 mute into a digest — an upgrade,
   contradicting its stated intent. It is implemented as a modifier applied to a
   `notify` outcome, not as an early return.
5. **`message_events` has no single reaction field.** Five independent booleans,
   with 287 of 412 rows setting more than one. The evidence renderer's single
   reaction label needs a documented precedence order.
6. **`why_user_knows_account` is high-cardinality free text** (~90 distinct
   values over 106 rows). Reason-variant selection is keyword rules over that
   string with an explicit default, not a lookup table.
7. **Quiet hours and the muted-group mention exception have zero gold
   coverage.** No labeled row falls in a DND window, and no labeled row is both
   muted and mentioned. Both branches are validated by hand against the shipped
   rows and reported as untested in the evaluation.

## Design Tradeoffs

Decisions recorded when they were made, with the alternative that was rejected
and the cost of being wrong. Both entries predate the implementation of the
component they govern.

### Decision 1 — the action comes from a deterministic ladder, not the model

**Decision.** The routing action is produced by a deterministic precedence
ladder consuming a structured semantic reading from one model call, rather than
by the model directly.

**Alternative considered.** A single call in which the model returns the action
directly, with code performing only schema validation. This is simpler, has
fewer moving parts, and would have shipped sooner.

**Why the ladder.** In the June edition of this hackathon my pipeline reasoned
about content severity before establishing evidence integrity, and the official
feedback was that it needed a first-pass integrity gate deciding the outcome
before any severity reasoning. Here the equivalent inversion risk is that the
model weighs how useful a message sounds before checking whether the sender is
impersonating a brand or whether the user has muted the conversation. Putting
precedence in code makes that inversion structurally impossible rather than a
matter of prompt discipline. A prompt can be argued with; a branch that returns
before the next branch is reached cannot.

**Cost.** The ladder can be wrong in ways the model would not have been. A
branch ordering mistake mislabels a whole class of rows at once, where a model
error is usually row-local. The failure mode is therefore correlated rather than
scattered, which makes it more damaging when it happens and easier to detect
when it does.

**How it will be measured.** Config A is the full ladder. Config B is an
ablation with Layers 1 and 2 removed, writing the model's `proposed_action`
straight through, so it reaches the output unfiltered on exactly the rows those
layers would have caught. The primary metric is action accuracy on the 30
labeled rows in `dataset/sample_messages.csv`. Evidence grounding breaks ties;
it never overrides action accuracy.

**The noise rule, stated once.** One row is 1/n of the score — 3.33 percentage
points on n=30 — and the band is 10pp, which is 3 rows. A delta is reported as
**within noise** when it is strictly below the band; a delta exactly equal to
the band is reported as **on the band edge**, which the rule does not resolve;
only a delta strictly above it is a result. Verdicts are computed in whole rows
rather than percentage points, by one function in
`evaluation/write_report.py`, so that the same difference cannot be called
noise in one place and a result in another. This wording is identical to the
one the evaluation report uses.

**Measured outcome — recorded here because it is not the outcome this decision
predicted.** Config B scored **higher** on action accuracy: **0.9000** against
Config A's **0.8333**, a **2-row difference — 6.67pp, within noise** under the
rule above. Layers 1 and 2 decide **9** of the 30 labeled rows, the model
independently proposed the same action on **8** of them, and the ablation loses
**zero** rows to their removal; the one row where Layer 1 acts on its own
(`sample_msg_049`) is a row it gets wrong. **The ladder is not validated by these
30 rows, and the case for it is structural rather than empirical** — content
reasoning cannot invert an integrity or consent decision because it is never
reached, which is a property of the code rather than of any number here. Config A
ships regardless, because Config B deletes the safety and consent layers and was
never a deployment candidate; that disqualification is independent of the score
and does not soften the result above. Full analysis in
[`evaluation/evaluation_report.md`](evaluation/evaluation_report.md).

**These figures moved once already, and the reason matters.** An earlier revision
of this paragraph read 0.8000 for Config A and called the gap a 3-row difference
landing exactly on the band edge. Nothing in the code changed between the two
readings — the router cache was cleared and the labeled rows were re-routed, and
the model returned a different reading on enough rows to move Config A by one.
A one-row swing flipped the comparison from *on the band edge* to *within
noise*, which is a concrete demonstration of why 30 rows cannot settle this
question. See the Reproducibility section of the evaluation report; the same
caveat applies to every number in this decision record.

### Decision 2 — complementary engagement rates count as one signal

**Decision.** `user_open_rate_30d` and `user_dismiss_rate_30d` count as one
signal, not two, in the confidence tier. The same applies to `biz_open_rate` and
`biz_dismiss_rate`.

**Why.** Both pairs are computed over the same denominator — opened plus
dismissed — and therefore sum to 1.0 by construction. Admitting both would let a
single underlying quantity satisfy two independent-signal conditions, inflating
the tier without adding evidence. The confidence column would then rise for
reasons unrelated to whether the decision is more likely to be right, flattening
the calibration curve the rubric grades.

**Cost.** An engagement pattern that genuinely deserves two corroborating votes
gets one. Some rows will carry a lower confidence than the evidence available
for them would justify. I would rather under-count confidence than report a
calibration curve that is an artifact of double-counting.

### Decision 3 — quiet hours downgrade `notify` only, and bind on zero rows

**Decision.** Quiet hours are a downgrade applied over a `notify` outcome, not a
returning branch in Layer 2. On this corpus they bind on **zero rows**. The 8
`dnd_active` rows and the 5 direct-mention rows are disjoint sets, and the only
mention-independent notify branch requires `urgency == "high"` — which the
downgrade excludes by design.

**Alternative considered.** Widening the downgrade to cover `urgency == "high"`
when the sender is not a known counterparty, so the branch would have some
reach.

**Why not widen.** The one row a widening would reach is legitimate. `msg_093`
is a verified FedEx delivery notice: matching domain, 2.14 reports per thousand,
and text that explicitly disclaims asking for payment or an OTP. Its
`has_relationship` is False only because no `user_business_history` row exists,
which is also true of 11 of the 30 business rows and is not by itself a risk
signal. Delaying that notification would be the wrong outcome. A gate is not
made better by being made to fire.

**Consequence.** `DIGEST_QUIET_HOURS` is in the catalogue and emits on **0 of
the 110 rows**, re-measured directly after the Layer 3 hoist was reverted. That
zero is the measurement, not a failure — and it is reported as such rather than
hidden by loosening the branch until it produces output.

**This claim was briefly false, and the history matters.** While the business
notify branch was hoisted into Layer 3, it produced a `notify` at `urgency=low`
on `msg_008`, which sits inside a DND window — so the downgrade bound on exactly
1 row and this section was wrong for as long as the hoist was in place. The
revert restored it to 0. The lesson is that "binds on zero rows" is a property of
the whole ladder, not of the quiet-hours branch alone: any change that creates a
new low-urgency notify can make it fire, and this passage has to be re-measured
whenever a notify branch moves.

**How it will be measured.** `dryrun_gates.py` reports rule-key reachability
across the full reading space; `DIGEST_QUIET_HOURS` appearing in its
never-reached list is the expected result, not a regression. The evaluation's
"what was not tested" section records that quiet-hours behaviour has no gold
coverage either: none of the 30 labeled rows falls inside a quiet-hours window.

### Decision 4 — `MUTE_HIGH_REPORT_SENDER` reads the global report rate

**Decision.** This branch reads the sender's report rate across every
recipient, not this recipient's own rate, gated at `global_sender_n >= 3` and
`n_prior > 0`.

**Alternative considered.** The per-pair rate — this recipient's own experience
of this sender — which fires on 10 rows against global's 26.

**Why.** The set of rows where per-pair fires and global does not is **empty**
on this corpus. Per-pair was therefore not a different signal; it was the global
signal with an arbitrary attenuation, and every row it caught global catches
too. What per-pair misses is the case that matters most: four rows in which one
recipient with `n_prior = 12` and an 83% open rate receives repeated
OTP-harvesting from a sender 45% of recipients have reported. That is precisely
the situation in which personalization must not be allowed to vouch for a
sender — a user's own good experience is not evidence that the next message is
safe, and Layer 1 exists to stop exactly that inference.

**Cost.** Three plausible-looking first contacts from high-report senders
(`msg_090`, `msg_089`, `msg_096` — a package mix-up, a lost water bottle, a
found passport) would be muted on sender reputation alone, before any content
is read. The `n_prior > 0` clause releases them: a first contact falls through
to the suspicious-and-unknown branch and then to Layer 4, so a stranger is
judged on what they wrote. If they are pretexting, the model's `content_risk`
read is what should catch them.

**Known gap.** `msg_046` is reported twice by its recipient while its sender is
globally clean at 0.083. Variant D does not catch it, and its routing depends
entirely on the model reading `content_risk` correctly. It is the only row in
the corpus where a recipient's own experience flags a sender the population
does not.

**Not chosen for.** Proximity to the gold mute share. The 110 unlabeled rows
and the 30 labeled rows are disjoint sets with no reason to share a
distribution, so that comparison carries no weight and was not used.

### Decision 5 — reaction precedence in the candidate pool

**Decision.** `message_events` sets five independent booleans per row. The
candidate pool renders exactly one reaction label per historical message, using
the precedence `reported > dismissed > replied > opened > no_record`.

**Why.** The label is read by the model as evidence about how this recipient
treated this sender, and a negative signal is not erased by a positive one that
co-occurs. A message that was both replied to and reported is a reported
message.

**Cost.** Co-occurrence is invisible in the rendered pool. A reply that was
later reported and a report with no engagement render identically, so the model
cannot distinguish "engaged then turned against it" from "rejected outright".

**Scope.** 287 of 412 history rows set more than one flag, so this ordering
determines the label on roughly 70% of the corpus. It is a presentation choice
with wide reach, not an edge case.

**Guard against citing it as a data property.** Because the ordering shapes the
majority of labels, the raw per-flag counts are recorded here alongside the
rendered distribution so the evaluation report cannot mistake a precedence
artifact for a fact about the data. `inspect_pools.py` prints the rendered
column; the raw column is measured directly from `message_events.csv`:

| flag | raw rows (of 412) | rendered label | candidates (of 640) |
|---|---:|---|---:|
| `message_opened` | 278 (67.5%) | opened | 136 (21.2%) |
| `message_replied` | 153 (37.1%) | replied | 272 (42.5%) |
| `notification_dismissed` | 134 (32.5%) | dismissed | 140 (21.9%) |
| `message_reported` | 55 (13.3%) | reported | 92 (14.4%) |
| `muted_after_message` | 134 (32.5%) | *not rendered* | — |

`message_opened` is set on more than two-thirds of event rows but survives
precedence on only a fifth of candidates. Any claim that "the corpus is mostly
unopened" would be reading the rendering, not the data. Note also that the two
columns count different populations — 412 event rows against 640 rendered
candidates, since a history row appears in as many pools as it is scoped to.

Every event row sets at least one flag (125 set exactly one, 232 set two, 55 set
three), and every candidate in every pool has an event row, so `no_record` is
defined but occurs zero times in this corpus.

**`muted_after_message` is dropped, and the reason matters.** It is set on 134
of 412 event rows (32.5%) — as often as `notification_dismissed` — and it does
not appear in the rendered candidate pool at all.

It is dropped **because the vocabulary is fixed by the specification**, which
defines the reaction label as one of `opened / replied / dismissed / reported /
no_record`. It is *not* dropped because it is subsumed by `dismissed`. The two
are different acts: dismissing is declining one notification, while muting is a
standing instruction about every future message from that source. Treating them
as the same signal would be a substantive claim about user behaviour, and the
data does not support making it silently — the two flags co-occur on some rows
and not others.

**No downstream component reads it.** It is absent from `MessageFeatures`, from
the candidate pool rendering, and from every branch of the ladder. The only
mute-state signal the system uses is `group_members.group_muted_by_user`, which
is a different column in a different table describing a different thing (this
user's standing state for this group, rather than one historical reaction).

This is a known omission rather than a considered exclusion. A per-sender
"muted after a previous message" rate is a plausible personalization feature
that no layer currently consumes, and it is recorded here so the evaluation's
"what was not tested" section can say so.

## Setup

Python **3.13** (developed and tested on 3.13.7; 3.10+ should work — the code
uses `X | Y` type syntax and `str`/`Enum` mixins, nothing newer).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
```

### Exact dependency versions

Every version is pinned so a run is reproducible across machines.

| Package | Version | Required for | Needed to run `main.py`? |
|---|---|---|---|
| `anthropic` | 0.120.2 | the single structured model call | **Yes** |
| `pandas` | 2.3.2 | loading and indexing the 13 CSVs | **Yes** |
| `Pillow` | 11.3.0 | image normalization before base64 | **Yes** |
| `faster-whisper` | 1.2.1 | ASR for the 13 voice notes | **No** — see below |

**`faster-whisper` is only needed to regenerate the transcript cache.**
`code/cache/transcripts.json` is committed on purpose, so `main.py` runs end to
end without downloading ASR model weights. Install it only if you want to
re-derive the transcripts from the audio; delete the cache file first, then run
`python3 transcription.py`.

The dataset is **not** included — it is the organizers' to distribute. Place it
at `dataset/` relative to the repository root, as the original archive lays it
out. `python3 code/check_dataset.py` verifies every file and reference resolves
and exits non-zero if anything is missing.

## Environment Variables

**One variable, and it is the only secret this project has.**

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Authenticates the router and the evaluation judge |

```bash
export ANTHROPIC_API_KEY=...
```

Alternatively, copy `code/.env.example` to `code/.env` and fill in the value.
`code/.env` is git-ignored and must never be committed. **A real environment
variable always wins** over the `.env` file, so an `export` overrides whatever is
in the file — verified by test.

The key is read only through `config.require_api_key()`; no other module reads
the environment, nothing logs it, and no default or placeholder value exists. If
it is unset, `require_api_key()` raises with an explanatory message rather than
sending an unauthenticated request.

## Run Commands

All commands are run from the `code/` directory.

### Produce the submission

```bash
cd code
python3 main.py
```

Writes `output.csv` to the **repository root** (not `code/`) with exactly one
row per `message_id` in `dataset/messages.csv`, in that file's order.

| Flag | Effect |
|---|---|
| *(none)* | Resumable — rows already in `output.csv` are skipped, so a crash at row 90 costs nine rows rather than ninety |
| `--fresh` | Delete any existing `output.csv` and route all 110 rows from scratch |
| `--limit N` | Route only the first N rows; for smoke tests |

Model responses are cached on disk under `code/.cache/router/`, keyed on the
message id, the rendered feature block, the candidate pool, the model name, and
`PROMPT_VERSION`. A second run over unchanged inputs makes **zero** model calls.
Deleting that directory forces a full re-route.

### Validate before submitting

```bash
cd code
python3 validate_output.py
```

Exits 0 only if the header matches the required column order, the row count and
`message_id` set match `messages.csv` exactly, every `action` and `message_type`
is in vocabulary, every `confidence` parses as a float in [0, 1], every
`reason` is a catalogue string, and every evidence id either is `none` or exists
in `message_history.csv`.

### Other useful entry points

```bash
python3 check_dataset.py      # 17 dataset integrity assertions
python3 test_gates.py         # the precedence ladder, 28 cases
python3 test_schema.py        # output contract and tool schema, 16 cases
python3 test_degraded.py      # per-row failure handling, 7 cases
python3 test_vision.py        # image normalization vs a PIL oracle, 5 cases
python3 dryrun_gates.py       # rule-key reachability across the reading space
```

## Evaluation

```bash
cd code
python3 -m evaluation.main                                        # Config A
python3 -m evaluation.main --config b --out evaluation/metrics_b.json
python3 -m evaluation.write_report
```

Scoring runs the **full pipeline** over the 30 labeled rows in
`dataset/sample_messages.csv` and compares against their gold columns. Those
rows are a disjoint id namespace from `messages.csv`, so this is held-out
scoring for action accuracy — see Decision 1 for the one place it is not.

| Flag | Effect |
|---|---|
| `--config a` *(default)* | The full precedence ladder — the shipping configuration |
| `--config b` | The Decision 1 ablation: Layers 1 and 2 removed, `proposed_action` written straight through. A diagnostic, never a deployment candidate |
| `--out PATH` | Where to write metrics (default `evaluation/metrics.json`) |
| `--skip-judge` | Run every section except the LLM judge — **zero model calls** when the router cache is warm |

`write_report.py` renders `evaluation_report.md` from `metrics.json`,
`metrics_b.json`, and `run_stats.json`. Every number in the report is read from
those files at render time; none is transcribed by hand, so a stale figure is
fixed by re-running the scorer rather than by editing prose.

**The judge is not cached.** Each scoring run without `--skip-judge` costs 30
model calls, and its verdict counts move by 1–2 between invocations at
temperature 0. Use `--skip-judge` whenever the judge is not the thing being
measured.

## Configuration

Everything below lives in `config.py`; nothing is hardcoded elsewhere.

### Model

| Setting | Value | Note |
|---|---|---|
| `PRIMARY_MODEL` | `claude-sonnet-4-6` | One call per message |
| `TEMPERATURE` | `0.0` | |
| `MAX_TOKENS` | `1024` | The reading is a small structured object |
| `ENABLE_PROMPT_CACHING` | `True` | The system block is stable across all rows |
| `MAX_RETRIES` | `3` | One attempt plus three retries |
| `PROMPT_VERSION` | `1` | Part of the cache key — bumping it invalidates every cached reading |

The call is made with `tool_choice` pinned to a single tool, so the response
shape is guaranteed rather than parsed. **There is no JSON-repair path anywhere
in the codebase and none may be added.**

### Concurrency and backoff

| Setting | Value |
|---|---|
| `ROUTER_MAX_WORKERS` | `4` |
| `ROUTER_MIN_INTERVAL_S` | `0.15` (shared rate limiter, minimum gap between requests) |
| `ROUTER_BACKOFF_BASE_S` | `1.0` (exponential, with jitter, on 429 and 5xx) |
| `ROUTER_BACKOFF_MAX_S` | `30.0` |

### ASR

Local `faster-whisper`; no audio is sent to any API.

| Setting | Value | Note |
|---|---|---|
| `ASR_MODEL` | `base` | |
| `ASR_TASK` | `transcribe` | **Pinned deliberately.** `translate` would alter message content before routing; a default is not a contract |
| `ASR_DEVICE` / `ASR_COMPUTE_TYPE` | `cpu` / `int8` | Runs anywhere, no GPU required |
| `ASR_BEAM_SIZE` | `1` | Greedy — deterministic |
| `ASR_TEMPERATURE` | `0.0` | |
| `ASR_LANGUAGE` | `None` | Auto-detect |
| `ASR_VAD_FILTER` | `False` | |
| `ASR_CONDITION_ON_PREVIOUS_TEXT` | `False` | Each note transcribed independently |

Transcripts are cached in `code/cache/transcripts.json` alongside the exact
parameters that produced them, so a parameter change invalidates the cache
rather than silently serving stale text.

## Known limitations

Stated plainly; the evaluation report's "what was not tested" section carries
the full list with measured counts.

- **`message_type` is a raw passthrough.** One of the six output columns is the
  model's own string, enum-validated and nothing else. No join or ladder branch
  cross-checks it, and it is where the message_type accuracy shortfall comes
  from.
- **The pipeline is not reproducible run to run, and `temperature=0.0` does not
  make it so.** Re-routing all 110 rows from a cleared cache against
  byte-identical prompts changed **43 of 140 cached readings (30.7%)** on at
  least one field, and **21 (15.0%)** on at least one *structured* field:
  `media_summary` on 26 (free prose, which reaches a routing decision only
  through a substring test), `evidence_message_ids` on 8, `urgency` on 6,
  `content_risk` on 6, `message_type` on 3, and one each of
  `asks_user_for_action`, `promotional`, and `proposed_action`.
  The deterministic layers are genuinely deterministic:
  they absorbed most of that drift, and only **8 of 110 output rows** differed,
  of which **1 changed its `action`**. But the system as a whole is not
  bit-reproducible, so every count below is "as measured on the shipped run",
  not a fixed property. The on-disk router cache is what makes a *re-run*
  reproducible; delete it and the numbers move.
- **96 of the 110 shipped rows depend on at least one model field** for their
  action; only 14 would be decided identically with no model input at all. The
  ladder controls how the fields combine, not whether they are true. (These two
  counts are among the ones that move between runs — the previous run measured
  86 and 24.)
- **The safety layers are unexercised where it matters.** Layers 1 and 2 decide
  9 of the 30 labeled rows, and on 8 of those the model independently proposed
  the same action — so the corpus contains no row where the layer had to
  overrule a disagreeing model. `MUTE_IMPERSONATION_DOMAIN` fires on **zero**
  rows in both corpora, leaving the entire impersonation signature untested.
- **Only 30 labeled rows exist**, and they are disjoint from the 110 predicted.
  One row is 3.33 percentage points, so most deltas this project measured are
  inside the declared noise band. The 110 shipped rows have no ground truth at
  all.
- **Calibration is not held out.** The confidence bases were read off the same
  30 gold rows the calibration curve is computed on. Action accuracy is held
  out; the absolute confidence values are not.
- **The evidence column emits at most one id**, while the tool schema and pool
  enforcement still permit two — so the two-id path ships on no row. Which id
  survives is the model's own ordering; no component re-ranks.
- **Quiet hours and the muted-group mention exception have no gold coverage.**
  `DIGEST_QUIET_HOURS` binds on 0 of 110 rows, and no labeled row is both muted
  and mentioned.
- **10 of the 33 rule keys never fire** on the shipped corpus, so their reason
  strings have never been emitted against a real row.
- **The degraded path has never run for real.** Zero rows degraded in any run;
  it is covered by `test_degraded` with injected failures and by nothing else.
- **Voice and image rows are not scored as a stratum**, so a failure confined to
  one modality would not show up in any reported number.
- **`transcription.transcription_failed()` has no caller**, and `unresolved` is
  named in a docstring but never written. The live failure record is
  `degraded_rows` in `run_stats.json`.
