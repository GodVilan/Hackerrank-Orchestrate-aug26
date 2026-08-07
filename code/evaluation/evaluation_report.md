# Evaluation Report — Message Notification Router

Model `claude-sonnet-4-6` · prompt version 1 · scored on the 30 labeled rows of `dataset/sample_messages.csv`.

Those rows are a **disjoint id namespace** from `messages.csv` — `sample_msg_001`…`sample_msg_053` against `msg_001`…`msg_110`, zero overlap, and the labeled range is non-contiguous (30 rows across 53 id slots). They are extra labeled examples, not a labeled subset of the rows the system predicts.

**What is held out, and what is not.** An earlier draft of this report claimed that nothing was fitted to the labeled rows. That was not accurate, and the accurate version is narrower:

| quantity | where | fitted to the labeled rows? |
|---|---|---|
| `CONFIDENCE_BASE`, `CONFIDENCE_STEP` | `schema.py` | **Yes** — read directly off the 30 gold confidence values |
| `T_FWD_MESSAGE` | `config.py` | **Partly** — bounded above by `sample_msg_014`, which gold labels `mute` and which no other branch catches |
| `T_REPORT`, `T_DISMISS`, `T_FWD_SENDER_MEAN` | `config.py` | No — chosen from empty bands in the 110-row distributions |
| `T_REPORT_PER_1K`, impersonation age bounds | `config.py` | No — same method, 110-row distributions |
| Ladder branch conditions and their ordering | `gates.py` | No — not tuned per-row against gold |

The consequence is specific: **action accuracy below is held-out, and the calibration section is not.** The confidence bases were derived from the same 30 rows the calibration curve is computed on, so that curve measures whether the tier logic tracks correctness *given* bases fitted to this set — it is not evidence that the absolute confidence values would be calibrated on unseen rows. Tier 1, Tier 2, and Tier 2b carry no such circularity. The `T_FWD_MESSAGE` exception is real but narrow: gold constrains it from above only, and any value in 3…11 behaves identically on this corpus.

Every number below is read from `evaluation/metrics.json` and `evaluation/metrics_b.json` at render time by `evaluation/write_report.py`. None of it is transcribed by hand.

|  | Config A | Config B |
|---|---|---|
| routing rows | 30 | 30 |
| routing cache hits | 0/30 (0.0%) | 30/30 (100.0%) |
| live routing model calls | 30 | 0 |
| degraded rows | 0 | 0 |

**The control for this entire comparison, stated mechanically.** The router cache key is `sha256(message_id + rendered facts + candidate pool ids + model + PROMPT_VERSION + media)` — it takes **no configuration argument**, so a given row has one cache entry regardless of which config is scoring it. Config A ran first against a cleared cache and made 30 live calls, writing those entries; Config B then ran and made 0 live calls, reading 30 of them back. Verified from file timestamps: every labeled row's cache entry was written inside Config A's run window, and Config B read those same files.

So the two configs consumed **byte-identical model payloads** — not merely equivalent readings from two invocations, but literally the same JSON on disk. That matters because, as the reproducibility section below measures, two live invocations of this pipeline do *not* agree with each other. Had Config B re-routed instead of reading the cache, part of any A-vs-B difference would have been sampling noise rather than the ladder. It read the cache, so every difference between the two columns is the ladder and nothing else.

## The two configurations

**Config A — full precedence ladder (shipping configuration).** One model call per row returns a structured semantic reading and nothing else: no action, no reason, no confidence. That reading feeds a four-layer deterministic precedence ladder — safety, hard user state, personalization, content substance — in which each layer overrides everything below it and a branch that fires returns immediately. The action comes from the ladder, the reason string is looked up from a fixed catalogue by rule key, and the confidence is band floor plus one step per corroborating signal. Content reasoning cannot invert an integrity or consent decision, because content reasoning is never reached when Layer 1 or Layer 2 fires. This is the shipping configuration.

**Config B — the Decision 1 ablation. A diagnostic, never a deployment candidate:** it deletes the safety and consent layers, so it exists to measure what those layers contribute, not to compete for shipping. Recorded in Design Decision 1 before the ladder was built. Layers 1 and 2 are removed and the model's own `proposed_action` is written straight through as the row's action, so the model's verdict reaches the output unfiltered — including on exactly the rows the safety and consent layers would have caught. It is a flag on the same pipeline (`main.process(ablate_layers_1_2=True)`), not a forked copy, so loading, features, media handling, pooling, and evidence enforcement are provably identical between the two. The reason column still needs a rule key; when the shortened ladder agrees with the model its key is kept, and when it does not the key comes from the ladder's own sub-rule selectors for that action, so no new mapping is invented for the ablation.

## Reading these numbers: n = 30

There are 30 labeled rows. **One row is 3.33 percentage points**, and the noise band is 10pp — 3 rows. A delta is reported as **within noise** when it is strictly below the band; a delta exactly equal to the band is reported as **on the band edge**, which the rule does not resolve; only a delta strictly above it is a result.

Verdicts are computed in whole rows rather than percentage points, and from one function, so that the same difference cannot be reported as noise in one table and as a result in another. The rule is applied symmetrically: a delta that flatters a design decision is discounted exactly as hard as one that embarrasses it.

Applied to this project's own changes:

* **The fix pass shipped two changes, not three.** The verified-brand exemption on `MUTE_SCAM_FIRST_CONTACT` and the one-evidence-id emission cap are in the shipping configuration. A third change — hoisting the business-notify branch out of the Layer 4 urgency gate into Layer 3 — was applied and then **reverted after review**, and does not ship. Earlier revisions of this report described it as shipped and defended it on its merits; that text was written while it was in the tree and is corrected here.
* **Why the hoist was reverted.** It made `DIGEST_PROMO_OPTED_IN` structurally unreachable for every verified business with a recorded relationship and recent activity, because the hoisted branch returned before Layer 4's terminal digest mapping could be consulted. On the shipped corpus it converted 7 rows from `digest` to `notify`, including a cinema feedback survey and an opted-in promotion the model had itself read as `suspicious`. It also made `DIGEST_QUIET_HOURS` bind on a row for the first time, falsifying a documented design claim. Against that it was worth **one row** on the labeled set — 3.33pp, within noise. A within-noise gain is not a reason to keep a branch that reorders the ladder.
* **The verified-brand exemption**, which did ship, releases **zero rows** across the 110 shipped rows and 2 of the 30 labeled rows, and both of those still land on `mute` via `MUTE_UNSOLICITED_BUSINESS`. Its measured effect on action accuracy is **zero rows in both corpora**; it changes the reason string only.
* **The evidence cap**, which also shipped, is defended on grounds other than the action score, which it cannot move: gold cites exactly one id on 25 of the 30 labeled rows, and capping emission raised evidence F1 and roughly doubled exact-set-match.

## Tier 1 — field accuracy

| metric | Config A | Config B | B − A | rows | verdict |
|---|---|---|---|---|---|
| action accuracy | 0.8333 | 0.9000 | +6.67pp | +2 | within noise |
| macro-F1 (action) | 0.8306 | 0.9034 | +7.28pp | +2 | within noise |
| message_type accuracy | 0.7667 | 0.7667 | +0.00pp | +0 | no difference |
| both fields correct | 0.7000 | 0.6667 | -3.33pp | -1 | within noise |

**Action accuracy and macro-F1 carry the same verdict because they are the same 2 rows.** Macro-F1 is a re-weighting of the identical set of correct and incorrect action calls, so reporting it as a separately significant delta would be double-counting one difference. An earlier revision did exactly that — it derived each verdict from its own percentage-point gap, which put macro-F1 at 10.10pp and action accuracy at 10.00pp and therefore on opposite sides of the band. Both now read from one computed verdict in row units.

`message_type` is copied from the model's reading in both configs and is untouched by the ladder, so its accuracy is identical by construction — the ablation cannot move it.

### Per-action precision / recall / F1

| action | support | A prec | A recall | A F1 | B prec | B recall | B F1 |
|---|---|---|---|---|---|---|---|
| notify | 9 | 1.000 | 0.667 | 0.800 | 1.000 | 0.889 | 0.941 |
| digest | 11 | 0.750 | 0.818 | 0.783 | 0.786 | 1.000 | 0.880 |
| mute | 10 | 0.833 | 1.000 | 0.909 | 1.000 | 0.800 | 0.889 |

### Confusion matrices

**Config A**

| gold \ pred | notify | digest | mute | total |
|---|---|---|---|---|
| **notify** | 6 | 3 | 0 | 9 |
| **digest** | 0 | 9 | 2 | 11 |
| **mute** | 0 | 0 | 10 | 10 |
| **total** | 6 | 12 | 12 | 30 |

**Config B**

| gold \ pred | notify | digest | mute | total |
|---|---|---|---|---|
| **notify** | 8 | 1 | 0 | 9 |
| **digest** | 0 | 11 | 0 | 11 |
| **mute** | 0 | 2 | 8 | 10 |
| **total** | 8 | 14 | 8 | 30 |

### Misses

**Config A — 5 of 30**

| message_id | gold | predicted | rule_key | confidence |
|---|---|---|---|---|
| sample_msg_004 | notify | digest | `DIGEST_BIZ_LEGIT` | 0.84 |
| sample_msg_005 | notify | digest | `DIGEST_BIZ_LEGIT` | 0.78 |
| sample_msg_046 | notify | digest | `DIGEST_EVENT_FUTURE` | 0.78 |
| sample_msg_048 | digest | mute | `MUTE_UNSOLICITED_BUSINESS` | 0.81 |
| sample_msg_049 | digest | mute | `MUTE_SCAM_FIRST_CONTACT` | 0.81 |

**Config B — 3 of 30**

| message_id | gold | predicted | rule_key | confidence |
|---|---|---|---|---|
| sample_msg_004 | notify | digest | `DIGEST_BIZ_LEGIT` | 0.84 |
| sample_msg_014 | mute | digest | `DIGEST_GROUP_INFO` | 0.82 |
| sample_msg_047 | mute | digest | `DIGEST_UNKNOWN_SENDER_BENIGN` | 0.82 |

### Where the two configs disagree

The two configs disagree on 6 of the 30 rows, listed below. The layer column is the ladder layer that produced Config A's verdict.

**1 further row is wrong in both configs** — `sample_msg_004` — and so does **not** appear in the table: both configs predict the same action there, which is why it is not a disagreement, and both are wrong. The miss sets therefore overlap, and the ablation is not a clean partition of the errors — a row can be missed by neither config's distinctive behaviour and still be missed.

| message_id | gold | Config A | layer | A rule_key | Config B | B rule_key |
|---|---|---|---|---|---|---|
| sample_msg_005 | notify | digest ✗ | 4 | `DIGEST_BIZ_LEGIT` | notify | `NOTIFY_BIZ_MATCHES_BOOKING` |
| sample_msg_014 | mute | mute | 3 | `MUTE_FORWARD_PATTERN` | digest ✗ | `DIGEST_GROUP_INFO` |
| sample_msg_046 | notify | digest ✗ | 4 | `DIGEST_EVENT_FUTURE` | notify | `NOTIFY_SCHOOL_OPERATIONAL` |
| sample_msg_047 | mute | mute | 3 | `MUTE_UNSOLICITED_BUSINESS` | digest ✗ | `DIGEST_UNKNOWN_SENDER_BENIGN` |
| sample_msg_048 | digest | mute ✗ | 3 | `MUTE_UNSOLICITED_BUSINESS` | digest | `DIGEST_UNKNOWN_SENDER_BENIGN` |
| sample_msg_049 | digest | mute ✗ | 1 | `MUTE_SCAM_FIRST_CONTACT` | digest | `DIGEST_UNKNOWN_SENDER_BENIGN` |

### What Layers 1 and 2 actually contributed

This is the question the ablation was built to answer, and the aggregate accuracy comparison does not answer it. Below are every row Config A decided inside Layer 1 or Layer 2 — the layers Config B removes — with what the model proposed for the same row.

| message_id | layer | rule_key | gold | Config A | Config B | model agreed | A correct |
|---|---|---|---|---|---|---|---|
| sample_msg_013 | 2 | `MUTE_MUTED_GROUP` | mute | mute | mute | yes | yes |
| sample_msg_015 | 2 | `MUTE_OPTED_OUT_MARKETING` | mute | mute | mute | yes | yes |
| sample_msg_019 | 1 | `MUTE_SCAM_OTP` | mute | mute | mute | yes | yes |
| sample_msg_020 | 1 | `MUTE_SCAM_OTP` | mute | mute | mute | yes | yes |
| sample_msg_043 | 1 | `MUTE_SCAM_FIRST_CONTACT` | mute | mute | mute | yes | yes |
| sample_msg_045 | 2 | `MUTE_MUTED_GROUP` | mute | mute | mute | yes | yes |
| sample_msg_049 | 1 | `MUTE_SCAM_FIRST_CONTACT` | digest | mute | digest | **no** | **no** |
| sample_msg_052 | 1 | `MUTE_SCAM_OTP` | mute | mute | mute | yes | yes |
| sample_msg_053 | 1 | `MUTE_INJECTION_ATTEMPT` | mute | mute | mute | yes | yes |

**Layers 1 and 2 decided 9 of the 30 labeled rows. On 8 of those 9, the model independently proposed the identical action, so removing the layers changes nothing. The ablation loses 0 rows to their removal.**

The one row where Layer 1 and the model part company is `sample_msg_049` — and there **the layer is wrong and the model is right**: gold says `digest`, Layer 1's first-contact branch says `mute`. On these 30 rows, the only independent contribution Layers 1 and 2 make to the outcome is an error.

Config B misses 3 rows in total, of which 1 (`sample_msg_004`) is also missed by Config A. That leaves **2 losses relative to Config A** — the rows the ablation gets wrong that the full ladder gets right — which is the number this section is about. Those 2 are **not** attributable to the removal of the safety and consent layers: Config A decides them at Layer 3 (2) — none at Layer 1 or Layer 2:

| row | gold | Config A rule_key | layer | Config B |
|---|---|---|---|---|
| sample_msg_014 | mute | `MUTE_FORWARD_PATTERN` | 3 | digest |
| sample_msg_047 | mute | `MUTE_UNSOLICITED_BUSINESS` | 3 | digest |

Each is a case where `proposed_action`, written straight through on every row, overrides a personalization branch that the gold label agrees with. That is a property of how the ablation is defined, not evidence that Layers 1 and 2 are doing work.

The 5 rows Config A gets wrong — `sample_msg_004`, `sample_msg_005`, `sample_msg_046`, `sample_msg_048`, `sample_msg_049` — are rows where the ladder overrode a model reading that matched gold, or where both agreed and both missed.

This is the central negative result of the report, and it is stated plainly because it cuts against the design: **on this corpus the safety ladder is redundant with the model's own judgement everywhere it fires correctly, and wrong on the single row where it disagrees.** Whatever case exists for Layers 1 and 2, these 30 rows do not make it — see §"Which configuration ships" for why that does not make Config B deployable.

## Tier 2 — evidence grounding

| metric | Config A | Config B |
|---|---|---|
| emitted ids | 28 | 28 |
| ids drawn from that row's candidate pool | 28 (1.0000) | 28 (1.0000) |
| pool violations | 0 | 0 |
| tp / fp / fn | 16 / 12 / 15 | 16 / 12 / 15 |
| precision | 0.5714 | 0.5714 |
| recall | 0.5161 | 0.5161 |
| F1 | 0.5424 | 0.5424 |
| exact set match | 0.4667 | 0.4667 |
| gold says `none` | 2 rows, agreement 0.500 | 2 rows, agreement 0.500 |

**The in-pool invariant holds at 1.0000 with 0 violations in both configs.** Every id that reaches `output.csv` was present in that row's deterministically built candidate pool. This is enforced in code by `evidence.enforce_evidence`, not requested in the prompt — the model cannot invent a message id that survives to the output even if it tries.

**`evidence_violations: 0` does not mean no evidence was discarded, and should not be quoted as if it did.** It counts only ids that were never admissible — out-of-pool, duplicate, or over-length. Valid in-pool ids discarded by the emission cap are a separate quantity, counted separately in `run_stats.json` since this run:

| `run_stats.json` key | value | what it counts |
|---|---|---|
| `evidence_violations` | 0 | ids the model returned that were never admissible |
| `evidence_truncated` | 83 | valid in-pool ids kept by enforcement, then dropped by the cap |
| `evidence_truncated_rows` | 83 | rows on which that happened |
| `emitted_evidence_ids_cap` | 1 | `finalize.EMITTED_EVIDENCE_IDS`, the cap itself |

So across the 110 shipped rows the system discarded a valid, model-selected, in-pool evidence id on 83 of them. Which id survives is the model's own ordering — `format_evidence` keeps the first returned and no component re-ranks them.

Evidence selection is identical across the two configs, because it depends on the model reading and the candidate pool and not on the ladder. The identical Tier 2 block is a control: it confirms the two runs differ only where they are supposed to.

## Tier 2b — reason quality

### Rule-key stability

Do rows sharing a gold `(action, message_type)` receive a single rule key? Instability is not automatically a defect — two rows can share a gold label for different reasons — but a group that fragments is where to look for an incoherent branch.

|  | Config A | Config B |
|---|---|---|
| groups with more than one row | 7 | 7 |
| of those, a single rule key | 1 | 1 |
| unstable groups | `digest/business_update`, `digest/promotion`, `mute/promotion`, `mute/scam`, `notify/event`, `notify/urgent` | `digest/business_update`, `digest/promotion`, `mute/promotion`, `mute/scam`, `notify/event`, `notify/urgent` |

### LLM-as-judge

Rubric: `same_rule` — both explanations cite the same underlying reason; `compatible` — different reasons, both plausible, pointing the same way; `contradictory` — they cannot both be the operative reason, or they imply opposite routing. The judge compares two strings and is explicitly told neither is authoritative about the message.

| verdict | Config A | Config B |
|---|---|---|
| `same_rule` | 13 (43.3%) | 15 (50.0%) |
| `compatible` | 13 (43.3%) | 9 (30.0%) |
| `contradictory` | 4 (13.3%) | 6 (20.0%) |

**Judge variance — read this before comparing the two columns.** The judge is not cached and is not reproducible. Identical pipeline output, scored at temperature 0, returned 13 `same_rule` on one invocation and 15 on another. A judge delta of 1–2 verdicts is therefore not signal, and no claim in this report rests on one. Each config's judge was run exactly once, and Config A's verdicts are carried over verbatim from the invocation that produced them rather than re-measured for this report.

This also means the reason column must not be tuned against the judge: the measurement is noisier than the effects being chased.

## Tier 3 — consistency on shared media

Identical image bytes reaching different recipients. If every row in a cluster gets the same action, personalization is not working. If actions differ, the differing user features should be the explanation.

| media | rows | A distinct actions | B distinct actions | differing user features |
|---|---|---|---|---|
| `img_003` | 2 | digest, mute | digest, mute | 6 |
| `img_008` | 5 | digest, mute, notify | digest, mute, notify | 10 |
| `img_010` | 3 | digest, mute | digest, mute | 8 |

Flags raised: **Config A 0, Config B 0**. No cluster collapsed to a single action despite differing features, and no cluster produced differing actions from identical features.

**Caveat on this section, stated rather than hidden.** Each cluster mixes two provenances. Rows carrying a `sample_msg_` id were routed live by the config being scored. Rows carrying a `msg_` id are read from `output.csv` — **the current one, written by the clean post-revert run that produced every other shipped figure in this report**, so they are not stale. They are still not evidence about Config B: `output.csv` is a Config A artifact, so those cells are identical in both columns by construction. Only the labeled rows in each cluster are config-sensitive:

| media | message_id | recipient | gold | Config A | Config B |
|---|---|---|---|---|---|
| `img_008` | sample_msg_044 | u_032 | digest | digest | digest |
| `img_008` | sample_msg_045 | u_033 | mute | mute | mute |
| `img_010` | sample_msg_047 | u_007 | mute | mute | digest |

## Calibration — Config A

Confidence is `BASE[action] + 0.02 × tier`, where the tier counts independent corroborating signals capped at 3. The bases are read off the labeled rows rather than invented, and reproduce all 30 gold confidence values exactly.

| confidence | n | correct | accuracy |
|---|---|---|---|
| 0.78 | 2 | 0 | 0.000 |
| 0.81 | 4 | 2 | 0.500 |
| 0.82 | 4 | 4 | 1.000 |
| 0.84 | 6 | 5 | 0.833 |
| 0.87 | 9 | 9 | 1.000 |
| 0.91 | 5 | 5 | 1.000 |

* Low band (≤ 0.84): n = 16, accuracy 0.688
* High band (> 0.84): n = 14, accuracy 1.000
* **Band separation +0.312.** **Strictly monotonic across every value: False.**

The curve is not monotonic, and the report does not claim it is. Accuracy rises across the bands but dips at 1 value(s) in between:

* **0.84** — n = 6, accuracy 0.833, below 0.82.

**The mechanism is band-floor rows, and it is structural.** Each action has its own floor — `digest` 0.78, `mute` 0.81, `notify` 0.85 — and the bands overlap, so a poorly corroborated row of one action can outrank a well corroborated row of another. A row sitting exactly on its floor is a **tier 0** row, and on the healthy path a row reaches tier 0 exactly one way: the tier is pinned to 0, overriding every other signal, when the model's `proposed_action` disagrees with the action the ladder chose. Those rows are the least trustworthy in the file and the confidence column says so — the dip is the disagreement penalty working, not the tier logic failing.

The floor buckets in this run, which are exactly the tier-0 rows:

| confidence | action band | n | accuracy |
|---|---|---|---|
| 0.78 | `digest` | 2 | 0.000 |
| 0.81 | `mute` | 4 | 0.500 |

**How well does the confidence column isolate the errors?** Exactly 6 of the 30 rows carry tier 0, every one of them via `disagreement_override`, and that set is identical to the set of rows on which the two configs disagree.

|  | count | inside the tier-0 set? |
|---|---|---|
| tier 0 (`disagreement_override`) | 6 | — (it is the set) |
| configs disagree | 6 | identical set |
| Config A errors | 5 | **1 outside**: `sample_msg_004` |
| Config B errors | 3 | **1 outside**: `sample_msg_004` |

**The isolation is good but not total, and the exceptions matter more than the rule.** An earlier revision of this report claimed the tier-0 set captured every error in both configs. That was true of the configuration measured then and is **no longer true**: `sample_msg_004` sits outside it.

| row | gold | Config A | confidence | tier | tier signals |
|---|---|---|---|---|---|
| sample_msg_004 | notify | digest | 0.84 | 3 | `counterparty_history`, `evidence_selected`, `model_agrees` |

This is the failure mode the disagreement signal structurally cannot catch: the model and the ladder **agreed**, so the tier stayed high, and they were both wrong in the same direction. A confidently wrong row is worse than an uncertain one, because nothing downstream flags it for review. It is also the direct cost of reverting the Layer 3 business-notify hoist — that branch was what pushed this row to `notify`, and removing it traded a structural ordering bug and two questionable notifies that would have shipped for one high-confidence miss here. Those notifies never reached any published `output.csv`: the pre-revert file was deleted before the clean run, so the hoist's effect on the shipped corpus is counterfactual, not historical.

*Inference, not a demonstrated property — n = 30.* Even the part that does hold is one observation on a small sample, not an invariant. No mechanism guarantees that a row the model and the ladder agree on is correct; the agreement simply removes the one signal that would have flagged it. Read the tier-0 band as a useful triage filter that catches 4 of Config A's 5 errors without consulting a label — not as a guarantee that the rows above it are safe.

### Config B calibration, for contrast

| confidence | n | correct | accuracy |
|---|---|---|---|
| 0.80 | 1 | 1 | 1.000 |
| 0.82 | 7 | 5 | 0.714 |
| 0.84 | 6 | 5 | 0.833 |
| 0.85 | 1 | 1 | 1.000 |
| 0.87 | 8 | 8 | 1.000 |
| 0.91 | 7 | 7 | 1.000 |

Band separation +0.214, strictly monotonic across every value: **False**.

**Config B's curve is not monotonic either, so it cannot be described as buying a tidier curve** — an earlier revision of this report said exactly that, and it was wrong. The substantive point survives unchanged and does not depend on monotonicity: with the action defined as `proposed_action`, the model can never disagree with the ladder, so `disagreement_override` cannot fire on any row in Config B. The tier-0 band that isolates Config A's error set does not exist here — not because B's calibration is worse in shape, but because the signal that produces it has been defined out of existence.

Its separation is -9.82pp against Config A's, and it does not isolate the error set either way.

## Which configuration ships

**Correction first.** An earlier revision of this report opened this section with "decided on action accuracy alone" and treated the two configs as candidates in a contest. That framing was wrong. Config B removes Layers 1 and 2 — the scam, impersonation, router-injection, group-mute and promotion-opt-out gates — so shipping it would mean shipping a router with no safety or consent gate at all. It was never a deployment candidate, and describing the comparison as a contest misdescribed what the ablation was for. The ablation is a **diagnostic**: it measures what those two layers contribute, and that is all it was ever able to decide.

This correction is not a response to the result. It does not retract, soften, or reweight a single measured finding below — the gap, the boundary analysis, and the negative result about the ladder all stand exactly as measured, and Config B did score higher. What changes is only the claim that the score was deciding what ships.

|  | Config A | Config B | B − A |
|---|---|---|---|
| action accuracy | 0.8333 (25/30) | 0.9000 (27/30) | +6.67pp |

**Config B scored higher.** The gap is 2 rows, +6.67pp, against a noise band of 10pp.

Verdict, computed in row units from the single rule stated above: within noise.

**Config A ships — and it would have shipped at any score.** Not because it measured better on action accuracy, which it did not, and not because the ablation failed to clear a band. Config B is disqualified by what it is: a router with Layers 1 and 2 deleted does not suppress scams, impersonation, router-injection attempts, muted groups, or promotions the user has opted out of, whatever it scores on 30 labeled rows. No accuracy figure would have made it deployable, and it is worth being explicit that a higher number for B would not have changed this paragraph.

The measured result is a separate matter and is not diminished by that. Design Decision 1 predicted the ladder would earn its place empirically; on these rows it does not, and that stands as a negative result rather than something the disqualification explains away. What the ablation establishes is how much the safety and consent layers contribute on this corpus — the answer is nothing measurable — not which configuration to deploy.

Two things follow that are worth stating in the same breath, because omitting either would misrepresent the result.

**The ladder is not validated by this measurement.** Layers 1 and 2 decided 9 rows and the model matched them on 8; the ablation loses 0 rows to their removal, and the single row where Layer 1 acted independently is a row it got wrong. The argument for the ladder therefore rests on its structural property — that content reasoning cannot invert an integrity or consent decision, because it is never reached — and not on any number in this report. That property is real and testable by reading the code. It is simply not what these 30 rows measure.

**Nor is the ladder refuted.** The corpus does not contain a row where the model proposes `notify` for something Layer 1 correctly suppresses, which is the failure mode the layer exists to prevent. A safety gate that never had to fire against a disagreeing model has not been shown unnecessary; it has been shown untested. Reading '0 rows lost' as 'Layers 1 and 2 are dead weight' would be the same error in the opposite direction as reading '+6.67pp' as 'the model beats the ladder'.

What the measurement does **not** license anyone to conclude:

* That the ladder beats the model. On these 30 rows it does not, by this metric.
* That an unfiltered model output should ship. The gap is real — 2 rows, +6.67pp, within noise — and it still does not license that conclusion, because the configuration that produced it has no safety or consent gate. A better score from a router that cannot suppress a credential-harvesting message is not an argument for shipping it.
* That the safety layers earn their place. On this corpus they do not earn it empirically; the case for them is structural and is argued as such.
* That the two error classes cost the same. An unsuppressed scam or a notification pushed into a muted group is a worse outcome for a user than a delivery update that arrives in the digest instead of as an interrupt. This asymmetry is a design judgement, not a finding — the report has no evidence with which to quantify it, and does not pretend otherwise.
* That either number would survive a different 30 rows. Neither would, reliably.

## What was not tested

Concretely, with no coverage claimed:

* **The 30 labeled rows are the entire ground truth.** The 110 rows in `messages.csv` that the system actually predicts have **no labels at all** and are disjoint from the labeled set. Nothing in this report is a measurement of output quality on the shipped rows.
* **Layer 1 against a disagreeing model.** No labeled row exists where the model proposes `notify` or `digest` for a message Layer 1 correctly mutes. Every correct Layer 1 suppression in the corpus is one the model also proposed, so the layer's entire reason for existing — being the thing that stops a persuasive message from talking its way past the gate — is exercised by no scored row.
* **Quiet hours.** `DIGEST_QUIET_HOURS` binds on **0 of the 110 shipped rows** and on no labeled row, re-measured after the Layer 3 business-notify hoist was reverted. The 8 `dnd_active` rows and the 5 direct-mention rows are disjoint, and the surviving mention-independent notify branches all require `urgency == "high"`, which the downgrade excludes by design. So the modifier is validated by unit test and hand inspection only, never against a label — and while the hoist was in place it did bind on one row, so this figure is a property of the whole ladder rather than of the branch.
* **`MUTE_IMPERSONATION_DOMAIN`.** Fires on **zero rows in both corpora**. The entire impersonation signature — unverified, non-empty official domain, domain mismatch, young account or young sender domain, and report rate above threshold — is a Layer 1 safety branch that no row the system has ever processed has exercised. Its thresholds are read off an empty band in the observed distribution, which fixes their behaviour on this corpus and says nothing about a hidden set. It is unexercised code on the safety path.
* **10 of the 33 rule keys never fire** on the shipped corpus: `DIGEST_GREETING`, `DIGEST_INSUFFICIENT_SIGNAL`, `DIGEST_MATCHES_INTEREST`, `DIGEST_OFFER_RELEVANT`, `DIGEST_QUIET_HOURS`, `DIGEST_UNKNOWN_SENDER_BENIGN`, `MUTE_IMPERSONATION_DOMAIN`, `MUTE_SCAM_FAKE_SUPPORT`, `NOTIFY_DIRECT_REQUEST`, `NOTIFY_PAYMENT_LEGIT`. `DIGEST_INSUFFICIENT_SIGNAL` at zero is the good case — it means no row degraded. The rest are branches whose reason strings ship in the catalogue and have never been emitted, so their wording has never been checked against a real row.
* **The muted-group mention exception.** No labeled row is both in a muted group and a direct `@mention` of the recipient, so `NOTIFY_MENTION_IN_MUTED_GROUP` has zero gold coverage. The two rows in the shipped corpus that reach it are unlabeled.
* **Degraded rows.** Zero rows degraded during scoring in either config, so the fallback path — neutral reading, tier pinned to 0, `DIGEST_INSUFFICIENT_SIGNAL` — is covered by `test_degraded` with injected failures and by nothing else. It has never run against a real model failure on a labeled row.
* **The reason column, in any calibrated sense.** The judge is not cached, varies by 1–2 verdicts between invocations at temperature 0, and is the only instrument pointed at reason quality. Rule-key stability is a structural check, not a correctness one.
* **Evidence beyond the first id.** The pipeline emits at most one evidence id, while the tool schema and pool enforcement still permit two. The two-id path is exercised by no scored row.
* **`muted_after_message`.** Set on 134 of 412 event rows and read by nothing: absent from the features, the rendered candidate pool, and every ladder branch. A per-sender 'muted after a previous message' rate is a plausible personalization signal that this system does not use.
* **Voice and image rows as a stratum.** Transcription and vision are unit-tested and the ASR task is pinned to `transcribe` rather than `translate`, but neither modality is scored separately, so a systematic failure confined to voice notes or to image posters would not show up in any number above.
* **Tier 3 across configurations.** Only the labeled rows in each media cluster were re-run per config. The `msg_` rows are read from the current post-revert `output.csv` — they are not stale — but `output.csv` is a Config A artifact by definition, so those rows carry no information about Config B and the cluster comparison is single-config for them.
* **Cross-run stability — no longer unmeasured.** See the section below; it is now the largest known threat to every number in this report.

## Reproducibility

An earlier revision listed cross-run stability under "what was not tested". It has since been measured, and the result is the single most important caveat in this report.

**The experiment.** router cache cleared; all 110 rows re-routed against byte-identical prompts at temperature 0.0. The previous run's cache was preserved, so the two sets of readings could be compared entry by entry — same cache keys, same prompts, same model, same `temperature=0.0`.

| measure | value |
|---|---|
| readings compared | 140 |
| readings that changed on **any** field | 43 (30.7%) |
| readings that changed on a **structured** field | 21 (15.0%) |
| `output.csv` rows changed | 8 of 110 |
| rows whose **action** changed | 1 |
| Config A action accuracy, before → after | 0.8000 → 0.8333 |

Per-field breakdown of what moved:

| field | readings changed | structured? |
|---|---|---|
| `media_summary` | 26 | no — free text |
| `evidence_message_ids` | 8 | yes |
| `urgency` | 6 | yes |
| `content_risk` | 6 | yes |
| `message_type` | 3 | yes |
| `asks_user_for_action` | 1 | yes |
| `promotional` | 1 | yes |
| `proposed_action` | 1 | yes |

**The two headline percentages measure different things and both are reported on purpose.** `media_summary` is free prose and accounts for most of the gap between 30.7% and 15.0%; it reaches a routing decision only through a substring test in the scam sub-rule, so it rarely changes an outcome. The structured figure is the one that bears on routing. An earlier revision of this analysis quoted only the structured number as though it were the total — it is not.

**`temperature=0.0` did not make this pipeline reproducible.** That is the plain finding. Identical prompts, identical model, identical decoding parameter, and roughly a third of the readings came back different. Determinism is a property the deterministic layers have; it is not a property of the system end to end.

**What the ladder absorbed.** Of those changed readings, only 8 reached `output.csv` at all (`msg_012`, `msg_031`, `msg_041`, `msg_057`, `msg_060`, `msg_089`, `msg_104`, `msg_106`), and only 1 changed an action:

| row | action before | action after |
|---|---|---|
| `msg_089` | mute | notify |

That is the deterministic ladder doing exactly what Decision 1 built it to do — absorbing model variance that would otherwise reach the user. It is the strongest evidence in this report for the ladder, and it is worth noting that it is **not** the evidence Decision 1 predicted: the ablation was supposed to demonstrate the ladder's value, and did not.

**What makes a re-run stable is the on-disk router cache, not the temperature setting.** Re-running with a warm cache reproduces `output.csv` byte for byte and costs nothing. Clearing the cache re-rolls every reading. Any figure in this report is therefore conditional on the cache that produced it, and a grader who deletes `code/.cache/router/` and re-runs should expect the numbers to move by roughly this much.

**This also moved a headline result.** Config A's action accuracy went from 0.8000 to 0.8333 purely from re-routing — no code change. That shifted the A-vs-B gap from three rows to 2, and with it the verdict from the band edge to within noise. A one-row swing in the underlying model output changed which side of a pre-declared threshold the comparison landed on, which is a concrete demonstration of why a 30-row sample cannot settle it.

## Operational analysis

Every figure here is read from `run_stats.json`, which the 110-row production run writes. It describes **that run only** — the scoring runs above are a separate 30-row pass and are reported in their own sections.

| metric | value |
|---|---|
| rows routed | 110 |
| **live model calls** | **110** |
| cache hits | 0 |
| cache hit rate | 0.0% |
| input tokens | 128,220 |
| output tokens | 23,317 |
| cache-read tokens | 159,467 |
| cache-write tokens | 1,463 |
| retries | 0 |
| errors | 0 |
| degraded rows | 0 |
| summed request latency | 462.71 s |
| wall-clock runtime | 463.0 s |
| concurrency (`ROUTER_MAX_WORKERS`) | 4 |
| rate limiter (`ROUTER_MIN_INTERVAL_S`) | 0.15 s between requests |
| backoff (`ROUTER_BACKOFF_BASE_S` → `_MAX_S`) | 1.0 s exponential with jitter, capped at 30.0 s, on 429 and 5xx |
| `MAX_RETRIES` | 3 (one attempt plus three retries) |

Summed request latency exceeds wall-clock runtime whenever the four workers overlap, and falls below it when rows are served from cache without a request at all — the two are not the same measurement and are reported separately rather than reconciled.

### Cost

**Pricing assumption, stated explicitly:** `claude-sonnet-4-6` at **$3.00 per million input tokens** and **$15.00 per million output tokens** — Anthropic first-party API list price. Prompt-cache reads bill at 0.1× the input rate and cache writes at 1.25×. These rates are hardcoded in `write_report.py`; they are not read from the API, so a price change makes the figures below stale and they should be recomputed rather than trusted.

| component | tokens | rate | cost |
|---|---|---|---|
| input | 128,220 | $3.00/M | $0.3847 |
| output | 23,317 | $15.00/M | $0.3498 |
| cache read | 159,467 | $0.30/M | $0.0478 |
| cache write | 1,463 | $3.75/M | $0.0055 |
| **total** |  |  | **$0.7877** |

That is **$0.0072 per row** across 110 rows.

**Every one of those 110 rows was routed live** — the cache was cleared before this run, so nothing was served from disk. The per-row figure is therefore the *full* uncached cost of routing this corpus from scratch, not an understatement of it. A re-run against the warm cache costs $0.00 in API spend.

Two costs are **not** in the table and should not be inferred from it: the evaluation judge (30 uncached calls per scoring run, per config) and ASR, which runs locally via `faster-whisper` and costs no API tokens at all — no audio is sent to any API.

---

Generated by `evaluation/write_report.py` from `metrics.json` (Config A, 210.93s), `metrics_b.json` (Config B, 82.92s), and `run_stats.json`.
