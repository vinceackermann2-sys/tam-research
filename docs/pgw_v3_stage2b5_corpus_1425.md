# PGW-v3 Stage-2B5 — complete candidate corpus and evaluation protocol

**Status:** zero-training scientific design and static fixture integrity only. Frozen executable issue [#1429](https://github.com/vinceackermann2-sys/tam-research/issues/1429) following Stage-2B4 [#1422](https://github.com/vinceackermann2-sys/tam-research/issues/1422). A pass means `PGW_V3_STAGE2B5_CORPUS_METRICS_PROTOCOL_READY`, **not trained retrieval or an approved CPU run**.

## Source, isolation and scientific governance

The prior Stage-2B4 PR #1424 merged as `3e0b9c7e2fffe15c3e7ae10110dae67deb4275d6`; original exact-head CI #38075428852 and merged-main CI #38075759510 passed (3,213 tests, one skipped). The Stage-2B5 contract audited live main `ea874c019e03bfd4ac725677df6a3bf08ad7dd5f`; isolated branch `research/pgw-v3-stage2b5-corpus-metrics-v1` was created on live main `690b072e67623925e6b77c322ea4150441ca0d5a` after unrelated CHM-v4 and CORTEX files appeared. Historical PGW-v2 #1181 and PGW-core #1201/#1213 negative results remain unchanged.

No modification to frozen PGW Stage-0/1B/2A/2B0–2B4, scientific runner, CI, CHM/RLT/CPW/CORTEX, Modal/H100 or any paid infrastructure. **No optimizer, training loop, train-state update, scientific seed reservation or run trigger is implemented.**

## Complete candidate dataset specification

The corpus uses the existing, independently validated Stage-1B `make_example()`, unchanged key/value/filler grammar, and externally returned oracle answer labels. No answer is inserted into model input. Every input's SHA256 is recomputed independently, the strict key/value causal oracle revalidates its label, final READ anchor is verified, and the answer token is forbidden from the READ suffix.

| Split | Samples per factorial cell | Total samples | Present queries | Missing queries |
|---|---:|---:|---:|---:|
| Train | 16 (indices 0–15) | 2,880 | 1,440 | 1,440 |
| Validation | 8 (indices 0–7) | 1,440 | 720 | 720 |
| Test | 8 (indices 0–7) | 1,440 | 720 | 720 |
| **All** | | **5,760** | **2,880** | **2,880** |

There are exactly **180 factorial scenario cells per split**: last-tracked-WRITE position `2..6` (5), old overwrite count `0..2` (3), interference flag (2), completed-chunk delay `1/2/4` (3), queried key present/absent (2). Complete total input tokens: **522,240**. Input length is 80, 88 or 104 tokens (no padding). Fixture **indices are not scientific optimization seeds**.

Each row binds split/factors/index, `sha256(input_bytes)`, an ordered semantic WRITE/READ trace hash and the external verified answer into one canonical ASCII JSON line; sorted lines are SHA256-committed at global and per-split roots. Input fingerprints must be globally distinct across all 5,760 records. The new stricter semantic trace excludes filler text and encodes the ordered `[(key,value),...]` WRITE sequence followed by READ's queried key; exact semantic duplicates also fail, including cross-split collisions.

**Limit:** Observed exact trace uniqueness does *not* establish distribution independence. Train and test share the same syntactic task grammar and constrained key/value domains, so transfer to real language and immunity to template memorization are not established. These are proposed candidate sample identities and full roots; the actual digest strings need to be recorded from a completed CI result into a frozen follow-up scientific contract before any optimization can begin.

## Evaluation-only API

`evaluate_complete_predictions(manifest, split, predictions)` is pure Python: it accepts complete supplied `(sample_id, output_token)` predictions for a single split, refuses duplicate/missing/extra/invalid classes and reports:

- Overall, present-only and missing-only accuracy with exact numerators and denominators; 180-cell **uniform macro accuracy**.
- Correctness per delay bucket, overwrite count and last-tracked-WRITE position.
- Missing-key detection precision/recall and TP, FP, FN and TN confusion counts for class `NOT_FOUND=64`.
- Per-cell correct/total, including the absent-key half and overwrite/recency strata.

The baseline `always_not_found` must score exactly **50% overall and macro**, **0% on present-key queries**, **100% missing-only**, with absence precision=50% and recall=100%. The independent oracle should score **100% in every stratum**. Naive copy-second-chunk, latest-WRITE-ignoring-key, and most-frequent-WRITE-value algorithms are checked as shortcut integrity tests only, **not neural baselines**.

The evaluator does not invoke models, read logits, fit features, create trainable parameters or run any forward/backward learning steps. External scientific model predictions would need a new, source-bound and explicitly authorized run.

## Candidate prospective scientific decision rule — not a trained result

Any separately preregistered future study must freeze the scientific seed identities, job IDs/reservations, loss multiplier `lambda`, optimizer, learning rates, minibatch ordering, stopping schedule, compute budgets and actual active parameter/compute-fairness policy before optimizer execution.

If those fairness gates pass and at least **three new paired seeds** are authorized, an ambitious task-specific success threshold is:
- Held-out test **macro-cell >=80%**, **present-only >=80%**, **missing-only >=90%**.
- Across paired attempts, mean PGW hybrid minus full **global-position** causal Transformer test macro accuracy >= **+5 percentage points absolute**; PGW present-only at least 75% in each seed and positive paired macro deltas for at least two of three seeds.
- No post-hoc picking the best seed, data split, route mode, auxiliary coefficient or sampling budget. Show all negative results and missing-key shortcuts.

**These candidate thresholds do not resolve current active parameter mismatches, recurrent-vs-full-attention depth or measured compute gaps.** The 20,354 PGW and 20,347 Transformer instantiated parameter counts are not proof of capacity-matched science. The predictor auxiliary loss has different information inputs under full versus chunk-local attention, so matching loss expression is not enough. A future run contract must explicitly classify each arm as matched primary or unmatched mechanistic ablation based on measured evidence, normalize equal-token versus equal-wallclock/FLOP budgets, and include seed-level uncertainties.

No H100/Modal experiment can be justified from synthetic retrieval outcomes alone; a scaled language-model NLL test would require an entirely new preregistration and explicit paid-run authority.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`scale_up_authorized=false`  
`breakthrough_claim_allowed=false`
