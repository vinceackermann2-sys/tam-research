# CHM-v4 prospective supplementary no-match reservation — #1456

**Classification:** NAME-ONLY / ZERO-EPISODE / ZERO-SCORE / ZERO-TRAIN / NO SCIENTIFIC ATTEMPT.

This additive proposal is separate from frozen #1421 benchmark and #1450's 32-entity prospective screen. It neither changes the existing TEST nor authorizes any new evaluation. No heldout episodes, records, answer labels, model predictions, training steps, random seeds, GPU allocations, or scientific attempts are created here.

## Deterministic prospective identity reservation

- Study namespace: `CHM_V4_FPR_REPLICATION_20261016_NAME_ONLY_V1`.
- Proposed independent entity **names**, not yet verified independent sampled entities: `V4-FPR-REP-20261016-E{index:04d}` for every integer index in `range(176)` (0 through 175 inclusive).
- First: `V4-FPR-REP-20261016-E0000`. Last: `V4-FPR-REP-20261016-E0175`.
- Canonical bytes: UTF-8 of the 176 names in increasing index order, joined by `\n`, with **no trailing newline**.
- SHA-256 of those canonical bytes: `fc8b1127bcfcff57c766593b378db2924591ced465ec07f7359acbaef6d7efcf`.
- Number of unique reserved names: 176. Names must remain disjoint from all existing V3/V4 train/development/test IDs; this does not by itself prove distributional or statistical independence.
- Any future generator must use a **separately source-frozen** RNG/domain and disjoint document/decoy/record namespaces. No such generator is supplied or run by this manifest.

## Why 176 rather than the issue's minimum 128?

At global one-sided alpha 0.05 and seven Bonferroni claims, the no-match claim uses alpha=0.05/7. Under a binomial independent-entity model and a strict upper bound < 0.05, the minimum number of independent negative entities is 97 for zero false-positive entities, 138 for at most one, and 174 for at most two. The 176-name proposal offers *prospective capacity* for two errors, **not** an assurance of fewer than two errors. At n=176 the one-sided exact upper false-positive-rate bounds for 0, 1, 2, and 3 observed false-positive entities are approximately 0.027687, 0.039233, 0.049205, and 0.058406 respectively. Repeated identical rotations of one absent-query entity count as one negative, never eight independent trials.

The original `tam_research/chm_v4_grouped_go_no_go_1450.py` explicitly requires 32 negative entities and **cannot be reused unchanged** for a supplementary 176-entity test. A *new, prospectively reviewed and source-locked* supplement evaluator with independently validated provenance and multiplicity must be approved before any new evaluation. Do not alter the frozen original 32-case gate retroactively.

## Positive-family feasibility warning (independent of the negatives)

The frozen #1450 paired Hoeffding bound with n=32 independent positive entities per family and alpha=0.05/7 subtracts `sqrt(2*ln(140)/32) = 0.555745...` from the mean paired accuracy advantage. Thus the gate cannot pass unless the observed advantage is >55.57 percentage points per family; if the fair Transformer accuracy is at least 44.43%, even 100% CHM accuracy cannot pass this conservative gate. With a hypothetical fixed observed paired advantage of 20%, 10%, or 5%, a *new prospective* study would need at least 248, 989, or 3954 independent positive entities per family respectively under the same Hoeffding formula. These are **deterministic threshold calculations, not a power simulation, scored result, permission to inspect reserved tests, or justification to change a consumed protocol**.

## Conditions before any later experiment

1. Review this manifest against actual existing/reserved identifiers without generating DEV/TEST episodes. Freeze namespace, name digest, new RNG/domain, expected input distribution and independence/collision tests *before* any new cases are instantiated.
2. Preserve eight candidate CODE values and positive-matched memory lengths, distractors, layouts, query-subject frequency, and distributional difficulty. No shortcut features that identify negatives by namespace or layout. Verify group independence at the entity level, not by counting rotations.
3. Create a separate prospective source-locked protocol, fair Transformer control, bounded budget, first-attempt evidence and approved scientific execution for any future score; require an independent replication and compute/utility benefit for breakthrough consideration.
4. All implementation uses isolated branch -> PR -> original first-attempt CI PASS -> live main audit -> exact-head pinned merge -> first merged-main CI PASS. Failed/cancelled heads and consumed scientific attempts never rerun or amend. No direct write to main.
5. Original `CHM_V3_100M_DAEC_STAGE_C_STOP` and consumed seed `2013161` remain immutable. Colab Pro GPU/paid science remains unapproved.

**Status:** proposal only; not an authorized heldout set, training run, CI execution, scientific attempt, or breakthrough claim.
