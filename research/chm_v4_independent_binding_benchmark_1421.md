# CHM-v4 #1421 — new independent-entity binding benchmark, Stage A

**Status: zero-GPU, TRAIN-only benchmark geometry and anti-shortcut proof. This PR does not construct future DEV/TEST cases, perform model forward, train, evaluate, or authorize score workflows.**

## Why v4 follows the negative v3 alias report

Original successful #1419 CPU one-shot [workflow 38074130939](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38074130939) preserved 256 scored case records. Test positive /24: whole decoder 3, aliased decoder 3, whole pointer+TRAIN CE 3, aliased pointer+CE 4. One independent entity per family and split: no inferential claim. Aliased pointer direct selected correct first address 8/8, correct second CODE position 5/8 but generated the right answer only 1/8 and abstained on 7/8. Soft address argmax indices DO NOT imply sequential hard read. Both old development and test index3 are now consumed, alongside older index0,1,2 scored entities. The original 100M Stage-C remains STOP and seed 2013161 consumed.

The next *mechanistic* hypothesis separates three operations:
1. Find the correct prior-memory **source address**, respecting overwrite recency and latest two-hop relation;
2. Extract the CODE payload from exactly **that** completed-memory record (not an unrelated decoy);
3. Gate between emitting the extracted answer and abstaining, using only legitimate visible-text evidence.

## New independent experimental units, preregistered name-only manifest

The new source lives in `tam_research/chm_v4_independent_binding_benchmark_1421.py`. Identifiers use `V4-...`, **not** old `V3-...` namespaces. Separate sha256-derived RNG `CHM-V4-1421-ONLY-TRAIN-STAGE-A` decouples all new record positions/code rotations/layouts from previous v3 tests. Independent source entities: **64 TRAIN**, **16 development**, **32 TEST**; the manifest function returns only counts, first/last names and SHA-256 over the reserved name lists. No heldout cases, test labels, predictions or prior-score tuning are used to construct this audit. A future actual scoring protocol needs its own frozen harness, checksums and explicit release of panels.

Every independent entity has 4 families×8 variants using memory length 128: `direct`, `overwrite`, `two_hop`, `no_match`. Eight positive variations keep exact query, stored identity/role/physical positions, template and total **CODE-18001..18008** multiset. Only the authoritative target CODE and one distractor CODE swap. Negative `no_match` contains **every** candidate value elsewhere but none for the query subject, and its variants are identical. Blind query-only, code-bag-only, and redacted-layout-only signals are constant across each positive eight-way group: at most **1/8** correct by group.

- **Direct**: one exact query-subject value among seven unrelated subjects.
- **Overwrite**: older query-subject `CODE-19001`, newer query-subject correct CODE, plus all seven candidate decoys; use source-address recency.
- **Two-hop**: older query-subject relation refers to a decoy document that still has a CODE; newer relation refers to a distinct target document, among seven candidate decoy records. Correct result requires following latest relation→target document→its CODE. These are distinct name strings and document identities.
- **No-match**: query subject is absent from stored facts although each CODE appears once at an unrelated subject, preventing "CODE present therefore answer exists".
- Two deterministic text serialization templates and two query styles introduce limited lexical diversity; *no* tokenized model head or existing frozen source is changed.

The structured `Episode` and evaluator-only `oracle_read` may see source roles/positions/labels. A future learned model may only be given `SealedInput(query,memory_text,answer_options)`, whose dataclass has **no labels, variant ID, family/split metadata, entity index or gold address**. Exact source positions are printed in completed-memory text as ordinary record anchors, not secretly injected.

## Scope and gates

CI constructs exactly **2,048 TRAIN episodes (256 eight-way groups)**, validates the oracle and value-bag invariants, absence of anti-shortcut leakage, physical source geometry, latest overwrite/relation semantics, both serialization templates, and malformed-input failures. It does not construct any development/test examples or train the v3 model. The future heldout names are *reservations* only; not scores. In particular, 32 independent TEST entities **per family** give 32 independent groups/family, and each group contains eight correlated rotations; group/cluster is the statistical unit.

This is not evidence of novel architecture superiority. Before *future* scored CPU runs: freeze a genuinely executable source-address→payload-value hard-read mechanism with NO evaluator gold as model input; include original Transformer, pointer soft-read, value-only head, copy-only, and answer/abstain ablations. Match budget/optimizer/tokens/FLOPs and pointer-label supervision as appropriate. Define independent-entity accuracy/confidence lower bound vs 1/8 and maximum false-positive rate *before* opening TEST. Future threshold examples in #1421 are proposals, not retroactive posthoc cutoffs. A failed/cancelled original scientific or CPU-scored run is never rerun/amended.

Governance: additive isolated branch; original exact-head CI attempt 1 PASS; re-audit live main → pinned expected-head merge → separately verify exact merged-main CI attempt 1 PASS. **No GPU/Modal, no new scientific seed, no scale-up, no breakthrough authority.**
