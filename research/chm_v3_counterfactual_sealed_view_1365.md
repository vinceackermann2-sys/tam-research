# CHM-v3 #1365 — sealed-view paired counterfactual memory benchmark v2

**Status:** CPU-only data + oracle + adversarial benchmark gates, **no model trained**. Isolated successor to #1358 (PR #1362); never edit its v1 artifacts after their first CI-verified merge.

## Motivation and new constraints

The first counterfactual suite proved that identical queries with eight different gold answers prevent a *query-only* model from exceeding 1/8 within each group. However, v1 generated different memory addresses, document names (including the variant index), and distractor codes across variants. A future model might exploit those uncontrolled changes without reading the authoritative answer token.

V2 fixes all such incidental features **within a group**. For a single (split, family, entity, memory length), eight episodes have exactly the same query, same relation and document identifiers, same source positions, same decoys, same stale value, same eight answer options, and the same record order. **Only one target answer-code span changes**, cycling through the eight answer candidates once.

The v2 generator uses its own namespace and deterministic hash-derived memory positions that *exclude* the rotation index; no original GPT-2 aligned-v4 test cases or scientific seeds are reused.

## Sealed model input contract

Only `SealedModelView(query, memory_text, answer_options)` may be given to a learned model in a future experiment. It excludes `variant`, `split`, `family`, `gold_answer`, gold pointer positions, internal `kind` role IDs, oracle links and seed metadata. Memory facts are rendered as **ordinary text statements** about subjects/documents and code values in increasing memory-position order; no target pointer is revealed. Exposing the factual *relation itself* is legitimate input, whereas exposing an oracle-selected relation position is not.

**Evaluation contract for later experiment:** Freeze a separate model-facing text/token encoding and train-only labels. At inference, score eight candidates or abstain without consulting the episode metadata; then compare with evaluator-held gold labels. Report exact predicted candidate, abstentions, first/second selected positions, positive and negative breakdown, overwrite recency, two-hop routing and sensitivity when only the target fact is counterfactually changed. The CPU generator's oracle is only for dataset verification and scoring.

## Independent negative controls

- Query-only input removes `memory_text`. It is identical across all eight rotations and must score exactly 1/8 in each positive group.
- Layout/metadata-only input redacts every `CODE-<number>` string in memory while preserving its query, positions, document IDs, fact order and relations. It must likewise score exactly 1/8 within each positive group.
- Always-abstain is measured **only** on no-match negatives, separate from positive accuracy. A high negative score cannot mask failure to retrieve positive facts.
- A full semantic oracle can select the target answer from earlier completed memory at 128/256/512/1024 positions. Oracle reachability is **not** model quality.

The tests verify exact one-value-only differences, all eight counterfactual answers, no-answer-in-query, deterministic disjoint entity splits, immutable same-document memory geometry, latest-authoritative overwrite and relation-pointer resolution, strict no-future-memory bounds, and fail-closed tampered layouts.

## Interpretation ceiling

This benchmark is **more resistant to shortcuts**, but still synthetic and text-templated; no Transformer or CPW/DAEC-A model has been trained or evaluated using it. It cannot demonstrate natural-language understanding or a breakthrough. Any meaningful matched trainable comparison is separate from this data-only PR and requires its own preregistration; a GPU/Modal paid run additionally requires explicit authority and unique reserved scientific seed.

Historical original #1323: `CHM_V3_100M_DAEC_STAGE_C_STOP`; seed `2013161` irrevocably consumed; missing original trained-checkpoint parity certificate not repaired by this work.
