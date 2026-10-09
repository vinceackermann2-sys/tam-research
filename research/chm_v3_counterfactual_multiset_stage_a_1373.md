# CHM-v3 Stage A — matched CODE-multiset counterfactual benchmark (#1373)

**Status:** data/negative-control engineering only; CPU-only; zero new scientific execution authority.

## Why v3 is needed

The frozen original aligned-v4 test had an entity-to-answer positional shortcut, and v2 paired counterfactuals changed the global set/frequency of answer codes when changing an authoritative fact. A bag-of-code token classifier could theoretically exploit this without entity-binding. The model-facing text/labels of old test episodes and v1/v2 artifacts remain unchanged. V3 is a new disjoint benchmark series, not an amendment to historical science.

## New matched-binding contract

- Exactly eight cases per positive (split, family, entity, memory_length), with **identical query, positions, document/subject IDs, memory ordering, candidate options and all token-value frequencies**.
- An authoritative target record plus seven decoys contains **exactly one of each of eight candidate code values**, in every variant.
- To rotate the correct answer, **swap** the target value with the corresponding unrelated decoy value. This changes *two bindings* rather than adding or deleting tokens, while the global candidate-code multiset stays identical. In overwrite cases, a separate fixed noncandidate STALE code remains in the prior memory.
- `direct`: query entity→value. `overwrite`: latest entity→value; the stale previous value cannot be selected. `two_hop`: entity→document→record value with seven unrelated record decoys. `no_match`: no relation/fact about queried entity, but all eight candidate values present in unrelated memory.
- Entity namespaces are disjoint across train/development/test; `variant` never appears in a query, document ID, memory position or sealed text. For all families, memory positions are strictly before the query, at 128/256/512/1024 allowed envelope sizes.
- Use only v2's `SealedModelView` for a future model: query, rendered text of completed prior memory, and eight fixed answer choices. Oracle positions, ground-truth answer, split/variant, dataclass role masks, internal seed and episode metadata are withheld from the model. The source-bound oracle is strictly for CPU dataset checks/scoring.

## Controls and interpretation

Three model-free deterministic policies see respectively (a) query only, (b) query plus code-redacted memory layout, (c) **only the exact bag/multiset of code values**, without subject/position bindings. Because these observations are identical across all eight positive variants and the correct answer cycles once through each code, each control must score exactly **1/8 per group**. Always-abstain is scored separately on negatives.

The oracle locates the correct answer for all positives. It is not a trained model or a prediction task success. Later small trained CPW/pointer and Transformer comparisons must be prospectively preregistered with matched parameters/compute, disjoint held-out entities, per-case predicted IDs, hard retrieval trace indices, no-match and overwrite results, and confidence intervals across independent entity groups. At inference no gold pointer/role label or variant ID may be supplied. This v3 suite alone demonstrates **benchmark identifiability controls**, NOT better intelligence, natural-language generalization or a breakthrough.

Old #1323 remains `CHM_V3_100M_DAEC_STAGE_C_STOP`, scientific seed `2013161` irreversibly consumed; historical checkpoint/evaluator parity proof remains missing. No Modal/GPU, scientific rerun, 100M training or breakthrough authority here. All changes additive on a separate branch, first-attempt exact-head original CI gates. Frozen failed/cancelled heads must never rerun/amend.
