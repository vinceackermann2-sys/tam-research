# CHM-v3 #1410 — alias-vs-hash trainable tiny adapters (Stage A)

**Status: implementation and TRAIN-only gradient smoke, NOT a scored or scientific experiment.**

## Source and scope

The frozen existing Transformer and two-hop soft-pointer implementations from `tam_research/chm_v3_matched_tiny_models_1377.py` are inherited intact. Two additive classes use **exactly the same initialized parameters and attention/copy mathematics**. Their forward source ASTs are checked to match the original corresponding `forward()` function byte-for-byte in semantic AST except for a single explicit encoder call: replace `encode_sealed_view(view)` with `_encode_alias_view(view)`.

`_encode_alias_view` calls the already-merged #1407 query-anchored **text-only** alias encoder and passes its tokens/physical source anchors/CODE positions through the original immutable `EncodedView` schema. Identical input lengths, memory boundaries, position labels and candidate CODE payloads are required in every tested TRAIN case. Nothing from `PairedEpisode`, gold answer, `Fact` roles, variant or split may enter any model forward call. The evaluator may still provide source-position CE targets through the unmodified original TRAIN-only objective, clearly an additional supervision budget.

The four arms implemented for a *future independent* training evaluation are:
- Baseline whole-name-hash TinyDecoderOnly
- Query-anchored alias TinyDecoderOnly
- Baseline whole-name-hash TinyTwoHopPointer with TRAIN-only address CE
- Query-anchored alias TinyTwoHopPointer with identical TRAIN-only address CE

All four use identical initial tensor values within architecture, identical parameter counts (Transformer 271,049; pointer 273,097), the same 384-token context and originally fixed 8+abstain candidate head. Local aliases expose equality to every arm using them; they are a **different input inductive bias**, not learned memory or an improvement by themselves.

## CI/source gates

Ordinary repo CI executes only source-AST exact forward parity, initializer/state_dict equality, token/source-index parity, finite forward and gradients on a handful of TRAIN cases, one TRAIN example/optimizer step per arm, strict smoke cap, and hidden metadata rejection. It **never constructs or scores development/test examples** and does not launch a 256-step run.

Model aliases derive only from the sealed text and may legitimately make no-match detection easier by making absence of a queried entity explicit. Every future model that is claimed fairly comparable must share the same model-visible input transformation, and the difference between whole-hash and alias is reported as a **representation effect**, not an architectural effect. The pointer's second hop remains conditioned on a **soft** first-hop read; argmax positions cannot validate real hard sequential retrieval.

## Next separate one-shot experiment (not included)

Only after original exact-head CI PASS, pinned merge after fresh live-main audit, and exact merged-main CI PASS, freeze an independent source-locked CPU report PR on a new branch. It may train all four arms on the original balanced 256 TRAIN examples in exactly the same order, with original AdamW settings and source model initialization. Explicitly record the extra CPU preprocessing time for aliasing, forward/backward wall time, parameters/tokens, answer/source-index predictions, and negative controls.

Candidate NEW heldout dev/test entity index 3 must be checked prospectively against all prior scored runs before source lock; earlier dev0,dev1 and test0,test2 are consumed. No test tuning or rerun after observation. Four arms × two splits × four families × eight rotations means 256 scored case records if all four arms included. One independent entity per family/split permits no valid confidence interval or scientific superiority inference.

Historical original Stage-C #1323 remains `CHM_V3_100M_DAEC_STAGE_C_STOP`; seed `2013161` irrevocably consumed and original trained-checkpoint evaluator parity absent. No GPU/Modal credits, new paid scientific run, 100M/Stage D experiment, or breakthrough claim.
