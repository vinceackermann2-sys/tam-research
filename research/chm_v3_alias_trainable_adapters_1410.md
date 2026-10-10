# CHM-v3 #1410 — matched trainable tiny adapter pre-scoring

**Classification:** `CHM_V3_1410_ALIAS_VS_HASH_TRAIN_ONLY_ADAPTER_CPU`. Additive Stage A, not a scored scientific experiment.

## Hypothesis to test in a future, separate PR

Could a *visible-text-only* query-anchored local identifier alias (`ID_ALIAS_Q`, `ID_ALIAS_001`, etc.) improve memory binding across new entity names versus original whole-name SHA256 hashing, under exactly the same tiny Transformer and pointer math? We cannot infer this from #1395's negative scored run, #1399's static character audit, or #1407's static alias audit.

## What is implemented

- The preexisting frozen `TinyDecoderOnly` and `TinyTwoHopPointer` remain unchanged, with same fixed initialization, `nn.Embedding(8192,32)`, one 4-head causal Transformer block, answer classifier, and (pointer arm only) query/key projections and soft two-hop code-copy logits.
- `AliasedTinyDecoderOnly` and `AliasedTinyTwoHopPointer` subclass the frozen arm with *no new trainable parameters*. Only the encoder is swapped: `alias_sealed_input(view)`→`EncodedView`; visible source/code/anchor/token geometry is independently required to equal the original encoder. The internal alias dictionary is never passed to the forward path.
- Reimplemented forward math is tested **bitwise-equal** to original model forward for the old tokenizer on TRAIN views at identical init. Query-to-memory matching inductive bias remains a disclosed intentional intervention; no gold labels, source roles or evaluator oracle enter input adaptation.
- Four pre-scored comparison arms: whole-hash decoder vs alias decoder, and whole-hash pointer+CE vs alias pointer+CE. Within each pair we fix initial parameter names/values, count, 256-case TRAIN data, optimizer type, train order, 128-position memory and encoded sequence lengths. Address CE is only compared within pointer arms, *not* across Transformer/pointer as equivalent supervision.
- `verify_train_only_encoder_parity()` scans all 256 TRAIN episodes, no gradients. `one_train_step_four_arm_smoke()` performs exactly ONE identical TRAIN optimizer step per arm, without running learned inference for a heldout example, computing accuracy, or starting an unapproved 256-step scoring experiment. CI checks finite answer, pointer and embedding gradients and exact backward wiring.
- Equal token counts imply equal coarse attention-length/parameter FLOP *proxies*, not equal wall compute: alias tokenizer does extra query parsing, mapping and legacy validation. A source-locked run logs this preprocessing timing as *nonportable development measurement*, not scientific latency advantage.
- Existing original `train_only_targets()` derives gold address indices outside model forward, still valid because aliases preserve every token position. The baseline and aliased pointer CE have identical auxiliary target indices and supervision coefficient 0.4.

## Strict future prerequisites

1. Verify merged-main exact CI before preparing new scored task; failed/cancelled heads frozen, not rerun or amended.
2. Preregister never previously scored **fresh** development and test groups before training/scoring, with audit of old uses; prior #1377 entity index0, #1395 dev1/test2 never reused. #1410 proposed dev3/test3 subject to independent audit, not approved here.
3. A *separate* exactly source-locked CPU one-shot report may use 256 train steps per arm and balanced candidate rotations only after preregistration; include every prediction, counterfactual flip-following, abstention and query-only/multiset controls, entity-level uncertainty and pretrained-token collision checks.
4. Add a **non-query-anchored canonicalization** arm to distinguish general alias compression from special query-anchor inductive bias and separately test no-match query-only cues. Compare optimizer samples, precision, training tokens, per-step preprocessor wall time, and FLOP estimates; never claim identical actual compute from token lengths alone.
5. Original 100M DAEC `CHM_V3_100M_DAEC_STAGE_C_STOP`; scientific seed `2013161` permanently consumed. Original trained-checkpoint evaluator parity not established. No paid run, GPU/Modal, large-scale extension, rerunning past heldout panels, or breakthrough statement.

## Governance

Branch→new draft PR→original exact-head CI PASS attempt1→live-main audit→expected-head pinned merge→merged-main CI PASS attempt1. One development source-bound TRAIN-only evidence workflow is permitted. Failed/cancelled first heads frozen.
