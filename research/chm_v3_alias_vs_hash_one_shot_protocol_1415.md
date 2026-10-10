# CHM-v3 #1415 — frozen one-shot alias-vs-hash CPU evaluation

**Status: prospective Stage-A synthetic comparison only.** This is not a 100M scientific run, not a new CPW capability result and not a GPU authorization.

## Provenance

Prerecorded issue #1415 before scoring. Stage-A alias adapter PR #1412 exact source head `37a9bcd5aa6fefb9cdeb109ff2daf50e00bbbd25` passed original CI `38072252566` (3132 tests, one skipped), pinned merge `508f17ebbf9829543b488472399dfc04efc6db4f`, exact merged-main CI `38072579871` PASS attempt1. Source dependency blobs pinned in dedicated workflow.

The original model/backbone `tam_research/chm_v3_matched_tiny_models_1377.py`, balanced TRAIN scheduler `tam_research/chm_v3_balanced_train_schedule_1391.py`, sealed view `tam_research/chm_v3_counterfactual_model_view_1365.py`, multiset benchmark `tam_research/chm_v3_counterfactual_multiset_suite_1373.py`, query-alias front end `tam_research/chm_v3_query_anchored_alias_1407.py` and alias adapters `tam_research/chm_v3_alias_tiny_adapters_1410.py` are **not modified by this PR**. All their exact Git blob SHAs are verified before one-shot training.

Old scored runs are immutable: confounded 48-step `38064856253` used development0/test0, balanced 256-step `38066888678` used development1/test2. Neither source run nor its scored entity panels may be replayed or amended.

## Experimental units and fairness

Four precommitted conditions:
- `whole_decoder`: original Transformer using original whole-identifier hash tokens
- `alias_decoder`: **same initialization and trainable parameters**, with query-anchored aliases from visible text only
- `whole_pointer_ce`: original Transformer-plus-two-hop-soft-pointer, TRAIN-only source-position CE
- `alias_pointer_ce`: **same pointer initialization and trainable parameters**, using alias tokens

All four receive the identical 256 TRAIN-only balanced samples in the exact frozen order (8 train entities × 4 families × 8 answer rotations). One AdamW optimizer update/example, LR .005/wd0, no dropout and fixed inherited model seed `1377005`, CPU one thread. Within corresponding whole/alias pairs the trainable count and transformer attention math are exact; the input token lengths and physical source address indices match. Encoder CPU wall time on TRAIN views is reported separately, including the alias encoder's additional legacy-parity safety check. Four arms have distinct optimizers and no test-derivative data access during training.

Only *previously unscored* development entity index **3** and test entity index **3** may be instantiated after training. Every split has 8 variants of each of four families = 32 examples/arm; exactly **256 predictions across 4 arms × 2 splits**. New-case source positions and evaluator-only gold labels are read **after the model produces a prediction**. Per case: predicted candidate/abstention/confidence; first/second soft-attention argmax diagnostic indices for pointer arms; evaluator-only gold answer and selected source positions. Score positive answer-following separate from no-match false positives, task-family breakdown, prediction diversity and exact diagnostic pointer-source matches.

The blind controls are mathematically exact: for each positive group, query-only, code-redacted-layout-only, and **code-value-bag-only** observations are identical across all eight answer rotations, so a deterministic blind predictor has at most 1/8 correct. This is verified on **only these new panels** and not by instantiating previously scored dev/test entities.

## Execution controls

Five entirely additive new files: scorer module, entrypoint script, static CI tests, frozen protocol and one dedicated first-addition workflow. Regular PR CI never executes the score/training loop; it tests TRAIN-only controls plus static scorer/CI invariants. Dedicated workflow triggers on the initial **main push that adds its own unique path**, no schedule or manual dispatch, with `github.run_attempt == 1`, CPU PyTorch, `CUDA_VISIBLE_DEVICES=""`, one thread, timeout <=35min, no Modal secrets, and fail-closed frozen source/blob hashes before training. Archive complete JSON/256 predicted rows plus original job logs in Actions artifact. If CI/run fails or is cancelled, **freeze it; never rerun/amend**.

## Limits of every possible result

This is a **representation ablation**, not a fair CPW architecture vs Transformer test. Query-anchored aliases explicitly encode string equality from visible text into shared model input symbols. Even a positive gain is credit to that front-end (which every alias condition receives), not proof of understanding or higher intelligence.

Only one independent entity group per family per split, so no meaningful CIs or generalized architecture superiority. Tiny 271,049-decoder vs 273,097-pointer parameter counts are within 1%, but pointer models have **extra TRAIN-only gold address labels, different FLOPs and SOFT (not sequential hard) two-hop reading**. Hash identifiers are split-specific embedding tokens and may themselves under-generalize. Previous old scores are not retuned here. Original #1323 100M `CHM_V3_100M_DAEC_STAGE_C_STOP`; scientific seed `2013161` consumed forever; historical trained-checkpoint scoring parity still absent. No breakthrough claim, scaling or GPU/Modal allocation.
