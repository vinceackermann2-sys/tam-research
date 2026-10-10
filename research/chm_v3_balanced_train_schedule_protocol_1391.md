# CHM-v3 #1391 — balanced training schedule, CPU Stage A

**Engineering only; no scored heldout run in this PR.** First #1377 scored CPU run `38064856253` (artifact 11674256746) is immutable. It had 48 steps per arm and 192 archived predictions; its TRAIN label schedule accidentally paired each family with one constant answer.

## Original confound and exact fix

For original step `4k+r`, entity `k mod 8`, variant `(5k+r) mod 8` and positive-family offset `r`, the answer index was `(3k+r+5k+r) mod 8 = 2r mod 8`. Thus all original TRAIN examples had direct→`CODE-18001`, overwrite→`CODE-18003`, and two-hop→`CODE-18005`. The original heldout models predicted those constant answers on every overwrite and two-hop rotation. A trained-model difference could not be meaningfully attributed to retrieval.

The NEW `tam_research/chm_v3_balanced_train_schedule_1391.py` enumerates the Cartesian product of eight TRAIN entities × four task families × eight counterfactual variants, then shuffles the **256 examples exactly once** using frozen CPU-only RNG seed `13772001`. All three arms see the same list, optimizer and 256-step upper bound. Every (training entity, positive family) has each of the eight answer values **exactly once**; 24 positive groups × 8 = 192 positives. Each of the eight negative entities has eight repeated no-match/abstain episodes = 64 negatives. The old source model, serializer, positional encoding and benchmark are not modified.

## Future independent one-shot scored report requirements

Before scoring, create a SEPARATE new source-locked report workflow PR, after exact-head and merged-main CI on this scheduler:

- Train exactly **256** CPU-only steps/arm; AdamW LR 0.005, weight decay 0; common original model initialization `1377005`; one CPU thread, no Modal/CUDA.
- Evaluation on **unused** development entity index **1**, unused test entity index **2**. Original #1377 used index 0 for both splits and those panels may never be reused/tuned on. Each future split 4 families × 8 variants × 3 arms = 96 predictions, 192 records total. Train entities always in separate `train` namespace.
- Use original `SealedModelView` and multiset-controlled records, no gold answers/pointers/role tensors/variant IDs at model inference. Each case: actual answer/abstain, confidence, hard-argmax pointer diagnostics, evaluator-held gold source indices, family, entity, split. Score positive answer-change following and negatives separately. Exactly three memory-blind controls at 12.5% per positive family. One independent entity per family prevents valid CI/superiority inference.
- Log exact frozen source blob SHAs, source tree, CPU execution time, train tokens, parameter counts, every predicted ID. One path-gated first merge-only workflow, original first attempt only and durable JSON Actions artifact. No test peeking/retrain/rerun if it fails.
- Model's regex tokenizer hashes the **whole identifier** (for example `V3-train-E003`), so novel development/test names have unseen embedding IDs; this may itself prevent generalization. That is a separate future tokenizer experiment and MUST NOT be silently changed while isolating the answer-schedule confound.
- The pointer model adds TRAIN-only gold address supervision and additional FLOPs. A 1%-parameter budget difference is not a fair compute/supervision match, nor is its soft-conditioned second-hop argmax proof of hard-read retrieval.

**No scientific claim:** tiny synthetic toy, 256 CPU steps, one heldout entity per family, not real-language or 100M capability. The 100M Stage-C `CHM_V3_100M_DAEC_STAGE_C_STOP`, consumed seed `2013161` and missing original trained-checkpoint parity evidence remain untouched. No GPU/Modal spend, new scientific seed, replication, scaling or breakthrough authorization.
