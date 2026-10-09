# CHM-v3 Stage A #1377 — tiny matched CPU implementation v1

**Scope:** unscored engineering implementation only. No GPU/Modal, no 100M model, no historical checkpoint replay, no new scientific seed and no breakthrough claim.

## Fixed model/data contract

- Source benchmark: merged v3 multiset-preserving counterfactual suite in `tam_research/chm_v3_counterfactual_multiset_suite_1373.py`. It contains direct/overwrite/two-hop/no-match cases with the full candidate-value bag constant across all positive answer rotations.
- Only `SealedModelView(query, memory_text, answer_options)` reaches `model.forward()`. The tokenizer derives chronological line positions and CODE values from the **rendered text**, not from an episode's gold pointers/role masks.
- One shared SHA256-hashed **8192-bucket** regex-token vocabulary for all arms; query follows the memory; fixed sinusoidal positions; single **causally masked** 32-dimensional Transformer layer, 4 heads, FFN 64; no dropout. The conventional decoder predicts one of 8 codes or abstains through its final readout.
- A: Transformer decoder-only baseline with 9-way answer CE.
- B: same-initialized decoder and classification head, plus query/key pointer projections. It uses a learned first-hop soft read from the visible memory tokens, then a **predicted**, not gold, first-hop context for second-hop attention over visibly coded memory positions. It combines candidate-code copy probabilities with the conventional answer-head logits. TRAIN loss = answer CE + **0.4×(first-position CE + second-position CE)** on positives. On negatives, answer-only abstention supervision. **No evaluator-held gold pointer/payload enters model.forward().**
- C: identical B architecture and initialization, **without auxiliary position CE** (answer CE only). This controls whether performance depends on explicit address supervision.
- Both pointer variants have extra trainable projection weights, so the exact parameter count must be reported. A and B/C total trainable-parameter mismatch is capped at **1%** by construction, and runtime/token counts must be reported too. Parameter parity is not proof of FLOP parity; any CPU-time/FLOP mismatch forbids a strict fair-science claim.
- Training is on `train` split only, at most 48 deterministic sequential synthetic episodes per arm, identical episode order and 1 optimizer step per episode, AdamW LR 0.005, weight decay 0; same answer vocabulary/encoder. The ordinary repository CI only performs a **single-step CPU smoke**, source validity, gradients, oracle-target separation, initialization and parameter parity tests. This PR **does not score held-out development or test**.

## Next, separate one-shot scored study (not included here)

After this PR's original exact-head CI and merged-main CI pass, independently freeze a single one-shot CPU report script/workflow with exact source tree, batch order, total step budget, metrics, and report location *before execution*. Cap to 48 train steps/arm and at most 32 development plus 32 held-out test cases per arm, cluster confidence by independent entity. Log exact predicted answer code, abstentions, confidence, first/second hard-selected visible token indices, source oracle positions only on evaluator side; report direct/overwrite/two-hop/no-match separately and counterfactual flip adherence, bag-only, query-only and redacted-layout controls.

**Important limitations:**
- Hash collisions and train/development/test lexical identity disjointness can strongly limit learned generalization; the model uses synthetic templated text and one small deterministic seed, not natural language corpora.
- Extra pointer computation is not yet FLOP-matched to the baseline; a positive pilot does NOT demonstrate Transformer superiority.
- No oracle roles or source indexes at inference, but training-position CE directly supervises memory addresses; such labels would need clear budget accounting in any future matched scientific comparison.
- Historical original scientific run #1323 remains `CHM_V3_100M_DAEC_STAGE_C_STOP`, consumed seed `2013161` never reused. Original trained-checkpoint scorer parity remains unproven. No 100M science, no paid runs or Stage D authority.

This PR is additive and isolated. If first-attempt CI fails, freeze the entire head and open a fresh successor with an explicit diagnosis; never retry/amend the failed run.
