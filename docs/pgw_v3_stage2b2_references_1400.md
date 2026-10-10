# PGW-v3 Stage-2B2: near-parameter-count causal references

**Classification:** `PGW_V3_STAGE2B2_NEAR_INSTANTIATED_PARAMETER_REFERENCES_STRUCTURAL_ONLY`. This is **not** a trained model study.

Preregistered in [issue #1400](https://github.com/vinceackermann2-sys/tam-research/issues/1400) from live main `29250b7e474bc47855fa482449fd1d793c9f7e3c`. Branch `research/pgw-v3-stage2b2-causal-references-v1` was actually created from audited live main `081b089731a9663e20c14b5d96a65bbea4d0129d` after three unrelated reduced-attention/CORTEX paths appeared. The PGW history is immutable: Stage-0 #1359; Stage-1B #1374; Stage-2A #1379; auxiliary path #1384; static preflight #1394, PR #1396, merged at `b2f7825fafacddee801517bd9132c0ee2b60259f`; original-head and merged-main CI passed. Frozen PGW-v2 #1181 and PGW core #1201/#1213 remain negative.

## Actual reference architectures

Both references share the same code, initialized from identical parameter tensors. Only attention scope differs:

- **Full causal Transformer:** one 4-head, width32 self-attention block over **all** prior tokens in the complete input prefix, with a strictly upper-triangular future mask. A query at final READ offset2 can therefore read earlier WRITE tokens directly. Attention complexity grows quadratically with sequence length.
- **Chunk-only negative reference:** same attention/FFN module and parameters, but each eight-token chunk is encoded independently and identically with a causal 8×8 mask. Query logits cannot access earlier chunks; this deliberately tests whether a model can bypass memory on local cues or missing-key shortcuts.

Both receive the Stage-1B canonical 256-token inputs, 8-position reused within-chunk embedding, externally verified `batch_verified_examples` from frozen Stage-2A, and identical 33-class value/NOT_FOUND answer-head semantics. There is **no oracle target in the token stream**. The final READ/QUERY markers are strict and the final QUERY logit cannot read later filler.

Both also have the same **genuinely used** 512-parameter predictor `Linear(32,8,bias=False) → GELU → Linear(8,32,bias=False)` for the same Stage-2B0 auxiliary objective:

`mean_{batch,8 WRITE event chunks,7 prediction steps} ||LN(pred(H[p-1])) - stopgrad(LN(H[p]))||²`

The full causal reference's latent H includes earlier completed chunks; the chunk negative control's H does not. Neither uses READ labels or delay chunks in the auxiliary loss. **No auxiliary multiplier `lambda`, optimizer, scientific seed, training schedule or checkpoint is selected.**

## Parameter accounting

| Parameter group | PGW Stage-2A | Each reference |
|---|---:|---:|
| Token + within-chunk position embeddings | 8,448 | 8,448 |
| Causal attention layer | included in local block | 4,224 |
| Residual norms + feed-forward (hidden width 90 in reference) | included in local block | 6,010 |
| Classifier + output norm | 1,153 | 1,153 |
| Predictor used by auxiliary objective | part of 2,209 workspace | 512 |
| Other workspace and utility-routing parameters | part of 2,209 workspace | none |
| **Total instantiated** | **20,354** | **20,347** |

Absolute gap is **7 parameters, about 0.0344%**. Parameter count is within the proposed 0.1% *instantiated* tolerance; this is NOT yet an active-parameter capacity match. All baseline parameters participate in either the answer or auxiliary pathway, but exact live-gradient coordinates depend on task/initialization and must be measured separately. PGW route modes have unequal active counts despite equal instantiation.

## CPU-only structural guarantees

The new tests verify dimensions and exact per-module counts, output logits and auxiliary loss on valid present/missing examples for all 1/2/4 chunk delays, future-READ suffix invariance, older-chunk intervention effects and *input-position* gradients for the full Transformer, lack of earlier-chunk pathways for the chunk control, positive predictor gradients from auxiliary-only backward and zero predictor gradients from answer-only CE. All checks are single-pass derivatives; **no optimizer update or model training occurs**. They also reject illegal tensor geometry, IDs and final query markers.

## Explicit limitations before research training

The full-attention baseline and PGW have **very different computation costs and information access**, even at nearly equal parameter count. The full Transformer sees every prefix token; PGW has chunk-local encoding plus four bounded workspace slots, two selected events per completed chunk. The chunk-only reference cannot access earlier events at all. The design exposes mechanisms, not an already-fair performance contest.

A future scientific study still needs: (a) empirical AND analytical active-parameter parity, not inert padding; (b) consistent auxiliary-loss coefficient/optimizer/initialization and matched objectives; (c) exact full training/validation/test corpus/manifest hashes and fresh paired seeds; (d) identical token budgets and normalized compute/latency and walltime evidence, as well as separate equal-FLOP or equal-walltime views; (e) multi-seed preregistered present-only retrieval superiority thresholds and per-cell/missing-key rates to beat the trivial 50% NOT_FOUND baseline; (f) separately explicit bounded CPU training authorization, no reuse/retry/resume; and (g) a separately preregistered paid 25M language benchmark gate before any H100 attempt.

If the first-attempt exact-head CI and merged-main CI pass, the classification is `PGW_V3_STAGE2B2_NEAR_INSTANTIATED_PARAMETER_REFERENCES_READY` **only**. No breakthrough claim, no trained retrieval advantage, no Transformer/NLL claim.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`breakthrough_claim_allowed=false`  
`scale_up_authorized=false`
