# PGW-v3 Stage-2B0 — predictor auxiliary gradient path (CPU only)

**Status:** Structural engineering experiment, NO model training, NO optimization, NO GPU.

**Source/lineage:** Frozen negative PGW-v2 mechanism #1181 and PGW core #1201/#1213; Stage-0 #1359/PR #1363; Stage-1B oracle #1374/PR #1376; Stage-2A #1379/PR #1381 merge `5f799e625cdf992d5e70e4f381e537a659642e25` (original and merged-main CI PASS). Stage-2B learning design [#1383](https://github.com/vinceackermann2-sys/tam-research/issues/1383), Stage-2B0 contract [#1384](https://github.com/vinceackermann2-sys/tam-research/issues/1384).

Branch: `research/pgw-v3-stage2b0-predictor-gradient-v1` from audited live main `e25e4f91124f3ce52f2679c229e6510ee8bcc957`. Since frozen issue source `0f01e4a7ec9069cb77930344457b61aaf0a3d140`, only unrelated CORTEX/reduced-attention engineering files were introduced.

## Problem

The frozen `UtilityAddressedWorkspace` computes a predictor's latent surprise ranking, but detaches its prediction and target. Stage-2A answer cross-entropy therefore **cannot teach the predictor**. This Stage-2B0 helper constructs one isolated differentiable self-supervised auxiliary loss using the **existing predictor matrices**, leaving the original detached Stage-0 routing completely unchanged.

## Exact tensor contract

Stage-1B batch inputs remain `tokens LongTensor[B,T]`, anchored at the `QUERY` token in the final `READ` chunk. The Stage-2A local encoder uses its original 32-dimensional token embeddings, 8-position embeddings, 4-head causal chunk attention, and FFN. Only the first eight completed WRITE chunks are used as event data.

Construct `H[B,8,8,32]`, each chunk independently causal. For `c=0..7`, `p=1..7`:

```python
z[c, p] = predict_up(gelu(predict_down(H[c, p-1])))
target[c, p] = layer_norm(H[c, p].detach().float(), (32,), eps=1e-5)
error[c, p] = ((layer_norm(z[c, p].float(), (32,), eps=1e-5) - target[c, p]) ** 2).sum()
L_aux = error.mean()     # batch × 8 event chunks × 7 predicted positions
```

Within each chunk, position 0 is never predicted from any previous *chunk*. Position 7 appears **only as a detached target**, so a synthetic independent `H` leaf has exactly zero gradient at each `H[...,7,:]`. All READ tokens, labels, oracle answers, and delay chunks are omitted entirely. No target token is supplied to the model as input.

The helper adds **zero trainable parameters**; Stage-2A still instantiates 20,354. The auxiliary loss is mathematically computable for the `no_workspace` control using its otherwise-unused predictor matrices, allowing future matched-loss comparisons. This is not the same as active-parameter parity: effective gradients remain dependent on architecture and objective.

## What the CPU unit tests prove

- Frozen geometry, finite nonnegative per-position errors and positive finite mean objective on a tiny deterministic structural fixture.
- `L_aux.backward()` gives finite nonzero gradients to both existing predictor matrices but no answer-head gradient.
- In synthetic independent local hidden states, target-position gradients are cut off by detach while earlier input-side gradients remain.
- Future token changes within a WRITE event chunk do not modify earlier causal local outputs or earlier prediction errors.
- Edits to delay chunks or the READ suffix never change the auxiliary loss.
- `hybrid` and `no_workspace` with identical initialized parameter values have the same auxiliary objective; the no-workspace predictor receives auxiliary gradients when deliberately evaluated.
- Original answer-only cross entropy still yields NO predictor gradients. The frozen Stage-0 surprise and routing logic are not changed.
- Malformed event prefix, missing final READ/QUERY, wrong anchors, token IDs and hidden geometry fail closed.

`backward()` in tests checks mathematical derivatives only; there is **no optimizer, no gradient-based parameter update, no checkpoint**, and CPU-only tensors. Test fixture initialization seeds are NOT scientific experimental seeds.

## Critical scientific gates

The following remain **unresolved and unauthorised**: choice of auxiliary-loss multiplier `lambda`, whether optimization improves retrieval, active-gradient parameter matching, equal objectives/compute across Transformer and all PGW routing arms, actual training seed/reservation, exact complete corpus fingerprints, train/test leakage checks, three-seed paired validation, and preregistered superiority thresholds. Any later learned improvement must beat an always-NOT_FOUND 50% shortcut on **present-only and by-cell** accuracy, plus capacity-matched controls. Four workspace slots have substantial collision risk.

Passing original exact-head CI and merged-main CI classifies this only as `PGW_V3_STAGE2B0_AUXILIARY_GRADIENT_PATH_READY`. It is **not** a learned predictive system, memory breakthrough, language-model NLL advantage or evidence of transformer superiority.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`scale_up_authorized=false`  
`breakthrough_claim_allowed=false`
