# PGW-v3 Stage-2B3 — global-position baseline and static operation accounting

**Scope:** strictly CPU-only structural engineering. No optimizer, no training, no scientific seeds, no GPU/Modal. Frozen source-bound contract [issue #1411](https://github.com/vinceackermann2-sys/tam-research/issues/1411) and predecessor [design #1409](https://github.com/vinceackermann2-sys/tam-research/issues/1409).

- Original Stage-2B2 baseline: issue #1400, merged PR #1406 at `4f1e8e38be1c23e374a1a1194e9092ed8e72fd73`, first-attempt original PR #38068336757 and merged-main CI #38068661156 PASS. Original failed head #1403 is immutable and never rerun/amended.
- Contract issue initially audited main `2e624ca67e3274da397b625e1b5debe63c195a79`; branch `research/pgw-v3-stage2b3-position-accounting-v1` created from live main `14a01d5c435e303cc6df60532b35fc0faf00fb12` following unrelated CORTEX/reduced-attention file additions.
- Original PGW-v2 mechanism #1181 and PGW-core #1201/#1213 negative classifications remain immutable.
- No historical Stage-0/1B/2A/2B0/2B1/2B2 code, scientific scripts, CI, other research paths or baseline results changed.

## Fixed absolute positions on the full causal reference

The new `GlobalPositionCausalReference` inherits the frozen Stage-2B2 `CausalReference` with `attention_scope="full"`. It overrides **only** input embedding assembly, adding a parameter-free float32-constructed deterministic positional sinusoid over full-sequence indices p=0..T-1:

`PE(p,2i)=sin(p·10000^(-2i/32)), PE(p,2i+1)=cos(p·10000^(-2i/32))`.

Its input tensor is `existing_token_embedding + existing_repeating_within_chunk_position_embedding + PE(global_p)`. Thus the old baseline without global PE remains a separate unchanged positional ablation, and the chunk-only reference remains a no-cross-chunk negative control.

No learned positional vectors, parameters, buffers, random draws, loss changes, data labels or oracle outputs enter the new model. **Exactly 20,347 instantiated parameters**, versus historical PGW Stage-2A **20,354**: a seven-parameter difference (0.0344% instantiated), not matched gradient-active model capacity. The original 512 predictor weights still receive auxiliary gradients via the unchanged frozen Stage-2B2 target-detached per-event loss.

A repeated token at the same within-chunk offset in two distinct chunks has equal input embeddings under the historical no-global-position reference, but distinguishable representations under the global-position treatment. This addresses a **representational confound**; it neither proves the untrained Transformer can solve last-write retrieval nor guarantees a trained model will learn the correct temporal relation.

## Strict causal validation

CPU tests exercise three complete Stage-1B delays (1,2,4), present/absent query keys, full-sequence attention future-mask invariance at QUERY, gradients from a later READ answer to earlier completed events, unchanged auxiliary predictor semantics, no predictor gradients from answer-only loss, strict grammar/anchor rejection, and unchanged model state after derivative probes. All three original Stage-2B2 reference and old PGW implementations are imported untouched.

## Analytical operation counts — limited scope

`attention_operation_counts` exposes exact symbolic structural quantities for a batch B, time T divisible by chunk8, attention heads H=4, width32, four PGW workspace slots and two selected writes/chunk:

| Operation | Full causal reference | Chunk-only and PGW local encoder |
|---|---|---|
| Dense score and value attention pair ops | `2·B·H·T²` | `2·B·H·(T/8)·8²` |
| Ideal causally useful pair ops | `B·H·T·(T+1)` | `B·H·(T/8)·8·9` |
| Dense score+AV multiply-accumulates | `dense pair ops · (32/H)` | `dense pair ops · (32/H)` |

PGW's additional **algorithmic** memory work includes `B·(T/8)·2` selected event-write calls, `B·T·4` token/read-slot comparisons and `B·(T/8)·2·4` write-event/slot comparisons. A READ is computed in every processed chunk in the frozen hybrid workspace implementation, even though the first chunk's read output is deliberately suppressed. The last READ chunk's writes happen after its output, by construction. This table is an **algebraic structural audit**, not a profiler measurement.

These counts deliberately **exclude** token embedding lookup, QKV projections, MLPs, layer normalizations, softmax, numerical kernels, bandwidth, autoregressive step behavior, allocated temporary tensors, backprop, measured latency, peak memory and aggregate training FLOPs. A dense masked attention implementation may still compute masked pairs. No equal-compute, equal-walltime, equal-active-parameter or efficiency result can be derived from this table alone. The full reference grows quadratically in dense attention pair count; the PGW workspace also has sequential overhead that is not represented by only the local-attention column.

## Scientific gate remains closed

Before *any* CPU training: separate preregistration of complete train/val/test split hashes and sample leakage audit, position-fair reference treatment selection, analytically and empirically gradient-active capacities, measured forward/backward FLOPs, latency and peak memory on the same CPU, equal auxiliary loss coefficient and objectives, optimizer/schedule, unique durable reservations for >=3 independent paired scientific seeds, hard budget limits/no rerun, and preregistered present-only latest-key accuracy and per-cell margins over the 50% always-NOT_FOUND aggregate shortcut. H100 remains a separate explicitly authorized paid gate.

Passing source-head and merged-main CI means `PGW_V3_STAGE2B3_GLOBAL_POSITION_STRUCTURAL_REFERENCE_READY` **only**; no trained superiority, no retrieval breakthrough, no NLL evidence.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`scale_up_authorized=false`  
`breakthrough_claim_allowed=false`
