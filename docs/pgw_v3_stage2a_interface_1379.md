# PGW-v3 Stage-2A — causal token-to-workspace answer interface

**Classification:** `PGW_V3_STAGE2A_CAUSAL_INTERFACE_STRUCTURAL_ONLY` — CPU structural tests, no learning.

| Evidence | Immutable referent |
|---|---|
| Research track | PGW only, `vinceackermann2-sys/tam-research` |
| Stage-2 design | [#1378](https://github.com/vinceackermann2-sys/tam-research/issues/1378) |
| Frozen Stage-2A implementation contract | [#1379](https://github.com/vinceackermann2-sys/tam-research/issues/1379) |
| Stage-1B oracle integrity | [#1374](https://github.com/vinceackermann2-sys/tam-research/issues/1374), PR #1376, merged-main CI `37899805966` PASS |
| Stage-0 workspace | [#1359](https://github.com/vinceackermann2-sys/tam-research/issues/1359), PR #1363 |
| Source main | `875c5e5cc583494d2248587d157e9e98aa738ec0` |
| Branch | `research/pgw-v3-stage2a-causal-interface-1379-v1` |

Prior PGW-v2 mechanism issue #1181 and core scientific #1201/#1213 remain **UNSUPPORTED**; none of those modules, seeds, results, workflow or research branches are mutated by this PR. This is not the separate CPW research architecture.

## Frozen, testable architecture (untrained)

Input: `tokens LongTensor[B,T]` with complete chunks of length 8, vocabulary 256. Query anchor: the `QUERY` token at offset 2 in a final `READ` chunk, delivered separately as `anchors LongTensor[B]`. External target is one of token IDs 32..64 inclusive (32 values + NOT_FOUND), mapped to classification target `target_token - 32`. Nothing injects the target into input.

Encoder: learnable embedding `[256,32]` plus a learnable eight-position embedding `[8,32]`, reused for every chunk. All chunks are processed independently through **one shared causal masked 4-head multihead self-attention block**, model width 32, dropout 0, residual LayerNorm, FFN 32→64→32 (GELU) and a second residual LayerNorm. Mask disallows attending to strictly future tokens *within* a chunk. There is no unrestricted global-token attention.

Memory: import the **unchanged** Stage-0 PGW-v3 `UtilityAddressedWorkspace` configured for 4 key/value slots and 2 selected events per 8-token chunk. It reads old slots **before** writing selected events from the newly completed chunk, so READ current chunk cannot access its own later tokens by workspace updates. The local and workspace residual are added, LayerNorm applied and the selected QUERY token read through a linear 32→33 answer head.

Frozen instantiated model size: **20,354 parameters** (8,192 token embedding + 256 local position embedding + 8,544 causal local block + 2,209 Stage-0 workspace + 64 output LayerNorm + 1,089 classifier).

Routing controls: `hybrid`, `utility_only`, `surprise_only`, `recency`, `fixed_random`, `no_workspace`. They have identical *instantiated* parameter counts but NOT identical effective/gradient-active counts. No active-parameter matching or fair learned comparison is claimed.

`batch_verified_examples` accepts only already-frozen Stage-1B samples; it recomputes SHA256 over input bytes, invokes an independent causal latest-write oracle, validates the external target and the final query anchor, forbids mixed-length batches and returns only `tokens, anchors, targets`. No oracle metadata or labels become model features. Different delay lengths require explicit separate batches; no future/padding leakage.

## Original CPU structural tests

1. Exactly 20,354 instantiated parameters in all six modes.
2. End-to-end tensors `[B,T]`→`[B,33]`, with NOT_FOUND in class index 32.
3. Changing a future READ-chunk filler token *after the QUERY anchor* cannot change its logits.
4. With `no_workspace`, changing an earlier completed chunk must not change final QUERY logits.
5. With enabled workspace, changing a selected token in a completed chunk must change final QUERY logits, proving a nonzero **mathematical cross-chunk path**.
6. A single CPU cross-entropy `backward()` for an answer loss must yield finite nonzero gradient in enabled workspace query/write projections and learned-utility scoring; it must yield **no gradient** in the detached predictive surprise subnetwork. `no_workspace` must yield no workspace parameter gradient.
7. Malformed shapes, invalid IDs/query anchors, inconsistent externally supplied labels, stale fingerprint and mixed-delay lengths must fail closed.

Test `backward()` is a **one-off mathematical derivative probe**, not an optimizer, checkpoint, training attempt, or scientific seed. The model is randomly initialized in tests and **no accuracy metric is interpreted**. No active-param fairness conclusion can be inferred from exact total parameter counts.

## Gate beyond Stage-2A

Even after original exact-head and merged-main CI pass, this stage establishes only `PGW_V3_STAGE2A_CAUSAL_INTERFACE_READY`. It does not show *learned* key-value retrieval, a useful predictor, active-parameter parity, language-model NLL gains, Transformer superiority, or a breakthrough.

Before any optimization step, a **separate frozen Stage-2B scientific protocol** must specify training objective including a trainable predictive target/auxiliary loss; matched auxiliary-loss controls; full train/eval/test corpus manifests and sample hashes; at least three fresh scientific seeds, unique durable attempt IDs, CPU budgets and hard stops; parameter AND active-parameter counts; fair Transformer/chunk-only/PGW ablations; frozen quantitative thresholds and equal-compute evaluation. Paid H100 experiments require separate explicit authorization regardless of CPU outcomes.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`scale_up_authorized=false`  
`breakthrough_claim_allowed=false`
