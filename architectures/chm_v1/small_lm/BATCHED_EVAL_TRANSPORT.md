# CHM-v1 batched evaluation transport

Classification: `ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY`.

Scientific authority remains issue #854. This file authorizes no GPU, Modal, paid execution, seed allocation, scaling, or scientific claim.

Base implementation merged at `08d7fdc81981313472de28f492ea1e67867656aa`.

## Why this exists

The exact reference path intentionally crosses between model-device tensors and the NumPy correctness index. The scalar implementation performs that transfer once per token per retrieval hop. With two retrieval hops and long held-out probes, the number of tiny synchronizing transfers can dominate wall-clock without changing the scientific question.

`tam_research/chm_v1_batched_eval.py` coalesces only those transfers:

1. Each token's query projection is still computed independently with the same scalar tensor shape as the merged reference path.
2. The resulting query vectors for one session chunk are transferred to CPU together.
3. Exact flat/index search still runs one query at a time, in the same order, using the same index implementation and tie contract.
4. The selected exact values are transferred back to the model device together.
5. Each token's memory integration is still computed independently with the same scalar tensor shape as the reference path.
6. Hop 2 is formed only after hop-1 integration, preserving the frozen sequential two-hop dependency.
7. Current-chunk memory writes still occur only after logits are complete.

## Merge gate

The batched transport path must remain separate from the scalar reference and may be used by a future scientific runner only if ordinary zero-credit CI demonstrates:

- identical retrieved item IDs and positions for flat and indexed modes;
- identical address-vector-read and directory-node accounting;
- bit-identical CPU logits on the same model, state, and tokens;
- identical post-logit memory writes;
- no model-parameter mutation.

Timing counters are not expected to be numerically identical because reducing transfer/synchronization overhead is the purpose of this path.

Any failure of the equivalence checks blocks use of the optimized path and does not count as scientific evidence.