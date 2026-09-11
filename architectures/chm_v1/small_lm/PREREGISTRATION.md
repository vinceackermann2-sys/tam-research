# CHM-v1 / EIEM small matched-LM gate

Authoritative preregistration: GitHub issue #854.

Classification: `PREREGISTRATION_ONLY_NO_GPU_AUTHORITY`.

Frozen scientific base: `91b02ec575440b42050377351a6e404139632e22` (merge of PR #852).

This implementation branch was forked directly from that commit.  This file is a local implementation record; if wording here ever conflicts with issue #854, the issue is authoritative and the scientific run must remain blocked until the conflict is resolved before observing a seed result.

## Frozen comparison

- `LOCAL`: bounded-local causal Transformer, 512-token dense window, no episodic memory.
- `EIEM-FLAT`: same bounded-local backbone, learned normalized query/key address projections, exact episodic values, exhaustive flat retrieval, learned memory integration.
- `EIEM-INDEXED`: the same trained EIEM checkpoint and episodic state with only flat search replaced by the zero-parameter exact branch-and-bound index.  It is never separately trained.
- Indexed and flat retrieval share distance, tie-breaking, selected item/value, integration, checkpoint, and state.  Any discrete mismatch is a gate failure.
- No oracle task/event/key ID may enter model or index input.  Monotonic insertion IDs used solely for deterministic equal-distance tie-breaking are storage metadata, not task labels.

## Frozen fairness

- scientific paired seeds: `8611`, `8612`, `8613`;
- approximately 25M trainable parameters, LOCAL vs EIEM-FLAT within 1%;
- GPT-2 vocabulary 50,257;
- exact immutable repository FineWeb-Edu corpus/validation bytes;
- 8,388,608 training tokens/model;
- AdamW betas `(0.9, 0.95)`, weight decay 0.1, peak LR `3e-4`, cosine decay, 2% warmup, grad clip 1.0;
- bfloat16 on GPU and identical global token/batch accounting;
- dense local working window 512 tokens;
- evaluation evidence must lie beyond 512 tokens;
- episodic state resets at session boundaries and is never shared across examples.

Scientific seeds are reserved.  CPU smoke/development must use a separate explicitly non-scientific seed.

## Full-pass gates

All are required across the three paired scientific seeds:

1. EIEM-INDEXED matches EIEM-FLAT on 100% of evaluated discrete retrieval calls.
2. EIEM-FLAT beats LOCAL by at least +10 percentage points mean accuracy over rare-fact, overwrite, and two-hop, and wins each family in at least 2/3 seeds.
3. Overwrite stale-value error is at least 50% lower relative than LOCAL, unless LOCAL is already below 1%, in which case EIEM may be at most +0.5 pp worse.
4. Mean FineWeb-Edu NLL regression is <=+0.10 and no seed regresses >+0.15.
5. Local/no-memory-needed accuracy regresses by <=2 pp on every seed.
6. At largest preregistered memory size, exact indexed address-vector reads are <=25% of exhaustive flat reads; directory traversal is reported separately.
7. No NaN/Inf, cross-session aliasing, hidden persistent state, future/oracle leakage, or inference-memory mutation of base parameters.

Required evaluation families: held-out FineWeb-Edu NLL/PPL, rare fact, overwrite/stale state, two-hop remote retrieval, and local/no-memory-needed negative control.  Hidden generator metadata is scoring/training metadata only, never model/index input.

Required ablations include memory disabled/reset, shuffled/wrong-session memory, random untrained address projections, exhaustive flat retrieval over the same learned addresses/state, and capacity/distance slices.

## Execution boundary

This branch may contain implementation, tests, and ordinary zero-credit CI only.  It must not contain a scientific trigger.  No Modal/GPU allocation or paid compute is authorized by #854.  After a green reviewable implementation is merged, any scientific run requires a separate run/trigger issue citing the exact merged SHA, GPU class, and budget.  No automatic retry is permitted; a seed that reaches GPU allocation is consumed.

A full small-LM pass would be evidence only for a reproducible small-LM long-memory signal plus exact sparse inference lookup in this setup.  It is not a novelty, SOTA, production-speed, scaling, AGI, or breakthrough claim.
