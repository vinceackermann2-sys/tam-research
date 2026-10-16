# BE checkpoint forensics — REVIEW DRAFT ONLY (2026-10-16)

**NOT registered, not authorized for inference.** Do not use as a launch trigger.

Frozen BC source commit: `6dd09e019a65ea307f9027f6aabd4f31b11a5b3c`. BC run `38799067479` consumed; do not retry/reuse seeds. Three case SHA256: case0 `e962f3b96ff7529d9ac831b16f5db4c1e448413c6794a0c84f16800773934e4d`, case1 `cddabe44dd98f5df47dc18e2f63e997822ecd5d1536449dc86f309e2e6690b6e`, case2 `9b89eda7de641638d6c5605a032e846dec9d2e8923f6c02213aa4f095cf6030b`.

Preflight-only: source-bound hash verification, 27,712-parameter model reconstruction, target oracle checks, no output collision, one-shot manifest, unique fresh eval seed and ID; no seed from frozen BC, protected 58232/58233, or model-free preflights 314159265,271828183,161803399,173205081. Before **one CPU-only inference attempt**, original CI green and durable source-bound unique reservation required. Zero retries/resumes; no GPU/Modal/Colab; preserve all negative outcomes.

Primary naturalistic Bernoulli0.08 earlier-SET balanced holdouts, trained gaps16/64/127, 128/class/gap, all three checkpoints: per-class accuracy, 2-logit NLL, margins, paired forced-final-SET label flip, and reset recurrent carry only before final chunk. Separate final gap1/4 positive controls.

Secondary selected-stratum causal value interventions: flip only the value of an already-existing earlier SET, never insert SET; preserve write positions/count, target and suffix. Same-row local-vs-remote matched comparison picks original SET values equal, contrasts target-margin change under local versus remote flips. Report eligible denominators/class composition and temporal-distance caveat. BM model-free preflight with seed173205081 checked 256 examples/gap: gap16 matched128 (52 class0,76 class1); gap64 matched86 (41/45); gap119,120,127 unmatched; no model was evaluated.

Tertiary clean data/forced-insertion and gaps outside BC-trained {1,4,16,64,127} labeled OOD, never pooled into primary result. State/gate traces are mechanistic descriptions, not proof. Source-bound immutable results + hashes on reviewable PR, not main/unrelated lanes. Synthetic success cannot justify breakthrough; GO needs independently replicated fair language NLL and compute-efficiency improvement vs strong Transformer, STOP robust preregistered negative evidence; otherwise INCONCLUSIVE.

This is a short review marker; fuller local draft: `rlt_be_bm_checkpoint_forensics_review_draft_20261016.md` (not committed).