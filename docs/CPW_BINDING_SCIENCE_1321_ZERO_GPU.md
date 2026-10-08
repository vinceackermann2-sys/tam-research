# Corrected CPW five-arm harness — CPU gate (#1321)

Parent *preauthority-only* protocol: #1319. **No paid science is authorized.**

This package selects five existing models without modifying them:
Transformer (24,940,288 params), sequence-only (21,745,408),
AFM-last1 (21,721,344), AFM-first1 (21,721,344), R1-final
(21,745,408). Only the AFM pair and the sequence/R1 pair
are parameter-matched. One narrow causal attention layer in R1 is quadratic
in context length, independent of its trainable parameter count.

## Evaluation integrity

- Canonical generator: `tam_research.cpw_binding_v2.task`, without edits.
- One freshly constructed CPU `torch.Generator` per independent evaluation
  bucket; caller-provided `eval_seed` is explicitly re-used in each arm.
  Train seed is separately provided to the CPU step integrity test.
- Query-only logits. `predict(tokens)` is the entire inference interface;
  delay, targets, pair indices, query slot and ground-truth metadata are never
  passed to the model.
- Per-distance held-out accuracy, CE and input SHA256 on 32/64/128/256.
- Separate counterfactual four-query evaluation. Each group preserves the
  entire source prefix and changes only the final queried key. Record both
  per-query accuracy and all-four-correct **source-group** accuracy. The
  key-aware symbolic oracle must score 100%; the fixed-slot key-blind
  negative control must score 25% per-query and 0 all-four-groups.
- `evaluation_wall_seconds` is an observation hook only, **not** scientific
  training seconds, H100 billable GPU time or full-run Pareto evidence.
  GPU peak-VRAM and throughput/total compute require a separately
  authorized run-control implementation.
- Tiny CPU one-step gradient smokes are only implementation checks, never
  evidence that memory was learned.

The current package deliberately has **NO H100 runner, Modal workflow,
permanent scientific seeds, scientific RESULT_ROOT, retry or continuation
policy**, because #1319 requires those identities to be frozen by a later
explicit one-shot authorization. Do not infer available credits from prior runs.

Until a fresh paired Transformer control is actually trained on corrected
binding and passes the preregistered validity gate, prior Transformer 1.0
scores on the shortcut-affected old generator establish nothing about
learnability here. Synthetic recall alone is not evidence of general language
model superiority or chat ability.
