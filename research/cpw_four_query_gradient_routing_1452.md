# CPW #1452 — Four-query source-gradient routing matrix (random initialization only)

**Scope:** additive CPW experiment branch only (`experiment/cpw-v0-brain-predictive-workspace-20261005`); no writes to `main`, no CHM/PGW/RLT/reduced-attention changes, no trained checkpoints or paid compute.

## Frozen inputs and diagnostic contract

Start from verified CPW merge `acc14276e33c74ef13e2dd8b962f19d4c5de8135`, original issue #1335 random-init one-query CPU gradient audit. Frozen sources:

- `tam_research/cpw_binding_v2/task.py` blob `79dc0ceda617985eba40f119343bdb2992aa395d`
- `tam_research/cpw_binding_science/harness.py` blob `63ab35fc82ea06d91ce2a9b5cbd1d2d9c56baf3f`
- `tam_research/cpw_binding_grad_audit/audit.py` blob `49c6154e48fce7edace97ad630783153d39a6b0d`

New TRAIN-engineering source generated with **fixed engineering seed 1452001** (not a scientific seed). One corrected binding sample contains eight unique random key→value pairs at frozen delays 32,64,96,128,160,192,224,256 and length 320. Four counterfactual versions query delays 32/64/128/256 while retaining exactly the same 319-token prefix. The corrected oracle must return four distinct answers and a query-blind fixed-value policy succeeds exactly **one of four** queries.

Keep five frozen models unchanged: `transformer`, `sequence_only`, `afm_last1`, `afm_first1`, `r1_final`. For each arm, instantiate the model **once** with the same frozen engineering initialization, disable stochastic dropout via `.eval()` without disabling autograd, and calculate query-only cross-entropy gradients for each four-way counterfactual separately. **Zero optimizer updates.** The same model parameters, entire source prefix and physical source-token positions persist across four measurements. Build a 4×8 activation-gradient matrix at source KEY and VALUE positions. Log requested query delay/key/answer, selected source positions, target/distractor norms, normalized target share, distractor median/max, total-variation change between pairs of query-conditioned normalized eight-slot distributions, zero-gradient indicators, last-15 context and final query-token gradient.

Strict structural checks before any claims: CPU tensors; exact corrected key lookup 4/4; unique targets and queries; source prefix SHA256 shared across all arms; frozen expected model parameter counts; finite nonnegative gradients; local sequence-only has zero far-source gradients at all eight source pairs. A zero vector produces **null** normalized concentration rather than divide-by-zero or a false selectivity win. No user-facing result before original first source-locked run; no manual rerun or changed seed after results.

## Execution and scientific ceiling

Regular PR CI uses source-contract tests and a sequence-only CPU routing probe, never a trained model or a heldout science set. A **unique path-only first-merge** workflow on the canonical CPW branch checks source and test Git blobs, uses CPU-only PyTorch, runs one complete five-arm diagnostic, validates one JSON marker and archives raw 4×8 matrices for independent inspection. Exact original first-attempt PR-head CI PASS, live branch audit, expected-head pinned merge, original canonical branch gate and first-attempt diagnostic run are required. Failed/cancelled heads and original scientific runs remain frozen and must never be rerun/amended.

This is a random-init gradient *path/routing* diagnostic. Even if early AFM shows query-specific source gradients, this alone does **NOT** establish learned binding, useful recall, natural language skill, matched-transformer superiority, 100M scalability or a breakthrough. Next informed gate would be a separately preregistered bounded CPU **trained** learnability comparison under equal inputs, supervision and compute, then independent heldout replication. Original #1319 scientific comparison remains PREAUTHORITY; no GPU, Modal, Colab, new scientific seed or paid job is authorized.
