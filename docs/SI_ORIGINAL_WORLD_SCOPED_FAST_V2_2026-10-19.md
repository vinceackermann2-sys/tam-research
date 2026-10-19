# Original user SI architecture — V2 world-scoped factual fast adaptation

**2026-10-19 — engineering correctness only. No science, GPU, heldout or superiority claim.**

## Why this change is necessary

The user's screenshot-specified SI system includes **hierarchical persistent memory, online fast-weight adaptation and continual learning across experiences**. The integrated V1 (draft #1523) has world-scoped *retrieved* memory, but its two fast-weight tensors are **global**. `ingest_executed(..., world="A")` mutates the same `fast_u/fast_v` used during a later `forward(..., world="B")`. That is a real cross-world adaptation leak, even though the V1's retrieval store itself is scoped.

This successor follows the first limited crossmodal learning pilot (draft #1525) **without changing any V1 source, v26 scientific evidence, PCMA negative findings or other project**. Original canonical `tam_research/si_original_hybrid_v0.py` stays Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`.

## Actual correction

New `SIOriginalWorldScopedV2` subclasses the existing integrated V1. For every explicitly named world it keeps an independent rank-two `(u,v)` adapter and update count. Causal executed feedback updates ONLY that world's tensors. The V1's original buffers are restored after ingestion and remain neutral as the seed for new worlds. `forward(world=...)` selects only that world's fast state; V1's already scoped factual retrieval is unchanged.

- Tests compare the SAME public input and shared frozen model weights before/after A's real reward: B's fast-enabled state must equal B with the fast path disabled; A must change.
- After B's own feedback, A's state must remain bit-identical. Public feedback with wrong provenance, illegal reward or invalid temporal sequence must never create an adapter entry.
- State export/import is versioned, bounded to at most 32 distinct world keys, rejects nonfinite tensors and mismatched per-world/global update totals, and restores both worlds' outputs.
- All changes are **additive**: `tam_research/si_original_world_scoped_v2.py`, `tests/test_si_original_world_scoped_v2.py`, this document. The V1 source and original 64,614-parameter v26 negative evidence are never rewritten.
- Model and replay remain **sequential/not thread-safe**. World names are supplied by a trusted caller, not authenticated receipts. The backbone itself remains shared; this fix does **not** claim cross-world privacy of learned neural weights or correct continual learning.

## Verified first integrated pilot (not scientific evidence)

On draft #1525's completed original CI attempt **39314299334**, 96 developer CPU optimizer updates per architecture trained the *actual integrated V1 backbone + five input encoder paths* to identify which earlier public sensor cue was requested by a motor token. The 256 developer-validation examples were disjoint from training generator IDs. Hybrid validation **0.515625** versus Transformer-with-same-other-components **0.45703125**; class chance **0.25**, trivial hand-coded decoder **1.0**. Hybrid accuracy by queried text/image/audio/video: **1.0 / 0.20339 / 0.29825 / 0.47887**. This is uneven trainability, NOT native multimodal perception, statistically significant superiority, or a demonstration of a successful integrated SI system. **62,366** vs **45,926** parameters; *active training FLOPs NOT matched*. CI targeted **24 passed** and full **3,567 passed, one skipped** on parent.

## Next full-architecture evaluation gate — not executed here

To achieve the ORIGINAL request, the next experiment must use a preselected independent real action-feedback environment and a trainable joint path including multimodal encoding, factual world dynamics, uncertainty-gated search, physically adaptive iterations, long-term memory, world-scoped fast adaptation, replay and sequential skill retention. No component may be silently disabled and then represented as tested. Compare to a similarly equipped Transformer and strong cheap memory/planning controls with matched active train/inference compute and independent episode budgets. Explicitly measure:
1. Executed return, cost of information, long-horizon planning accuracy, reward calibration, search expansions, latency/FLOPs and performance by modality, across separate independent worlds.
2. Continual learning and forgetting after sequential new tasks, persistence across process restart, action-conditioned transition accuracy, true vs learned halting utility, expert-route utilization, and reset/memory/replay/fast/search/verifier ablations.
3. Disjoint development and protected heldout worlds; no look-ahead, no unverifiable claims of public feedback, no counterfactual reward being logged as executed, no repeat science seeds, prior frozen STOP/GO threshold and confidence interval.

The V1 `ingest_executed` API currently accepts a **caller-asserted provenance string**. This is NOT a signed evaluator receipt. A scientifically trustworthy external worker/receipt boundary must be built and audited before real heldout experiments. The V1 planning tree's learned dynamics, verifier and halting rule are not yet performance-validated. Model replay is NOT a demonstrated protection from forgetting. These are **known unsolved requirements**, not solved by world-scoping alone.

**Decision:** V1 cross-modal training had limited success on the source-authored task; V2 repairs one important causal fidelity issue. The *complete user architecture remains scientifically UNTESTED*, and neither the v26 NO-GO proxy nor the V1 development result determines its real verdict. Do not authorize paid compute, new scientific seeds, main merges or protected evaluation from this document.
