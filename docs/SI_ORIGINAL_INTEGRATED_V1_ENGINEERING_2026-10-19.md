# User's original screenshot SI architecture — integrated V1, first execution gate

**2026-10-19 · ENGINEERING INTEGRATION ONLY · NOT a scientific efficacy test or breakthrough.**

This branch implements a **functional, small, integrated vertical slice** of the original screenshot architecture. It directly follows the correction in draft PR #1522: old v0/v26 hybrid and PCMA tests were incomplete **proxies**, not conclusive tests of the original full system. This branch does **not** reverse the original v26 proxy's negative evidence, consume a scientific seed, merge anything, or authorize paid compute.

## Source of truth and exact engineering decisions

The user supplied **functional architectural requirements**, not exact unique tensor widths, initializations, losses, layer counts or task distributions. Those decisions remain implementation choices, not user-provided specifications. This prototype's choices, deliberately bounded for CPU integration, are: width 48 by default, four interleaved blocks, four sparse top-1 experts per block, 4 attention heads, local attention window 12, up to three learned-halting latent loops, two-step exhaustive four-action planning tree (20 candidate model edges per invoked query), capacity 128 verified memory transitions and 128 replay entries, rank-two online fast weights. Test fixture width 16 does NOT establish scale fitness. Five modality inputs have **real learned encoders** but tiny synthetic/public 8×8 images, two-frame 8×8 video and 32-sample audio; there is NO claim of native perceptual or language understanding.

The exact unchanged original source `tam_research/si_original_hybrid_v0.py` retains Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. The new integrated code imports its real `SIHybridBlock` so its alternating attention+gated-state recurrence and actual sparse dispatched expert are preserved, not reinterpreted as a pure Transformer. All other new mechanisms are in `tam_research/si_original_integrated_v1.py` with tests `tests/test_si_original_integrated_v1.py`.

## Full system data flow

```text
public typed inputs [text IDs | RGB image | audio waveform | video frames | motor vector]
  -> distinct learned encoders + modality + causal positions
  -> interleaved original local causal attention / gated recurrence blocks
     + actually dispatched top-1 MoE per block
  -> verified per-world long-term fact retrieval (only past executed feedback)
  -> bounded rank-two fast-weight residual updated from real reward
  -> optional skill adapter (dedicated small trainable module)
  -> GRU latent recurrent refinement with per-sample computed stop decision
  -> calibrated-in-training verifier uncertainty head
  -> direct action or (if uncertain) bounded depth-2 action-conditioned
     learned latent transition/reward/value search
  -> executed action in external world (NOT simulated here)
  -> explicit public observed action/reward/next-observation ingestion
     -> versioned bounded factual memory + online fast update + replay
  -> offline replay consolidation: action-conditioned dynamics + actual reward
     + verifier calibration from observed prediction error
```

The loop in this code is **not connected to a robot or real web/software action executor**. The environment must supply verified public action/feedback using an explicit API. No unexecuted reward or privileged simulator state is ever requested by the model, but provenance is an **honest caller-supplied string rather than an authenticated cryptographic receipt**. A trusted evaluator sandbox and fully separated heldout generator are required before scientific deployment.

### Fidelity matrix: what V1 actually delivers

| Screenshot functional requirement | Integrated V1 evidence | Remaining limitation |
| --- | --- | --- |
| Hybrid causal attention + efficient recurrent path | Imports `SIHybridBlock` with 4 interleaved original mixers | Numerical causal smoke only; no long-context scaling results |
| Sparse expert routing | Four actually dispatched MoE experts per block; route indices and balance signal exposed | No learned expert specialization, load efficiency or balanced large-scale training yet |
| Adaptive latent compute | A GRUCell is iterated **1..3 times**, with early stop physically skipping subsequent recurrent work; tests force high vs low halting and show actual different counters/states | Halting threshold .5; no trained budget policy or quality-compute Pareto |
| Search when needed | Verifier threshold **gates real depth-2 tree** of learned action-conditioned transition/reward/value predictions; tests require 0 vs 20 expansions | World model is not yet trained enough for trustworthy multistep imagination, no MCTS |
| Working memory | Public sequence jointly processed by causal local attention and recurrent state | Model context ≤64 tokens; no production attention-cache paging |
| Fast weights updated during use | Rank-two low-rank buffers **change on actual reward** and affect subsequent forward state; explicit ablation | Handcrafted bounded SGD-style update; not Titans/test-time trained optimal |
| Long-term memory and consolidation | Versioned per-world verified transition deltas are **read in forward**, bounded, serialized/reloaded; factual replay trains latent dynamics/reward/verifier | External persistence requires caller to save/export; no automated secure store or learned selection policy |
| Predictive latent world and planning | Action-conditioned latent transition + reward/value, used for real branching search | No demonstrated rollout accuracy, intrinsic JEPA-style joint embedding loss, or Dreamer-scale training |
| Continual skill learning/replay | New named task adapters can be added and separately optimized, while original backbone stays fixed; verified replay performs gradient updates | No full sequential multitask train/eval, lifetime retention metrics or anti-forgetting claim |
| Real multimodal pathways | Distinct text embedding, CNN image, 1D CNN audio, 3D CNN video and motor projection **all backpropagate** through shared reasoning | Toy resolution and no real modality dataset or multimodal skill learning yet |
| Self-evaluation/verifier | Uncertainty head actively chooses search vs no search, and replay uses observed prediction-error calibration target | No calibrated uncertainty guarantees, external learned verifier or hallucination safety proof |
| Entire integrated system | All above components have active connections and targeted correctness/causality tests | **No completed end-to-end integrated training, no fair Transformer comparison, no SI result** |

## What tests establish, and what they CANNOT establish

Branch CI runs `pytest -q tests/test_si_original_integrated_v1.py` **before** the repository-wide test suite. The tests assert actual gradient flow from each modality encoder, exact original interleaved source blocks, four-expert routing, early-exit changing actual latent-step count, verifier-conditional action search with 20 real imagined edges, memory's effect on hidden representation after only witnessed feedback, world isolation and finite capacity, rank-two fast residual changing after one factual reward, replay changing trained world-model/verifier parameters, new skill adapter changing output while original backbone weights remain fixed, persistent serialization/rehydration, and rejection of forged/not-yet-executed feedback. Test IDs are developer-only fixture integers, not previously consumed science evidence.

**Important falsifier:** Passing these tests proves only that each proposed block is *functionally connected*. It does NOT demonstrate that any block improves learning versus removal, that search generates good plans, that the verifier is calibrated, that the learning rule prevents forgetting, that sparse routing is faster or that any combination beats a Transformer. The next stage should **not** jump from integration smoke to breakthrough wording.

## Required experimental route to a valid verdict for the user's full design

1. Freeze real model-training objective, actual input/output tasks and sensor/action spaces. Choose source-level faithful, **trainable** pathways rather than hardcoded demonstration logic. No new toy task tuned after a negative outcome.
2. Show joint **end-to-end training** of all implemented components (including halting, verifier, action-conditioned rollout and fast memory use) on preregistered **development** data with control of leakage and meaningful data scale. Use public objective feedback only. Probe memory, latent compute and policy gradients over real trajectories.
3. Test concrete separate mechanisms: MoE/dense, local recurrent vs full attention, fixed/variable latent reasoning, search/no-search, persistent memory on/off, fast adaptation on/off, replay/no-replay, verifier/no-verifier, modality-specific vs shared encoders. Report both positive and negative component results.
4. Compare the same full system with a fair **compute/data-matched Transformer**, one with comparable memory/search/adaptation, and strong cheap retrieval/classical planning/finite-state controls. Match **actual forward/backward FLOPs, training tokens/samples, walltime and inference budgets**, not nominal model size or optimizer-step count alone.
5. Only after success on development and independent source/attempt audit: preregister disjoint independent heldout environments, scientific seeds not in consumed ledger, blind release protocol, confidence bounds, success margin and **STOP** condition. No repeat of consumed v25/v26 seeds, no protected eval without approval.
6. If time or authorized compute cannot support faithful multimodal/world/planning and continual-learning training, report **UNTESTED / resource-limited**. If the fully integrated trained system loses fair confirmatory tests, record an actual **candidate NO-GO**.

### Current scientific status

- **User's exact screenshot-level functional architecture:** this PR begins its integrated implementation; **not yet scientifically tested.**
- **Original tiny v26 proxy:** scientifically NO-GO on its specific benchmark.
- **PCMA memory/probe ports:** negative/no-scale on their respective toy/port tests.
- **This integration branch:** correctness proof-of-work only (pending exact-head CI), not scientific data.
- **No claimed SI breakthrough, paid compute, reserved scientific seeds, merges or other-project writes.**
