# SI Original Hybrid Architecture — Phase 0 CPU Pilot (2026-10-15)

Status: **preregistered CPU-only engineering/learning probe**. No breakthrough, no paid compute, no claim of SI.
Dedicated branch: experiment/si-original-hybrid-workspace-v0-20261015
Base main: abe7f60490ed11dc7fa115aeeff39da50cefea8e
Prototype and benchmark frozen at head 8822fa6be5aefa98360c3846f91b17e3cf59d376 before three-seed testing.
This is **not** EHW: EHW is a separate causal-hypothesis research track (#1441). This branch directly tests the user's original hybrid SI concept as outlined in the screenshots.

## 1. Original architecture interpreted precisely

1. Hybrid sparse backbone: local causal attention layers alternating with gated recurrent state-space layers; one-expert-per-token MoE feed-forwards. Model can retain long-term state across a sequence in recurrence while attention handles local exact retrieval.
2. Adaptive compute: confidence/halting gate and optional refinement network, with an explicit inference skip. Current training evaluates the refinement for all tokens, so *no proven adaptive compute savings yet*.
3. World model: action-conditioned future-observation prediction; one-step planner queries alternative proposed actions against a goal. The planner is a smoke-tested interface, **not trained or benchmarked for useful agency**.
4. Memory hierarchy: positional causal working state + gated recurrent state; versioned external episodic store and an adapter intended for selective fast updates. The external store is **not yet fused into the network**, and the adapter has **not been trained for continual learning**. Replay/consolidation not implemented.
5. Self-evaluation: calibrated predictive variance and verified-error head, supervised using observed residuals with detached targets.
6. Unified multimodality: modality IDs and a shared internal tensor space are present; **image/audio/text encoders and actual cross-modal learning do not yet exist**. Current numerical state/action is a single modality.
7. Learning: gradient-based training over independent unseen dynamical worlds; no autonomous self-training loop, no self-improvement/ASI claim.

## 2. Code

- tam_research/si_original_hybrid_v0.py — model, dense Transformer control, synthetic environment, episodic store, objective, planning interface
- scripts/si_original_hybrid_benchmark.py — bounded CPU training + evaluation
- tests/test_si_original_hybrid.py — 5 correctness and safety-of-experiment checks

The Transformer comparison uses full causal attention, dense FFN, the *same observation/action stream*, shared targets and prediction/variance/verifier objectives. Its residual gain and slightly wider FFNs bring model sizes to 64,637 Transformer vs 64,614 hybrid (difference ~0.036%); parameters include optional adapter/refiner in hybrid, even though it is not all active in every pass.

## 3. Predeclared 80-step pilot

Training seeds: **93011 / 93012 / 93013**, fixed as pilot seeds (engineering only; do not reuse for future scientific replication).
For each seed:
- 80 AdamW steps, batch 16, sequence length 16, same deterministic batch seed stream for both models; LR=0.001, weight decay 0.01, gradient clip 1.0.
- CPU 2 PyTorch threads. Model initialization reproducible per seed; training data independent per episode.
- Primary held-out measure: stable-world next-step observation MSE against matched Transformer **and zero-training persistence baseline**. No tuning against held-out worlds.
- Secondary: switch and nonlinear out-of-distribution dynamics, CPU train wall-seconds, parameter count.
- Evaluation seeds: training seed*100000 + 85000 plus batch indices; 3 × 32 independent episodes per evaluation mode.
- If the persistence predictor wins substantially, the probe is non-informative for an architectural advantage; reject the benchmark as too easy / poorly calibrated instead of rationalizing model results.
- If a pilot numerically fails, save the failure and repair only on a *fresh* engineering attempt; do not rerun the same scientific job/seed.
- Do **not** claim a breakthrough if hybrid wins: this toy setting alone is inadequate and CPU training wall-clock is not GPU FLOP parity.

Run from repo root after installing requirements:

~~~sh
python -m pytest -q tests/test_si_original_hybrid.py
python -m scripts.si_original_hybrid_benchmark --steps 80 --batch 16 --seed 93011
python -m scripts.si_original_hybrid_benchmark --steps 80 --batch 16 --seed 93012
python -m scripts.si_original_hybrid_benchmark --steps 80 --batch 16 --seed 93013
~~~

No GitHub Actions GPU trigger or Modal runner in this branch.

## 4. Scientific progression and explicit stop conditions

**Gate 0 (current):** correctness, causality, matched parameter accounting, deterministic data, CPU feasibility, informative benchmark check against persistence. Outcome is PASS ENGINEERING or REDESIGN TASK, not SI/ASI.

**Gate 1:** stronger causal/information benchmarks: hidden world state, unrevealed dynamics, nontrivial long memory, proper held-out compositional interventions, surprise change-points. Mechanism ablations recurrence / sparse experts / memory / refiner / verifier; compare to equally trained parameter/compute-matched Transformer *and a recurrent/hybrid baseline* and classical predictors. At least three fresh independent seeds, predetermined splits and no target leakage. Measure held-out rollout/action success as well as loss and latency.

**Gate 2:** demonstrate *real* inference-time compute adaptation with measurements, online skills with replay retaining old tasks, memory retrieval/no contamination, calibrated uncertainty and cost reduction. Require non-degenerate routing. Report latency including retrieval/expert dispatch.

**Gate 3:** actual learned text/image/audio/action representations, agents, long horizons, held-out transfer; reproducible quality gains at equal-data/compute vs current strong models. New frozen protocol and hardware/credit accounting before any GPU attempt.

**GO / breakthrough-candidate:** multiple independent preregistered task families + 3+ paired seeds, meaningful advantage at equal training/inference cost, successful ablations attributing gains, real held-out task success and replication. This is a *research-go* signal, not proof of superintelligence.

**STOP / not worth further investment:** controlled repeated failures of essential mechanisms, failure to outperform simpler comparators after fair tuning, persistent worse compute/quality frontier, or infeasible value-per-credit given an explicitly fixed budget. Do not call one toy negative a global impossibility.

**Current expense:** $0 Modal. User estimates ~$25 credits; balance and authorization not verified. Keep every cloud/GPU action disabled until a separate user-approved frozen source/tree/harness, immutable attempt/seed and maximum dollar/time cap; retries=0 and never reuse consumed attempts. Other research branches are off limits. Branch -> draft PR -> original CI -> review; never directly modify main.
