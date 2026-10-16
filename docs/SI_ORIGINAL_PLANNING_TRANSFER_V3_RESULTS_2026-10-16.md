# SI Original — Hidden-dynamics planning and transfer v3 CPU results (2026-10-16)

**Verdict: FAIL — original hybrid does not demonstrate useful one-step goal planning at this budget.** This is an early-stage, synthetic, CPU-only test, NOT a finding that all SI research is futile. No Modal/Colab/GPU charges; no other research branches or main modified.

## Provenance and limitations

- Original SI hybrid branch (PR #1459), stacked follow-up `experiment/si-original-planning-transfer-v3-20261016`. This benchmark uses `OriginalSIHybrid` and `TransformerControl` (64,614 and 64,637 instantiated params), NOT the simplified surprise router or EHW.
- Harness file `scripts/si_original_planning_transfer_v3.py` committed at `1fa6f0ce342fed09e610e069e0d8dd4a48db9ecb`. **Exact GitHub blob `a266f7e86c4b7224aa93c20de6089c4543be26db` equals the local execution script's git blob** (local SHA256 `51280eb2d268ad7152b87a253d30697384dd70a922a7585bbfffbee7f1ff9e40`). The local runner used a manually transcribed, functionally matched subset of the original GitHub model definitions for the two classes, including identical 64,614/64,637 parameter counts. **The local model module bytes were not independently verified byte-for-byte against GitHub's full module.** The actual repository script imports the original full GitHub module. This is engineering evidence; run full-source integrity checks on GitHub before treating any result as authoritative scientific comparison.
- Protocol `docs/SI_ORIGINAL_PLANNING_TRANSFER_V3_PROTOCOL_2026-10-16.md` was committed **before** new seeds. Smoke seed 96800, 3 steps/batch4; not confirmation. Fixed confirmation seeds **97111, 97112, 97113** each completed once successfully, stderr empty, no retries.
- CPU/PyTorch 2.10, 2 threads, 450 AdamW steps per architecture/seed, batch16/seq20, LR=.001, WD=.01, clip1. Identical data stream for matched models; 3 x 64 independent held-out worlds per seed and family. Four alternative actions per world, goals sampled from actual simulator outcomes with .08 noise; action regret scored against true oracle. Strong *no-training* ridge estimates conditioned on all completed prior transitions or the last six; uniformly random actions scored on identical goals.
- **Training families:** direct action effects, delayed prior-action effects, surprise regime switches. **Held-out family:** nonlinear state interactions. No private transition matrices or next query outcome enter neural model inputs or classical fitted context. Memory interface in full hybrid remains disconnected; no multimodal generality, true continual learning, or useful multi-step planning has been demonstrated.

## Paired result: held-out one-step planning accuracy (%) and regret (lower better)

Mean across the three completed engineering seeds, 576 action choices per environment. Random accuracy fluctuates around 25% because finite samples.

| World family | Original hybrid accuracy | Transformer accuracy | Ridge all-history | Ridge last-6 | Random |
|---|---:|---:|---:|---:|---:|
| Direct (training family, unseen worlds) | 22.40% | 25.00% | **96.70%** | 80.03% | 27.78% |
| Delayed (training family) | 25.17% | 25.35% | **67.01%** | 37.33% | 28.12% |
| Switch (training family) | 26.91% | 24.31% | 59.55% | **77.95%** | 27.95% |
| **Nonlinear (unseen family)** | 23.09% | 25.87% | **96.70%** | 76.74% | 27.43% |

| World family | Hybrid regret | Transformer regret | Hybrid query prediction MSE | Transformer query prediction MSE |
|---|---:|---:|---:|---:|
| Direct | 1.5793 | **1.5082** | 0.2843 | **0.2805** |
| Delayed | 0.2032 | **0.1841** | 0.2007 | **0.1903** |
| Switch | **1.4777** | 1.4779 | **0.2864** | 0.2996 |
| **Nonlinear** | 1.4567 | **1.4325** | 0.2844 | **0.2808** |

**Per seed held-out nonlinear action accuracy**:
- 97111: hybrid 0.2500; Transformer 0.2344
- 97112: hybrid 0.2083; Transformer 0.2604
- 97113: hybrid 0.2344; Transformer 0.2812

Hybrid wins **1/3** paired nonlinear seeds and loses the mean, regret and MSE. **All preset continuation criteria fail**, especially comparison to strong ridge baselines. On the synthetic direct and nonlinear worlds, online ridge nearly solves the small hidden-rule inference problem without pretrained weights.

Mean CPU training wall-seconds excluding evaluation: hybrid **11.456 s** vs Transformer **10.646 s**, ratio **1.076**; no GPU throughput or FLOP parity claim. Parameter parity does not imply equivalent active parameters: original hybrid retains dormant fast-adapter parameters.

## Interpretation and next decision

The current *original* attention/recurrent/MoE/refinement stack behaves close to random on **realized goal-directed decisions**, despite being trained with action-conditioned next-state loss. The data are informative enough for an episodic learner (ridge controls 67–97% depending on family), so this is not merely an impossible task. Simply adding more attention/recurrent parameters is not justified.

**STOP the existing full-stack architecture as an unchanged GPU candidate.** The broader research question remains open. The principled CPU next step is to create an **integrated fast episodic world-memory module**, with query-conditioned writable per-episode experience, and compare (i) original+memory, (ii) Transformer+same memory, (iii) strong classical ridge, (iv) original unchanged, and (v) simpler alternatives. Do not claim novelty when a fixed linear solver dominates; test nonlinear, delayed and transferred rules where explicit learned memory may help. Keep tight parameter/active FLOP and opportunity-cost accounting, causal query integrity, and actual action success.

Before any subsequent scientific run: verify exact GitHub model source in the execution environment, preregister new unconsumed seeds, and reproduce source/test checks with independent evaluator. **No paid authorization or breakthrough claim.**
