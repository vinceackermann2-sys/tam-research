# SI Original v5 — two-action planning gate: completed engineering results
Date: 2026-10-16. Branch `experiment/si-original-two-step-planning-v5-20261016` stacked on original SI fast-memory v4 PR #1476, planning PR #1475 and original architecture PR #1459. **No new SI breakthrough, no paid GPU/Modal run and no merge to main.**

## Execution provenance and guardrails

- Predeclared protocol `docs/SI_ORIGINAL_TWO_STEP_V5_PROTOCOL_2026-10-16.md` committed **before** confirmation runs.
- Benchmark script `scripts/si_original_two_step_v5.py` GitHub blob **`412a2d772f4860727a4808a1ae0a1411f26f9df5`** equals the executed local script's Git blob exactly. Local SHA256 `f7da050d4864441f20e93a934dbd55c08abf7982daa9e6204b23f6e9abeca17b`. Three integrity tests passed locally (correct two-action simulator, hidden-state/target independence, oracle zero regret).
- Fresh engineering seeds **98121, 98122, 98123**: **all completed once** with stderr empty; immutable individual output records written locally. 300 AdamW steps per model, batch16, seq20, identical training worlds across paired models; CPU PyTorch 2.10, 2 threads. Three training families direct/delayed/switch; nonlinear entirely held out. Each eval family has **2×32 new worlds/seed**. Actual goal-conditioned two-step planning over 16 sequences; report actual simulator regret and accuracy rather than prediction-only metrics.
- Base model code in local execution was the earlier **functionally matched 133-line partial source copy**, not a byte-verified complete 298-line GitHub `si_original_hybrid_v0.py`. The new benchmark and fast-memory scripts are exact GitHub copies, but **the neural base code has not been independently source-attested**. The repo script imports the canonical complete base model. This prohibits strong scientific claims until a fully source-verified independent replication is completed.
- Model sizes **65,527** original hybrid+913-param fast memory and **65,550** Transformer+same fast memory; nominal parameter parity only, not active-FLOP parity. Two-step planning imagines next state via model prediction; it never accesses true next state during action choice. Strong controls: all-history and last-6 ridge (fit only completed observed transitions), random 16-way action sequence; oracle evaluator alone sees hidden matrices.
- All reported values are mean across 3 engineering seeds, so 192 evaluated worlds per family. This is small and result differences can be noisy. **No confidence-interval-based superiority claim** is supported.

## Primary: held-out nonlinear planning

| Seed | Hybrid top-pair accuracy | Transformer top-pair accuracy | Hybrid true regret | Transformer true regret | Ridge-all accuracy | Ridge-all regret |
|---:|---:|---:|---:|---:|---:|---:|
| 98121 | 0.500000 | 0.296875 | 0.067838 | 0.100920 | 0.578125 | 0.040871 |
| 98122 | **0.187500** | 0.250000 | **0.318146** | 0.149681 | 0.546875 | 0.015695 |
| 98123 | 0.500000 | 0.390625 | 0.090055 | 0.092421 | 0.750000 | 0.025629 |
| **Mean** | **0.395833** | 0.312500 | **0.158679** | 0.114340 | **0.625000** | **0.027399** |

Additional nonlinear baselines: ridge-recent accuracy **0.338542**, regret **0.268296**; uniform random-pair accuracy **0.052083**, regret **1.761801**. Hybrid nonlinear two-step query prediction MSE **0.108869**, Transformer **0.098082**. Hybrid has higher optimal-pair exact match but **38.8% higher mean regret** due to costly errors; failed predeclared quality gates, including strong classical controls.

## Full two-step action planning panel

| World family | Hybrid accuracy | Transformer accuracy | Ridge all | Ridge last-six | Random |
|---|---:|---:|---:|---:|---:|
| Direct | **0.380208** | 0.296875 | **0.703125** | 0.286458 | 0.046875 |
| Delayed | **0.260417** | 0.255208 | **0.546875** | 0.296875 | 0.041667 |
| Switched | **0.281250** | 0.255208 | 0.192708 | **0.322917** | 0.036458 |
| Unseen nonlinear | **0.395833** | 0.312500 | **0.625000** | 0.338542 | 0.052083 |

| Family | Hybrid true regret | Transformer true regret | Ridge-all regret | Ridge-last6 regret |
|---|---:|---:|---:|---:|
| Direct | **0.114230** | 0.119512 | **0.015401** | 0.243269 |
| Delayed | **0.233770** | 0.187105 | **0.037368** | 0.261868 |
| Switch | **0.255839** | 0.227970 | 0.401999 | **0.205735** |
| Unseen nonlinear | **0.158679** | 0.114340 | **0.027399** | 0.268296 |

Per-model mean reference CPU train time excluding evaluation: original hybrid+memory **7.3643 s**, Transformer+memory **5.1321 s**, ratio **1.435**. Mean two-step evaluation ~0.857 s vs ~0.837 s per candidate over four families (fast memory inference, model and ridge estimates). Not a GPU/hardware optimized FLOP comparison. The hybrid also has higher two-step query MSE in direct (0.098879 vs 0.089544), delayed (0.157577 vs 0.143464), and nonlinear (0.108869 vs 0.098082), lower in switch (0.196061 vs 0.197967).

## GO/NO-GO decision

**FAIL predeclared hybrid-versus-Transformer-and-ridge gate**. Hybrid's held-out nonlinear average regret is higher, not >=10% lower; classical ridge-all strongly dominates both exact-action accuracy and regret. Neither model demonstrates competitive multi-step world inference, and the original hybrid does not have a reliable compute-adjusted advantage.

**Do not scale unchanged original SI hybrid+memory or spend reported ~$25 Modal credits.** Do not interpret a pair-selection accuracy win as superiority when realized regret, MSE and classical controls oppose it. The v4 one-step fast memory is still useful on these narrow simulations; v5 reveals compounding errors in unrolled imagined states and an unresolved stability problem.

Meaningful next CPU-only scientific gate would be an **independently source-attested** implementation of a stronger contextual representation or online state-inference mechanism and **closed-loop multi-step planning** with partial observations/delays, comparing an equally equipped Transformer, non-neural online system identification, and strong explicit-memory baselines under identical training/inference compute. Freeze seeds and endpoints, preserve other project branches. No automated tasks, no paid compute, no breakthrough.
