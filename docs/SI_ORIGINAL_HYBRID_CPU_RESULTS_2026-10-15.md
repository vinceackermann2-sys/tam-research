# SI Original Hybrid v0 — fixed 3-seed CPU pilot results (2026-10-15)

**Status: FAIL INFORMATIVENESS GATE. No architecture breakthrough. No GPU spent.**

Predeclared, freeze-before-run protocol: docs/SI_ORIGINAL_HYBRID_PROTOCOL_2026-10-15.md
Frozen implementation/harness head before results: 8822fa6be5aefa98360c3846f91b17e3cf59d376
Local CPU: PyTorch 2.10.0+cpu, two threads, steps 80, batch 16, seq length 16, paired seeds 93011/93012/93013.
Models: original SI hybrid 64,614 parameters; dense Transformer control 64,637 (difference 0.036%). Same deterministic sample stream and optimizer. Counts include inactive adapter parameters; this pilot does **not** establish equal-active-FLOP training.

## Held-out next-state mean squared error (lower is better)

| Mode | SI hybrid mean | Transformer mean | Persistence mean |
|---|---:|---:|---:|
| Stable | 0.024595 | 0.024448 | **0.011487** |
| Surprise switch | 0.025007 | 0.024819 | **0.012126** |
| Nonlinear dynamics | 0.025087 | 0.024925 | **0.011447** |

Each mean is an average over 3 fixed paired seeds, with 3 batches × 32 new worlds per seed and mode.
Model seed-wise stable: hybrid [0.028733, 0.026359, 0.018691], Transformer [0.023112, 0.028555, 0.021678].
Switch: hybrid [0.028179, 0.027007, 0.019835], Transformer [0.022683, 0.029633, 0.022141].
Nonlinear: hybrid [0.028570, 0.027353, 0.019337], Transformer [0.023171, 0.029016, 0.022588].

## Verdict and scientific limitations

**Benchmark invalid for positive architecture attribution**: just guessing the next observation equals the current observation beats both trained models by >2× in MSE. The generator uses a slowly changing state, so persistence is overwhelmingly favored; 80 steps may also undertrain both architectures. We do not reinterpret an architecture's relative performance as progress when a no-learning predictor wins.

The hybrid does not win the 3-seed average, and model training CPU wall-seconds are not hardware-optimized, budget-matched FLOP evidence. This pilot does *not* disprove the original concept, because the test does not exercise useful memory, nontrivial planning, continual learning or actual multimodality. It establishes code/gradient/causality feasibility and reveals an uninformative proxy task.

The scored pilot seed/data combinations 93011/93012/93013 must not be reused or replayed as scientific attempts. Before further tests, develop a task with a nontrivial no-learning persistence baseline and with actual system identification, partial observability, longer-time dependencies, and information-rich action interventions, using **new** predeclared development/scientific seeds. Compare strong classical controls and fair Transformer/recurrent models. Keep old negative results public.

The current code implements an incomplete original architecture: memory storage is a separate interface (not connected in forward); modality tags are not trained cross-modal encoders; fast weights are not trained for continual learning; optional refinement has no learned stopping policy; one-step planner is an interface with no validated reward. Do not present these as completed claims.
