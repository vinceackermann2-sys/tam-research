# SI Original Hybrid v1 — preregistered causal-transition identification pilot

Date: 2026-10-15. Research track: original hybrid architecture, NOT EHW.
Predecessor draft PR #1459, frozen source e1bca79558d1404dd89ae2d77ce648f50244f404.
Separate stacked branch: experiment/si-original-causal-transition-v1-20261015
Source code/tests frozen at c725b3cfa19921019b12baa5f4c80e05f1951f9c before execution.
No paid/GPU/Modal/Colab job authorized.

## Why this is a new trial

v0 slow-dynamics pilot failed its informativeness gate because persistence MSE was >2x lower than either trained model. Preserve negative v0 results. In v1, each world has a *hidden random causal sign* that flips the effect of actions on future world state; it can be inferred only from earlier action/outcome correlations, making one-step prediction more information-sensitive. The sign is **never input to models**; future observations aren't available in past causal attention or recurrence.

## Fixed protocol

- Architectures: v0 OriginalSIHybrid vs v0 matched-parameter TransformerControl, no weight/code changes. Both get 64,614/64,637 trainable params, same input/action/target, same loss/optimizer.
- New seeds: **94011, 94012, 94013**, exactly 180 steps, batch=16 episodes, sequence=16, AdamW lr=.001, weight decay=.01, grad clipping=1, two CPU threads.
- Task: per-episode random sign of shared action→next-state mapping, deterministic sampled trajectories. Training only stable-rule mode. Evaluation always from **unseen** worlds and frozen seeds (seed×10000+85000, +0/+1/+2 batches). No validation tuning.
- Primary outcome: held-out MSE on final four transitions in stable unseen worlds, reporting both models and persistence. Secondary outcomes: hidden sign flips after half the sequence and nonlinear state perturbation, plus CPU training times, params.
- Interpretability gate: MSE should beat persistence meaningfully before making any relative architectural observation. If it does not, mark benchmark failed as in v0 rather than claiming an architecture win.
- A win on this synthetic model would remain insufficient for a scientific breakthrough: shared linearizable action effects, no image/audio/text learning, no multi-step closed-loop planner reward or continual learning; no equal measured training FLOP/H100-dollar comparison; no independent scientific replications.
- Do not rerun the three seeds after scoring, preserve negative results, CI original attempt only. No paid compute without separate frozen source/tree/harness/seed/cap and explicit authorization.

Reproduce:
~~~sh
python -m pytest -q tests/test_si_original_causal_transition_v1.py
for seed in 94011 94012 94013; do
  python -m scripts.si_original_causal_transition_v1 --seed "$seed" --steps 180 --batch 16
done
~~~

Next gates (only if useful): ablate recurrence/MoE/refinement and add no-history, classical system-ID and recurrent/hybrid controls; introduce partial observability, unseen causal mappings, multi-step intervention planning with real rewards; actual versioned memory/replay training, modality learning and calibrated active compute. Explicit preregistered stop at practical compute/value limit, not from a toy result.
