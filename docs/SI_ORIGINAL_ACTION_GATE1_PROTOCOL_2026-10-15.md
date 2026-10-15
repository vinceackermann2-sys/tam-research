# SI original hybrid: action-sensitive world Gate-1 CPU protocol (frozen 2026-10-15)

Research branch: `experiment/si-original-hybrid-workspace-v0-20261015`. Code: `scripts/si_original_action_probe.py` introduced at commit `b93a1ab94c8a6df4d88e9d24a0060180deac6b98`; base model is `tam_research/si_original_hybrid_v0.py`. Previous 80-step gate failed against trivial persistence; longer training remains exploratory.

**Why new task:** each episode generates unknown 3×3 state and 4×3 action-effect matrices; their action effects are strong and fixed within the episode. The model receives 24 sequential (observed-state, chosen-action) pairs, predicts each following observation, and cannot read the unseen transition matrices. OOD variants change all transition laws at t=12 or add quadratic nonlinear effects. Context and targets are correctly time ordered; no target is fed into a given time step's prediction. Before freeze a single **35-step development smoke** used seed 94000; it is not part of confirmation.

## Locked Gate-1 confirmation

- Models: `OriginalSIHybrid(SIConfig())` vs `TransformerControl(SIConfig())` with approximately 64.6k params; compare persistence and zero-output nontraining baselines. Unlike the original pilot this new task meaningfully changes states in response to actions. No EHW component is substituted.
- Three **fresh CPU engineering seeds**: **94111, 94112, 94113**, no previous runs with these seeds. Each receives **300 optimizer steps**, **batch 16**, sequence length **24**, AdamW LR 0.001, weight decay 0.01, gradient clipping 1.0 and *identical per-step generated worlds and action streams* for each model.
- Evaluation: 4×32 independent unseen worlds per mode/seed, evaluation RNG `seed*1000000 + 85000 + batch`.
- **Primary:** mean held-out MSE across all 24 steps on stable worlds, hybrid minus Transformer; report all three paired seeds and compute wall time.
- **Secondary:** late-period MSE (t=12..23) on stable, switch, nonlinear worlds, alongside late-period persistence and zero baselines; report all outcomes, including regressions.
- Engineering **signal worth continuing:** both learned models beat both no-training controls on stable worlds and original hybrid beats matched Transformer in >=2/3 seeds and mean stable primary, without numeric instability. This is *not a scientific breakthrough*, GPU authorization, or evidence of general intelligence.
- **No-go for advancing this design to paid compute:** both trained models lose to simple baselines OR hybrid loses to Transformer on mean stable primary; redesign/ablate rather than rationalize or spend.
- Seed/data streams are fixed before confirming runs. If infrastructure fails, log the failure; do not silently change parameters and reuse scientific seeds. No compute budget consumed outside local CPU.
- Proper follow-up before stronger claims: additive recurrent & classical baselines, independent data families, active-FLOP/latency match, sparse expert & recurrence ablation controls, working episodic memory replay and continual learning, learned adaptive compute under measured budgets, and multi-step action success.

Reproduce from repo root after package deps:
```sh
for seed in 94111 94112 94113; do
 python -m scripts.si_original_action_probe --seed "$seed" --steps 300 --batch 16
done
```

This is an engineering CPU gate only, not a preregistered large-model scientific claim. Modal credits remain unverified and unused.
