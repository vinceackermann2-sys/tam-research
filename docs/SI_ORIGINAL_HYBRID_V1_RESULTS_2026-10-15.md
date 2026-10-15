# Original SI hybrid v1 — frozen causal-rule pilot results (2026-10-15)

**Evidence status: improved benchmark informativeness, no demonstrated causal inference advantage or SI breakthrough.**
Source and test harness frozen before scoring at c725b3cfa19921019b12baa5f4c80e05f1951f9c, protocol at a83168a6414be6f3d8a796427b95a4015be5503b.
Predeclared 3 paired CPU training seeds 94011/94012/94013, 180 AdamW updates/model, batch 16, seq 16, identical data and optimized objective, 64,614 original SI hybrid vs 64,637 Transformer params.

## Results — last 4 next-step predictions, 3 fresh seeds

| Held-out task | SI hybrid MSE | Transformer MSE | Persistence MSE |
|---|---:|---:|---:|
| Stable hidden causal sign | 0.127888 | **0.127240** | 0.197005 |
| Hidden rule switches halfway | **0.128612** | 0.132283 | 0.196992 |
| Nonlinear interactions | 0.128212 | **0.127814** | 0.197169 |

Both learned models beat persistence, so this particular gate is **more informative than v0**. Hybrid is around 2.78% lower MSE on switch tasks than Transformer, but loses modestly on stable and nonlinear tasks. These gaps are too small to establish a reliable architecture benefit without stronger controls.

Per-seed switch MSE:
- Hybrid: 94011=0.129855, 94012=0.130096, 94013=0.125887.
- Transformer: 94011=0.133480, 94012=0.133285, 94013=0.130083.

Mean local CPU training times: hybrid 2.373 sec, Transformer 1.403 sec. This is a prototype Python recurrent loop and per-token expert dispatch, **not an equal-compute/GPU throughput result**. No Modal or Colab work performed.

## Post-hoc diagnostic (not a predeclared comparator)

After examining results, evaluated a simple **sign-blind** analytic predictor that uses only current observation: next_state ≈ 0.24 * current_state. On the same held-out batches, its MSE ≈ 0.130044 stable, 0.130045 switch, 0.130609 nonlinear.

This almost matches both trained models, despite **knowing nothing about the hidden causal sign**. It suggests neither model has proven actual causal rule inference from sequence history; observed switch advantage could result from trivial regularities. Because this control was introduced after seeing pilot data, treat it as exploratory diagnosis and **test it with future fresh locked seeds**, not as a predeclared p-value or discovery.

## Verdict and next decisions

- **Engineering gate:** model is trainable, causally masked, reproducible, and can outperform persistence in a new task.
- **Architecture gate:** inconclusive; no robust evidence that recurrence/MoE/refinement/memory provides meaningful improvement over Transformer or no-history baseline.
- **Breakthrough gate:** emphatically NOT passed.
- **Immediate next experiment:** controlled *history ablation*: train/evaluate paired models with context destroyed except present state/action; require statistically meaningful history-dependent improvement on unseen world-sign rules. Include sign-blind analytically and context-only recurrent/Transformer baselines in **new** seeds. Add changed sign, unseen action mappings and compute measurements. Prevent sign leakage by keeping hidden sign inaccessible.
- **Subsequent gates:** actual long-term retrieved memory, continual adapter learning with replay and old-task retention, measured dynamic early exit, longer-horizon counterfactual action planning, true multimodal encoders and representative language data.

The previous seeds are consumed for pilot scoring and must not be replayed as future scientific replication; reserve fresh runs separately. The user's ~$25 Modal credit estimate is unverified; $0 spent here. No GPU execution authority granted.
