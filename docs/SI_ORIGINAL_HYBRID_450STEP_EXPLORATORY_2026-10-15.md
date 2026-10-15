# SI Original Hybrid — 450-step exploratory continuation (2026-10-15)

This note supplements `docs/SI_ORIGINAL_HYBRID_CPU_RESULTS_2026-10-15.md` on the isolated SI-original branch; it does not supersede the fixed 80-step outcome.

**Reason:** both models lost to a no-training persistence predictor during the frozen 80-step pilot. To determine whether undertraining explains some of that failure, models were trained for 450 steps on the *same engineering seeds*, same batch 16, same world streams, same LR/optimizer, and same held-out suites. This exploratory follow-up was decided *after* the 80-step result, so it is **not a preregistered scientific replication**, and those seeds cannot serve as independent confirmation.

Mean over 93011, 93012, 93013. Next-observation MSE (lower is better):

| World | Original hybrid, 64,614 params | Transformer, 64,637 params | Zero-training persistence |
|---|---:|---:|---:|
| Stable | 0.0116787 | **0.0114597** | 0.0114866 |
| Surprise switch | 0.0123066 | **0.0120598** | 0.0121260 |
| Nonlinear OOD | 0.0117517 | 0.0115051 | **0.0114468** |

Per-seed stable MSE: hybrid [0.0112823, 0.0119152, 0.0118385]; Transformer [0.0110964, 0.0116777, 0.0116051]. Runs performed locally, not on GitHub Actions or Modal.

**Interpretation:** training longer gets both models near the persistence control but does not generate an architecture win. Original hybrid has slightly worse mean MSE on all three evaluation categories; it also took more CPU wall time in this reference implementation (sparse dispatch and Python-loop recurrence are not optimized). No claim about hardware-optimal FLOPs or GPU throughput. The benchmark remains largely an easy near-identity extrapolation problem.

**Decision:** no-go for paid training on this task. Next: precommit an action-dominant, hidden-dynamics and intervention-sensitive CPU test on fresh seeds; keep Transformer, persistence, causal recurrent and action-only controls; then evaluate mechanism ablations and efficient adaptive compute. No budget or scientific job authorization was given.
