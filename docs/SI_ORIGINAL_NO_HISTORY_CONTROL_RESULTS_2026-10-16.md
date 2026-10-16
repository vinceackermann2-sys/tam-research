# SI Original — No-history Control, CPU Results (2026-10-16)

**Outcome: history carries useful signal, but the trained adaptive gate still failed.** Code frozen before attempts: `scripts/si_original_no_history_control.py` at `e64aa7a7ae973d29188cc97f0e648d5cfb9878b4`; protocol `docs/SI_ORIGINAL_NO_HISTORY_CONTROL_PROTOCOL_2026-10-16.md`. Exact 300 optimizer updates, batch 16, 24 timesteps, paired worlds/actions/eval; no paid GPU/Modal. Original seed 95213 interrupted during local command (0-byte JSON). Addendum `docs/SI_ORIGINAL_NO_HISTORY_CONTROL_TIMEOUT_ADDENDUM_2026-10-16.md` records its consumption and selection of fresh replacement **95214 before running it**. These are engineering results with replacement, not strict scientific replication.

All candidates instantiate identical 20,265 parameters, with same weights at initialization, but `no_history` masks all earlier timesteps and leaves some projections functionally unnecessary, so effective active capacity is not exactly matched.

## Results

| Seed | Candidate | Stable full MSE | Random-switch late MSE | CPU train seconds |
|---|---|---:|---:|---:|
| 95211 | No history (current state/action only) | 0.332039 | 0.337626 | 2.55 |
| 95211 | Local window 6 | 0.308731 | **0.327205** | 2.81 |
| 95211 | Full-prefix | **0.305444** | 0.335964 | 2.74 |
| 95212 | No history | 0.317683 | 0.333670 | 3.19 |
| 95212 | Local window 6 | 0.291874 | **0.309881** | 3.20 |
| 95212 | Full-prefix | **0.289557** | 0.322176 | 2.53 |
| 95214 | No history | 0.310785 | 0.306027 | 3.04 |
| 95214 | Local window 6 | 0.284871 | **0.297306** | 2.79 |
| 95214 | Full-prefix | **0.281269** | 0.304308 | 3.21 |
| **Mean** | No history | 0.320169 | 0.325775 | 2.928 |
| **Mean** | Window 6 | 0.295159 | **0.311464** | 2.935 |
| **Mean** | Full prefix | **0.292090** | 0.320816 | 2.825 |

Mean nonlinear full MSE: no-history 0.321451; local 0.294982; full prefix **0.292251**.

**Interpretation:** the task isn't completely solvable by an action/current-state-only predictor. Full historical attention has ~8.77% lower stable MSE than no-history, local has ~7.81% lower; late after random changes, local has ~4.39% lower MSE than no-history, vs full-prefix ~1.52% lower. Direction is consistent across all three seeds (history helps, long prefix best stable, short prefix best after a switch). No statistical uncertainty estimate with n=3 establishes generalization beyond this synthetic family; the no-history model still has same nominal parameters but different active dynamics.

However the *learned adaptive soft mixture* did not produce change-aware behavior in the prior frozen test, so it remains a no-go for GPU promotion. New CPU research could assess a strictly causal **surprise detector with memory/rule-discovery tasks** and transfer to another world family, before returning to real active compute savings and the original full SI architecture. No claim of AGI/SI, nor of proven architecture superiority at equal active FLOPs. No credits spent.
