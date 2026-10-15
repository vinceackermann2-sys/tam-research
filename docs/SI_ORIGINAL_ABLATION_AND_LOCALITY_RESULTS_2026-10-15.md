# SI Original Hybrid: ablation and controlled locality evidence — 2026-10-15

All runs CPU-only on original isolated SI hybrid branch. No Modal balance consumed. Main and other architecture branches unmodified. Negative outcomes retained, no breakthrough claims.

## Post-Gate-1 exploratory mechanism screen

Code frozen `scripts/si_original_mechanism_screen.py` at `3c0ead6a2d36e5de92dd5f9ec4ea7394b988da49`; fresh seeds 94211, 94212, 94213; 300 steps, batch 16, 24 observations/action pairs. Except Transformer, all entries are hybrid class with named switches. These models do **not** match active parameter counts; mechanism *screen*, not causal proof.

| Candidate | Params | Stable whole MSE | Abrupt-switch late MSE | Mean CPU train seconds |
|---|---:|---:|---:|---:|
| Full original hybrid | 64,614 | 0.288024 | 0.325948 | 4.528 |
| No recurrence (all local attention) | 66,094 | 0.289036 | **0.324754** | 3.796 |
| No MoE (dense FFN) | 44,808 | 0.288091 | 0.325029 | 3.258 |
| No optional refinement | 64,614* | 0.288042 | 0.325558 | 4.185 |
| Full Transformer | 64,637 | **0.285876** | 0.358261 | **2.763** |

*The refiner parameters are instantiated but skipped, so nominal counts substantially overstate active parameters.*

**Conclusion:** no tested individual component explains the hybrid's switch advantage. Dropping a complex module can match or improve switch MSE, often at lower CPU cost. The full hybrid still loses stable performance and wall clock. A simpler relevant inductive bias may matter more than the added mechanisms. This is unfavorable evidence for retaining the full hybrid design as-is.

## Controlled attention-window experiment

Frozen script `scripts/si_original_locality_probe.py` at `3b2ddb84e50da905f7537135163c1052cb6ae601`, protocol `docs/SI_ORIGINAL_LOCALITY_PROTOCOL_2026-10-15.md`. Fresh confirmation seeds 94311, 94312, 94313. Three identical local/full-prefix dense-attention blocks, no recurrent/multi-expert/refine operations, *identical initialization, parameter count (46,288), data, optimizer and training steps* for the **local_dense** vs **global_dense** pair. Everything but the causal attention mask (window=6 vs unlimited) is held fixed.

| Candidate | Params | Stable whole MSE | Abrupt-switch late MSE | Mean CPU train seconds |
|---|---:|---:|---:|---:|
| **Local-attention dense** | 46,288 | 0.291263 | **0.330625** | 2.680 |
| Global/full-attention dense | 46,288 | **0.286477** | 0.366279 | 2.677 |
| Full original hybrid (context only) | 64,614 | 0.291868 | 0.333129 | 4.732 |
| Transformer (context only) | 64,637 | 0.287686 | 0.369692 | 3.132 |

Switch late per-seed local vs global:
- 94311: 0.323583 vs 0.362658
- 94312: 0.330432 vs 0.365497
- 94313: 0.337861 vs 0.370682

**Observed effect:** local attention obtains **9.73% lower** abrupt-switch late error than global attention at equal model parameters, and wins each seed. **Tradeoff:** local attention increases stable whole-sequence loss by **1.67%**, so this is not a universal improvement. Implementation uses an O(T²) attention operation with a mask and therefore **does not prove a compute or asymptotic memory advantage** at long context.

## Scientific interpretation and design decision

Our original screenshot-inspired idea of mixing attention and recurrence, sparse MoE, variable thinking, and hierarchical memory remains an unproven and currently incomplete architecture. On this task, a much simpler *fixed local attention bias* explains a large part of the secondary switch benefit without needing the proposed recurrent / MoE / refinement complexity. **Don't spend ~$25 Modal on the original full stack at this stage**.

Next falsifiable design: learn a **causal, compute-priced adaptive attention horizon** that chooses long context for stable worlds and recent context after change-points, and compare it to both fixed horizons under identical data, active parameter budgets and measured compute. Distinguish *training with a learned gate* from inference with oracle/hindsight labels. Include an abrupt-switch detector and an independent family of task generators to rule out reliance on one simulator.

Future requirements: real memory/replay/continual-learning adaptation, calibrated verification, verified multi-step action planning and true multimodality. No fair evidence yet that a full SI or novel large language model is superior to strong baselines.

Status: **secondary mechanism identified provisionally; full SI design does not pass quality/compute advancement gate. No SI breakthrough, no proven failure of the broader hypothesis, and no GPU authorization.**
