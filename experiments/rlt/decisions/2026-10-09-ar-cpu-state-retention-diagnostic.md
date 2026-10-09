# AR CPU last-write state retention: diagnostic interpretation

**Decision: NON-DIAGNOSTIC for memory advantage; all three models remained near chance. No model-win, no breakthrough, no GPU permission.**

Run: https://github.com/vinceackermann2-sys/tam-research/actions/runs/37895906237
Frozen result: `experiments/rlt/cpu/results/rlt-lastwrite-distance-probe-20261009-ar.json`
Frozen source: `experiments/rlt/cpu_lastwrite_probe_ar.py`
CPU-only 12 passing input/oracle/equal-parameter/gradient tests. 2 warnings, no test failure. All models exactly 29,504 parameters, identical 1,000 optimizer steps and 512,000 input tokens, one model seed and common data/eval seeds.

| Model | Actual CPU train time | Tokens/sec | Accuracy across 5 held-out gap bins (128 examples each) |
|---|---:|---:|---:|
| Static residual scan | 16.3774s | 31,263 | 52.344% |
| Adaptive scan | 16.3726s | 31,272 | 52.344% |
| Transformer | 6.5205s | 78,522 | 52.344% |

For every model, observed binary NLL was close to the 50/50 baseline (~0.693) across gaps 1, 4, 16, 40, 63. The held-out target frequencies range from 62/128 to 74/128, and predicting the majority label in each bin can account for a 52.344% aggregated score. Therefore the matching accuracies may reflect collapse to a majority class rather than state retention.

**Protocol shortfall:** AR set a nominal 35s budget, but capped training at 1,000 steps. All candidates reached 1,000 steps first. Consequently, AR was effectively an **equal-optimizer-step CPU probe**, not a matched 35s wall-time experiment. Report observed CPU times and throughput, never claim matched wall-clock execution. The Transformer processed the same 1,000 steps much faster. No success/advantage conclusion is warranted.

**Required next check (CPU only):** a positive-control experiment with 8 input tokens, the most recent SET immediately before the QUERY, independent held-out balanced labels, and the same readout. Train each exactly parameter-matched model for a fixed full wall-time without hitting a max-step cap; confirm the classification head/data pipeline can learn above chance (e.g. >=85% accuracy on independently generated near-write queries). If the easy task fails, halt synthetic distance-screening and debug the task/setup. If it succeeds, separately design a frozen distance-generalization challenge and compare with a common metric/budget.

AR data/test findings are immutable. No previously claimed GPU job or seed may be repeated. All new experiments, if any, use separate unique IDs; new CPU work grants no authorization for scientific replication, 250M/5B scaling, or breakthrough claims. All writes remain on `exp/rlt-colab`, not `main`.
