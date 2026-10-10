# AU recurrence-path attribution: post-AT stage gate

**Status: exploratory CPU diagnosis completed. Freeze all AT/AU evidence. No demonstrated RLT advantage over Transformer and no GPU authorization.**

Ground truth: `experiments/rlt/cpu/results/rlt-recurrence-path-attribution-20261009-au.json` and https://github.com/vinceackermann2-sys/tam-research/actions/runs/37899278588.

The single fresh-seed AU run trained static residual gated-scan RLT and coupled adaptive residual gated-scan RLT (each 29,504 parameters) for 20 seconds on 8-token immediate last-write, then 90 seconds on 64-token last-write at gaps {1,4,16,40,63}. All five held-out bins used 128 examples of each binary answer. No GPU was used; seeds were unique to this CPU attempt. The model and protocol are **not** a scientific replication.

## Held-out accuracy averaged across five equally sized distance bins

| Trained architecture | Unmodified full model | Encoder-only hidden initialization | Recurrent-only hidden initialization | Static encoder + recurrent gain |
|---|---:|---:|---:|---:|
| Static residual | 100.00% | **99.61%** | 99.22% | 100.00% (identical to full) |
| Coupled adaptive | 100.00% | **80.63%** | **97.66%** | 96.41% |

Detailed adaptive encoder-only accuracy was 98.83%, 100%, **52.73%**, **51.95%**, 99.61% at gaps 1, 4, 16, 40, 63 respectively. The recurrent-only adaptive path obtained 95.70%, 96.09%, 96.48%, 100%, 100%. These represent **paired inference interventions of trained networks**, not retrained architectures or equal-active-parameter comparisons. A drop from a surgical ablation can be caused by distribution shift. The strong residual encoder-only score shows that this toy task does not *require* the learned recurrence pathway for that checkpoint, even though the recurrence-only path is also strong.

Both models reached 100% across all full-model long-distance held-out bins in AU. In the previous separate CPU run AT, with a different seed, the coupled adaptive model collapsed to exactly 50% across all gaps while static residual and Transformer each reached 100%. The contrast **suggests initialization/data/optimization sensitivity**; two distinct one-seed runs cannot establish frequency, cause, or confidence bounds.

## Gate diagnostics measured after AU training

- Static residual: retention gate mean **0.88069**; near-one fraction **6.98%**; recurrent RMS **0.83859**, encoder RMS **0.87231**.
- Adaptive: retention gate mean **0.82306**; near-one fraction **9.50%**; output-gain mean **1.10850**, gain std **0.62540**, gain below 0.1 fraction **2.34%**, above 1.9 fraction **3.90%**.
- These descriptive statistics do **not** identify what caused AT collapse. A successful AU adaptive model already has some saturating gains, so saturation alone cannot explain the earlier failure.

## Next evidence gate: CPU-only seed sensitivity

Preregister a small fresh-seed, no-GPU sensitivity panel for the **same full 20s + 90s curriculum**, comparing static residual versus coupled adaptive under equal CPU time, exactly 29,504 parameters, balanced independent holdouts and frozen hyperparameters. Report full held-out accuracy by gap and both classes, NLL, number of steps/tokens per candidate, and gate-activation diagnostics. Include original AT/AU outcomes as historical observations only; do not cherry-pick best seed. Stop if independent conditions or reproducibility fail. Any outcome is a tiny synthetic-task result, **not evidence that RLT is superior in language modeling**.

No further 15M language sweeps, scientific replication, 250M/5B scaling, or breakthrough announcement is supported by AK–AU. Claims/results remain immutable; never rerun claimed GPU attempts, and protect seeds 58232/58233. All writes restricted to `exp/rlt-colab`; do not modify `main`.
