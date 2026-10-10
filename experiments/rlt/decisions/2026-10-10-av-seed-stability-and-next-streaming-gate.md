# RLT post-AV evidence gate: freeze adaptive seed panel; investigate bounded-context streaming

**Decision:** Both static residual gated-scan RLT and coupled adaptive residual gated-scan RLT can learn a short binary last-write task. The three new AV seeds all solved it, but this does **not** establish a language-model breakthrough, longer context superiority, or a requirement for recurrence. Stop repeated short-context toy quality sweeps; the next hypothesis must genuinely test recurrent memory across chunk boundaries under bounded attention.

## Frozen AV 3-case CPU evidence

Successful workflow: https://github.com/vinceackermann2-sys/tam-research/actions/runs/38064599757
Pre-registered job: `experiments/rlt/cpu/jobs/rlt-adaptive-seed-sensitivity-20261010-av.json`
Durable single-issue seed claim: `experiments/rlt/cpu/claims/rlt-adaptive-seed-sensitivity-20261010-av.json`
Frozen panel: `experiments/rlt/cpu/results/rlt-adaptive-seed-sensitivity-20261010-av-summary.json`
Full per-seed outcomes: `...-case0.json`, `...-case1.json`, `...-case2.json`.

Each case trained 29,504-parameter residual and adaptive RLT on identical synthetic data and evaluation seeds within the case; 20 CPU seconds on length 8, then 90 CPU seconds on length 64 at gaps 1,4,16,40,63. No max step cap; independent fresh case seeds; 128 class-zero and 128 class-one evaluation examples per distance.

| AV case | Residual: all 5 long bins | Adaptive: all 5 long bins | Residual: 8-token accuracy after long training | Adaptive: 8-token accuracy after long training |
|---|---:| ---: | ---: | ---: |
| 0 | 100% | 100% | 50.00% | 88.28% |
| 1 | 100% | 100% | 97.27% | 52.34% |
| 2 | 100% | 100% | 71.09% | 58.98% |

Every long-distance bin achieved both class accuracies 100%, not majority-class collapse. Report all seeds; do not cherry-pick case 1. CPU wall-time hardware throughput varied substantially among GitHub runners, but cases were trained under equal **per-model wall-time within a case**. Do not infer equal step counts or full cross-run hardware comparability.

Historical exploratory same-curriculum runs: AT adaptive collapsed (50%) while residual and Transformer reached 100%; AU adaptive and residual both reached 100%. Thus among AT+AU+three AV cases, adaptive succeeded in 4/5 runs and collapsed in 1/5; this **tiny selected set is not a reliable estimator of intrinsic failure frequency or statistical significance**. Residual succeeded in 5/5.

AV's 8-token post-curriculum accuracies show length-dependent forgetting/interference, not a stable streaming-memory mechanism.

## Why current state task cannot establish recurrence advantage

- The evaluated models' causal encoder and decoder cross-attention can access prefix memory throughout the 64-token input; the Transformer likewise has full causal self-attention over that input.
- AU paired inference ablation showed trained residual encoder-only accuracy **99.61%** even with recurrent output disabled. So the trained static residual does not require recurrence for this task at this scale/seed.
- The two-way adaptive gain/retention coupling in the successful AU model changed dependency on the state contribution, but no matched Transformer disadvantage emerged.
- AP and AQ still found the 15,129,344-parameter RLT variants **worse than matched Transformer** on pinned FineWeb-Edu held-out NLL with equal GPU training seconds.

## Next preregisterable architectural hypothesis (zero GPU first)

**Streaming-RLT state carry across bounded-attention chunks**: Given input chunks `C_0,...,C_n`, the recurrent scan should carry only a fixed-width state `s_end` from chunk `i` into chunk `i+1`; current chunk attention sees only a fixed window, not the unlimited historical prefix. Validate:

1. A pure differentiable affine-scan chunk-carry primitive must match a monolithic causal scan for forward values and gradients across variable chunk sizes and nonzero initial states.
2. Test causality, finite state size, no accidental detach or future leakage; avoid claiming memory-bounded *training* because full BPTT across chunks stores the computational graph.
3. Implement a full streaming **model** only after the primitive passes CPU tests and after specifying what encoder attention/cross-attention caches are dropped, what persistent state crosses chunks, and how a matched Transformer control is limited to the exact same input context and budget.
4. Follow with CPU-only last-write tests where the update is older than the accessible attention window. Include a strong bounded-window Transformer, a no-memory model, and a model with state reset at chunk boundaries. Equal information, active parameters, train/eval seeds and wall-time must be documented.
5. Future GPU or scientific work requires an independently approved and preregistered unique attempt. Do not launch fresh GPU studies, scientific replication, 250M/5B scale, or claim breakthrough from synthetic results.

Governance: preserve all AK–AV results and seed claims, protect 58232/58233, read live main and `exp/rlt-colab` before every mutation, write only to `exp/rlt-colab`.
