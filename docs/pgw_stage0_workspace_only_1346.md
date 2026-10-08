# PGW-only forensic continuation — October 8, 2026

This is an **evidence-preserving Stage-0 diagnostic**, not a new architecture or a paid training run.

- Repository: `vinceackermann2-sys/tam-research`
- Immutable source snapshot used to open this branch: `86ed6a48a84465eb5171ea8b042058bf200deef5`
- Branch: `research/pgw-stage0-workspace-only-20261008`
- Investigation: [#1346](https://github.com/vinceackermann2-sys/tam-research/issues/1346)
- Only added files: this note and `tests/test_pgw_workspace_only_1346.py`.
- The historical frozen experiments remain in their own `tam_research/pgw_v*` and `tam_research/pgw_core_mechanism` paths. No other architecture research files should be touched.

## Previous immutable evidence

From [#1201](https://github.com/vinceackermann2-sys/tam-research/issues/1201), terminal #1213 source `14c6df2600bd9f536a95b71b1da2a9398f3ae36f`:

| Five-arm control | Held-out mean NLL | Mean training tok/s |
|---|---:|---:|
| Transformer | 7.170983623 | 428187 |
| Full-width chunk-local256 | 7.206585973 | 407226 |
| Local224 predictor carry | 7.160163914 | 366662 |
| Local224 predictor reset | 7.159495192 | 370425 |
| Local224 only | 7.162791503 | 395120 |

The predictor's incremental effect is +0.002627589 NLL (3/3 wins) against local-only: **below** the preregistered +0.005 requirement. Predictor cross-chunk carry effect is -0.000668723 NLL (0/3 wins). The prior #1181 mechanism panel rejected workspace, salience and addressing support. The correct classification remains `PGW_LOCAL_PREDICTOR_CORE_UNSUPPORTED`.

## Structural question for this stage

Frozen PGW-v2 can convey information between completed chunks through **two** routes:
1. predictor previous-context carry;
2. sparse event workspace read/write.

The original cross-chunk causality checks did not isolate these two paths. This stage removes predictor output in *test instances only*, copies identical initial weights into frozen PGW-v2 and no-workspace control modules, and probes workspace-specific forward and backward information flow.

The tests show, **if they pass**, that a mathematical causal channel exists; they do **not** establish useful prediction, learned retrieval, calibrated surprise, or a model-quality gain.

## Frozen diagnostic success condition

`PGW_WORKSPACE_ONLY_PATH_EXISTS` requires:
1. past-chunk perturbation does not change its own already-fixed output;
2. past-chunk perturbation changes later output through workspace-enabled path when predictor output is disabled;
3. no-workspace control's later output is invariant to the same perturbation;
4. later-only objective produces finite, nonzero gradient to past inputs through workspace;
5. future-chunk perturbation does not affect prior outputs.

If any assertion fails, classify `PGW_WORKSPACE_ONLY_PATH_NOT_ESTABLISHED` pending a strictly separate zero-GPU repair review. Do not override frozen PGW scientific classifications either way.

## What follows

After original exact-head CPU CI, preserve the result and frame a **new independently preregistered hypothesis**, not a relabeling of the negative PGW trials.

A future PGW design would need to demonstrate that the surprise signal is aligned with downstream delayed information needs (rather than only detached normalized feature differences), show nontrivial delayed associative retrieval/update across chunks, and beat no-workspace/predictor/local-only controls at matched **total and active** capacity. A minimal artificial long-gap task could help establish a capability signal before any full language-model H100 spend, but task selection, baselines, seeds, thresholds, budgets and failure gates must be frozen **before** new training.

No GPU, new scientific seed, Modal volume write, trigger issue, rerun, architecture mutation, scaling, or breakthrough claim is authorized here. Other CPW/CHM/RLT/reduced-attention work stays untouched.
