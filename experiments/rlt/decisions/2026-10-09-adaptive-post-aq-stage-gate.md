# Adaptive dual-path RLT — AQ engineering stage gate

**Decision: no language-model breakthrough; stop 15M same-family quality sweeps, replication and scale until there is an independently testable new hypothesis.** Preserve AQ as a frozen engineering run. Scope: `exp/rlt-colab`, 15,129,344 parameters, seq 64, batch 64, pinned FineWeb-Edu D, A100, 60 seconds of postcompile training per candidate, cosine LR 1e-3 and shared train/eval streams.

## Ground-truth AQ (one engineering seed, 20261060)

| Candidate | Held-out NLL (lower better) | Train tokens/second | 60s train tokens |
|---|---:|---:|---:|
| Adaptive dual-path scan | **5.6400763243** | **280,507.75** | 16,834,560 |
| Static residual scan | 5.6501312852 | 275,481.79 | 16,531,456 |
| Transformer | **5.5447209030** | **316,886.10** | 19,017,728 |

Within AQ: adaptive versus static residual **−0.01005496085 NLL**, throughput ratio **1.01824425**; adaptive versus Transformer **+0.09535542130 NLL**, throughput ratio **0.88520054**. Transformer remains the held-out language-quality leader in AQ. The adaptive optimizer was borrowed from independently calibrated static residual AO; AQ is NOT a calibrated, fresh-seed scientific architecture comparison.

Frozen sources:
- Result: `experiments/rlt/modal/results/rlt-systems-adaptive-realdata-modal-20261008-aq.json`
- Job prereg: `experiments/rlt/modal/jobs/rlt-systems-adaptive-realdata-modal-20261008-aq.json`
- Claimed attempt: `experiments/rlt/modal/claims/rlt-systems-adaptive-realdata-modal-20261008-aq.json`
- Model: `experiments/rlt/model_gated_scan_adaptive.py`
- AQ preflight: `experiments/rlt/modal/diagnostics/modal-actions-systems-adaptive-realdata-20261008-aq.json`
- One-shot workflow: https://github.com/vinceackermann2-sys/tam-research/actions/runs/37846247264
- CPU prototype correctness: https://github.com/vinceackermann2-sys/tam-research/actions/runs/37844155104 (**29 passing tests** for the gated-scan, static residual, and adaptive model families).

The originally triggered AQ workflow briefly wrote its preflight to AN's diagnostic path. AQ preflight was preserved separately; AN diagnostic restored to original blob `3227154e44bed40b56f13f52dd700524e7977854`. A later corrected AQ workflow was queued but failed on the already-frozen claim guard; it is **not another scientific or engineering attempt**. The claim and result are immutable.

## Relation to previous evidence

AN (engineering, independent seed) demonstrated static residual scan better than scan-only by **−0.057706 NLL**. AP (scientific, independent seed) measured static residual scan worse than Transformer by **+0.031907 NLL**. AQ (different engineering seed) measured adaptive scan better than static residual by **−0.010055 NLL** but worse than Transformer by **+0.095355 NLL**.

**Do not make direct cross-seed arithmetic claims about adaptive versus AP.** These runs do not establish the intrinsic benefit of adaptive gating on language modeling.

## Next zero-GPU falsifiable question

Does explicit recurrent state improve **state retention across long irrelevant-token spans**, compared to the same-size short-context causal Transformer? The models all contain causal attention paths, so recurrence may offer *no* advantage. Evaluate this instead of presupposing one.

Prepare a CPU-only synthetic last-write-state task with controlled last-update distance, no labels leaked into the prompt, deterministic train/eval seeds, balanced targets, readout at query, and frozen model/compute configuration. Report per-distance-bin accuracy, cross-entropy, parameter counts and timing, including a no-learning/untrained sanity check. Any toy result is **not** general-language evidence; CPU diagnostics are exploratory and cannot grant GPU or scaling authority.

## Governance

No direct writes to `main`. Never rerun AM, AN, AO, AP or AQ, nor consume seeds `58232`/`58233` without independent explicit authorization. New paid experiments require a unique preregistered job, unconsumed seeds, pinned hashes, preflight and durable GPU claim, `retries=0`, no resume. No scientific replication, 250M/5B scale or breakthrough claims are justified at this gate.
