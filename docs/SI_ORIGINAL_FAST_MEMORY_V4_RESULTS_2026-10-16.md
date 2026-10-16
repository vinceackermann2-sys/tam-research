# SI Original fast episodic memory v4 — frozen three-seed CPU results (2026-10-16)

**Verdict:** huge improvement from introducing per-episode fast weights, but **NO-GO for claiming the original hybrid backbone is superior**. All predeclared hard advancement conditions did not pass; no paid GPU or breakthrough authority. Another simple model given the SAME memory performs as well or better, while ridge classical system-ID still dominates the heldout nonlinear task.

## Execution integrity

- Protocol `docs/SI_ORIGINAL_FAST_MEMORY_V4_PROTOCOL_2026-10-16.md` committed before running fresh engineering seeds **97511/97512/97513**, each completed once, no retry or failed attempt. Three CPU training runs × four conditions, 450 optimizer steps/model, batch16, 20 timesteps, three deterministic training families (direct, delayed, switch) and held-out nonlinear family. 3×64 independent evaluation worlds/seed/family and actual goal-directed action scoring.
- Frozen source `scripts/si_original_fast_memory_v4.py` at `23c43405d70bd110f876798037e075c5d97eb7f1`, **GitHub blob `2f35163986d5bc9f8d4f9e0feb1054b2f6e9b937` exactly matches locally executed file**; local SHA256 `418cdec0d90c912bb83c9543ce08f4c426447362420bc544d306bbee398452e6`. Local execution relied on the previously disclosed functionally matched partial copy of the original hybrid/Transformer module (nominal counts identical), not a byte-for-byte verified complete checkout of the parent GitHub source. Thus this remains engineering rather than high-authority scientific replication until a source-matched original-branch run verifies it.
- Four matched conditions: unchanged **OriginalSIHybrid** (64,614 params), unchanged **TransformerControl** (64,637), hybrid + **same** learned rank16 online delta-rule memory (65,527), Transformer + **same** memory (65,550). The added learned feature-keyed memory costs **913 parameters** in either case and updates only from *completed earlier transitions*, never from the unobserved query result. Online update rule, forgetting factor and feature shape are architectural priors; gradients learn the keys and write rate, not the entire memory algorithm. Memory resets every episode.
- Synthetic simulator matrices and true candidate consequences remain evaluator-only. Controls are uniform random actions, parameter-free all-history ridge, and last-6-transition ridge trained solely on preceding observed states/action pairs.
- Local integrity checks: exact next-state consistency for true candidate action, finite nonzero gradients into memory key encoder, strict causal prefix invariance when future observations are changed, no target leak through action query, and identical memory-parameter increments. Repo-local tests included for original exact-head CI.

## Three-seed mean action-selection accuracy (%) — higher is better

| Simulator | Bare hybrid | Bare Transformer | Hybrid + fast memory | Transformer + same memory | Ridge all | Ridge last-6 | Random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 26.39 | 27.26 | **94.79** | 94.97 | **97.57** | 81.08 | 24.83 |
| Delayed | 26.39 | 28.65 | 48.96 | 49.13 | **73.78** | 41.32 | 25.00 |
| Switch | 26.39 | 25.69 | 81.08 | 81.42 | 59.72 | 79.34 | 24.65 |
| **Unseen nonlinear** | 23.78 | 26.74 | 92.71 | **93.06** | **94.79** | 78.12 | 24.31 |

Three new engineering seeds **97511, 97512, 97513**, 576 held-out action choices per family, generated consistently per seed/model.

**Primary unseen nonlinear per seed**:
- 97511: hybrid memory **0.9427**, Transformer memory **0.9427** (tie).
- 97512: hybrid memory **0.9427**, Transformer memory **0.9427** (tie).
- 97513: hybrid memory **0.8958**, Transformer memory **0.9063** (Transformer better).
Thus hybrid memory wins **0/3** matched seeds and loses the mean to Transformer+same memory. Ridge-all still beats both, so the preregistered architectural and classical-control gates FAIL.

## Prediction accuracy and actual regret

| Family | Hybrid+memory query MSE | Transformer+memory query MSE | Hybrid+memory regret | Transformer+memory regret |
|---|---:|---:|---:|---:|
| Direct | 0.0257 | **0.0244** | **0.0049** | 0.0050 |
| Delayed | 0.0742 | **0.0725** | **0.0830** | 0.0883 |
| Switch | 0.1031 | **0.1027** | 0.1039 | **0.0975** |
| Unseen nonlinear | 0.0323 | **0.0312** | 0.0092 | **0.0082** |

Reference CPU training wall-seconds per model, mean across seeds: bare hybrid **9.15 s**, bare Transformer **5.11 s**, hybrid+memory **13.29 s**, Transformer+memory **9.42 s**. Hybrid-memory is **~1.41× slower** than Transformer-memory in this unoptimized implementation; not equal-training-FLOPs or inference-efficiency evidence. Learned update sigmoid(eta) hybrid per seed [0.7225,0.7196,0.7209], Transformer [0.7206,0.7215,0.7219]. Very similar: mechanism more important than backbone.

## Interpretation / go-no-go

A modest differentiable rank-16 fast episodic matrix shifts next-action decisions from ~25% random chance to ~93% on the unseen nonlinear family. That is a **useful targeted causal-memory feasibility result**, not a general intelligence, new-science or architectural breakthrough: a hand-designed delta-rule handles much of the structure; ridge achieves better accuracy without neural pretraining. This original SI hybrid does not yet demonstrate a competitive advantage over equal-memory Transformer.

**STOP unchanged original hybrid+memory as a paid-GPU architecture candidate.** Further CPU work may explore whether learned episodic memory **genuinely generalizes across diverse delayed/nonlinear latent worlds where ridge is weaker**, and optimize implementation cost, but it must compare against baseline ridge, Transformer+same memory, and no-memory. A better research candidate would learn to detect world family/hypothesis uncertainty and update distinct memory channels, *not* merely bolt on a solver whose advantage was already visible.

Original vision components **still absent**: persistent memory consolidation/replay, text-image-audio encoders, verified multi-step planning, adaptive compute savings, and long-context hardware/FLOP evidence. No Modal credits consumed or approved. Keep isolated draft branch and exact-head CI; do not merge directly to main.
