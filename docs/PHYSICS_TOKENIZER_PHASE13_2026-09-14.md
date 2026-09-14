# Physics Tokenizer Phase 13 — context-sufficient unseen NLS confirmation

Status: **PASS** under the preregistered 12/12 gate.

Phase 13 directly tested the failure diagnosis from Phase 12. The representation, compander, feature basis, ridge regularization, wave+Gray-only representation-training mixture, unseen NLS family, and evaluation regimes were held fixed. The only scientific change was increasing context from 3 frames (2 observed transitions) to 8 frames (7 observed transitions).

No NLS trajectory was used to fit state/innovation normalization statistics, residual-VQ codebooks, compander thresholds, features, or ridge strength.

## Registered result

Three fresh registered replicates were executed exactly once with no retries or replacement seeds.

Companded projected rollout MSE / raw-state persistence:

- NLS ID h8: 0.2359, 0.1375, 0.4329 — **3/3 pass**, median **0.2359**.
- NLS spectral OOD h3: 0.2299, 0.0972, 0.4017 — **3/3 pass**, median **0.2299**.
- NLS parameter OOD h3: 0.1465, 0.0749, 0.1666 — **3/3 pass**, median **0.1465**.
- NLS combined parameter+spectral OOD h3: 0.0438, 0.0239, 0.0865 — **3/3 pass**, median **0.0438**.

All **12/12 preregistered cells** beat raw persistence.

## What changed relative to Phase 12

Phase 12 used only three decoded context frames. It failed NLS ID h8 in all three registered replicates, despite extremely low target-token oracle error and very strong rollouts when the dynamics map was fit from raw context. The post-hoc context diagnostic showed that decoded transition noise was large relative to the very small one-step NLS signal.

Phase 13 uses the same generic absolute-next-state ridge identifier but estimates it from seven observed transitions instead of two. This additional evidence is sufficient to average down transition-estimation noise and restore robust ID and OOD performance on the unseen NLS family.

The result therefore supports a more precise conclusion: the Phase-12 boundary was **context/evidence sufficiency under quantized observations**, not an inability of the frozen representation to encode the NLS state or of the frozen feature library to express the NLS update.

## Remaining limitations

The representation is still trained only on two synthetic PDE families and evaluated on three known synthetic PDE families. The feature basis is a hand-specified generic local spatial+cubic library. Eight frames were selected after a post-hoc exploratory context-length diagnostic and then separately preregistered and confirmed here. This does not establish a universal scientific tokenizer, unrestricted equation discovery, novel physics, AGI, or SSI.
