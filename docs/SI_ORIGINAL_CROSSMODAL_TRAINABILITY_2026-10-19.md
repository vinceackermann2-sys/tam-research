# Original user SI architecture: first joint cross-modal policy-learning gate

**2026-10-19 · SOURCE-AUTHORED CPU DEVELOPMENT ONLY · no scientific seed, GPU or general intelligence claim**

## Why this task is next
Original screenshot architecture requires actual joint integration and training across real distinct input modalities with adaptive reasoning, not just ability to call the encoders. Parent [PR #1523](https://github.com/vinceackermann2-sys/tam-research/pull/1523) verified 17 integrated causal/mechanism smoke tests and the full existing repo suite, **but did not perform any genuine policy learning or baseline comparison**.

This PR's sharply limited FIRST TRAINABILITY task evaluates four public cues from separate input types and a fifth (motor) query: text token, 8×8 RGB image, 32-value audio waveform, and a two-frame video. Independently sample a class among four for each cue; the final motor token is a one-hot query index selecting WHICH earlier cue is needed. The public-action label is that earlier cue's class. The motor token does not contain or predict that class by itself; the output needs previous sensory context. All five inputs use the actual modality-specific encoders from the integrated V1, not modality-ID tags only.

The environment is self-authored for development, with intentionally trivial, independently decodable scalar patterns for modalities. **This is not a measure of realistic visual/language/audio/video understanding. A hand-coded classical decoder solves it perfectly; chance without past cues is 25%.** No hidden world, reward oracle, paid information acquisition, proprietary heldout task or pretrained feature network is involved.

## Exact source and control
Uses exact `SIOriginalIntegratedV1` from PR #1523, hence real original source's `SIHybridBlock` (alternating attention and gated state recurrence + actual top-1 sparse dispatch), five learned encoders, shared ACT-style latent reasoning, same action head, verifier, world-model/replay/memory/fast interfaces. **For this narrow trainability gate**, retrieval, online fast updates and untrained planning/search are explicitly disabled. Their earlier integration tests remain valid but **this is not full joint training of ALL mechanisms**, and no claim to that effect is permitted.

`SIIntegratedTransformerControl` inherits the identical integrated class and replaces ONLY core `SIHybridBlock` with the earlier source's `DenseBlock` (full causal Transformer-style attention). All matching surrounding parameter names and shapes are initialized to bit-identical weights between the two candidates. No change to user-required model source or previously frozen negative evidence. **Model parameters/active train FLOPs are NOT matched**, so no statistically or compute-fair winner may be claimed.

## Frozen fixed development recipe, before seeing learning scores
- One initial development initialization fixture `120312` per model family; **not a previously consumed or authorized scientific seed**.
- Width `24`, `4` backbone layers, `4` sparse experts (original hybrid), adaptive latent budget `3` with learned halting, five distinct encoders, four possible output actions.
- Exact training IDs `11001..11096`, disjoint development validation IDs `21001..21008`; fixed train size `96 updates × batch 16 = 1,536 examples per model`, validation `8 × 32 = 256 examples`. Each example has 5 public tokens. No restarts/tuning after result; no heldout.
- Same AdamW `lr=0.003`, weight decay 0, cross-entropy action objective and actual sparse routing balance regularization `0.001` where applicable, gradient norm clip 1, CPU threads 2. One optimizer update on hybrid versus Transformer does **not** imply equal FLOPs; report parameter count and train wall-time for context, not full compute audit.
- Validation for BOTH architectures uses the exact same seen-by-neither-development-training fixed generator IDs. Report initial training CE, final training CE, validation CE, overall accuracy, accuracy by which of four modalities was queried, measured training seconds, and a **shuffled-cue ablation**: shuffle the public sensory cues independently across examples while keeping the original motor query/label fixed. This is a perturbation diagnostic, NOT a scored counterfactual-world experiment. A truly learned temporal binding rule should lose accuracy under cue shuffling.
- The only nonlearned comparators here are **random/query-only chance 0.25** (in expectation) and a trivial classical decode-then-select policy with **perfect 1.0 accuracy** on these deliberately simple encoded patterns. No other baseline or physics/SI claim is warranted.
- The code is deterministic under frozen CPU torch seeds as far as current PyTorch build permits; wall clock duration can vary. No checkpoint trained here is promoted to a scientific model, and no training seed is consumed from the v25/26 scientific ledger.
- Development decisions: if trained models fail to exceed random chance, record a **negative trainability finding for this tiny budget**. If training loss falls or validation exceeds chance, it is evidence of limited cross-modal learning only, NOT a winner versus Transformer or a memory/world-planning breakthrough. Do not tune task, optimizer, seed or update count on the basis of an observed result.

## What is still needed to actually test the user's integrated proposal
This exact same new integrated model must NEXT be trained with *real executed causal experience*, with its actual per-world long-term memory retrieval, rank-two online fast weights, replay consolidation, action-conditioned multi-step latent model, learned verifier/search decision and adaptive compute all in active task-dependent feedback loops. An external environment with honest paid actions and long-term memory demands must be **independently chosen**, not retrospectively invented to favor one candidate. Fully equipped Transformers need identical external memory/search/adaptation and matched active FLOPs, tokens/episodes, compute wall-time and input datasets. Demand skill retention under sequential new tasks and realistic cross-modal transfer. Then independent frozen/sealed heldout seeds/ledger and preregistered STOP thresholds. These experiments require authorization before scientific/paid resources.

### Explicit status
- Parent integrated user architecture: **engineering pathways implemented/connected**; no measured learning advantage.
- This PR: first narrow CPU development cross-modal **policy learning**, not all-mechanism joint learning.
- Earlier original 64,614-parameter v26 proxy NO-GO and later PCMA NO-SCALE **remain valid for their own prototypes**, not this integrated architecture.
- User's COMPLETE original proposal: **scientifically untested**, no breakthrough verdict.
- Canonical original proxy source Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` MUST remain intact.
- Draft PR only; no main write/merge, GPU/Colab/Modal cost, scientific attempt or unapproved heldout.
