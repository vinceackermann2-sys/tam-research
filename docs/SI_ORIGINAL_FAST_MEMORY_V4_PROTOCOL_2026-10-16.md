# SI Original fast episodic memory v4 — CPU engineering protocol (frozen 2026-10-16)

**Branch:** `experiment/si-original-fast-episodic-memory-v4-20261016`, stacked on original SI planning gate PR #1475 -> original SI hybrid PR #1459. This is the **original hybrid architecture**, not EHW. No modifications to main or other projects, no Modal/Colab/GPU/automation.

## Motivation
The original recurrent-attention-MoE hybrid and parameter-matched Transformer both performed near random when choosing one of four actions toward an actual outcome goal on previously unseen worlds. A strong parameter-free online ridge system-ID baseline achieved ~97% on unseen direct/nonlinear worlds; thus the experience contains learnable information. Original hybrid's `EpisodicMemory` interface is not fused into its predictive forward path.

## Mechanism
`scripts/si_original_fast_memory_v4.py` imports both **unchanged** `OriginalSIHybrid` and `TransformerControl` from the original model branch. Adds *the same* learned feature-keyed rank-16 fast memory to either model:
- key `k_t = normalize(MLP([state_t,action_t,previous_action_t]))`
- read `r_t = k_t^T M_t`; predict `next_state_t = backbone_t + r_t`
- AFTER predicting t, write only the *completed* observed transition `M_(t+1)=0.985 M_t + sigmoid(eta) outer(k_t, obs_(t+1)-pred_t)`, where the outcome is observed at the **next** timestep.
- memory matrix resets at every episode; no hidden law or future query outcome enters reads. Gradient flows through the learned key encoder and recurrent write sequence during training. This is a trainable memory *read/write* mechanism, although the update formula itself is a hand-designed normalized delta-rule. **Do not mistake this prior for autonomous lifelong learning or a breakthrough.**
- Four conditions: bare OriginalSIHybrid, bare Transformer, hybrid+fast memory, Transformer+identical fast memory. Base models 64,614 vs 64,637 nominal params; memory adds 913 params to both (65,527 vs 65,550). No claimed active-FLOP parity. The existing `EpisodicMemory` and modality scaffolds are not magically made into complete cross-session memory or language/image/audio training.

The EXACT SAME `world`, training/evaluation streams, strong ridge-all and ridge-recent no-training controls, goal scoring and unseen nonlinear family as the v3 planning gate are reused. Four candidate action outcomes are independently evaluated against the **true** simulator. Training distribution: direct, delayed, switching action dynamics. Heldout OOD: nonlinear state interaction, with new randomized matrices.

## Predeclared bounded CPU gate
Implementation smoke: **97300**, 3 optimizer steps, batch4, for numerical shape/causality only; NOT data used as confirmation. The smoke demonstrates the mechanism outputs finite tensor values but is not a win claim. Freeze source/protocol BEFORE running confirmation.

**New fresh CPU engineering seeds 97511, 97512, 97513**, not previously consumed. Four models trained on exactly matching 450 AdamW updates, batch16, seq20, LR=.001, WD=.01, gradient clip 1.0 and CPU two threads. For each family/model/seed evaluate 3×64 worlds with seed*1000000+75000+19*j. No retries or reusing consumed seeds. Record per-seed accuracy, regret, action-counterfactual prediction MSE, learned write rate, parameter counts and actual CPU training time.

**Primary predeclared:** on unseen nonlinear family, compare **hybrid+memory vs Transformer+same memory** for accuracy, regret and MSE, and compare both against **ridge-all and ridge-recent**, with paired per-seed reporting. A positive signal to justify another **CPU** follow-up requires hybrid+memory > Transformer+memory mean accuracy and >=2/3 seed wins, lower mean regret, and > either ridge control on nonlinear action accuracy **or** meaningful improvement on heldout nonlinear query MSE against ridge where ridge is insufficient (with independently frozen baseline evaluation); memory must yield a clear gain over bare hybrid *at similar compute*. Because ridge already approaches 97% accuracy on this task, this is intentionally a hard gate. It is NOT permission for GPU expenditure or scientific breakthrough even if passed.

**Fail/no-go** if memory-augmented hybrid is no better than Transformer+identical memory, cannot approach strong ridge baselines or incurs excessive training/inference cost. A hand-designed LMS update may be the reason for improvement; strong controls prevent assigning credit solely to the hybrid recurrent/MoE architecture. Keep negative evidence, never alter thresholds after results. Scientific claims require fresh independent families, robust controls, bounded compute/accounting, many more seeds and source-matched execution.

Before scientific promotion, verify the GitHub original-module bytes in the runtime: local v3 engineering runs used a functionally matched subset, not a byte-identical full GitHub module (explicitly disclosed in v3 report). Frozen attempts only, no paid runs, and no main merge.
