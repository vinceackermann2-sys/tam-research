# Original SI / PCMA-v1 — Actual shared ten-action neural backbone engineering

**2026-10-19. DEVELOPMENT CPU ONLY. NOT scientific training or policy comparison. No GPU, heldout or research seed consumed.**

## Lineage and immutable original
This isolated draft directly builds on CI-passed external bandit triage PR #1519; it traces to the user's original screenshot-inspired SI hybrid v0, and no other research project. The canonical original source `tam_research/si_original_hybrid_v0.py` must remain at exact Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`, with `main` unchanged. Its original scientifically evaluated 64,614-parameter v26 hybrid remains **NO-GO**. Original v25 seed117214 was duplicated and its second use remains quarantined. Our custom PCMA history toy produced a weak/negative development signal (**NO-SCALE**) and we must not retrospectively retune it. POPGym's elementary memory tasks were solved perfectly by cheap classical memory, which precludes a neural-superiority claim on those scores.

The external published POPGym 1.0.7 MultiarmedBanditEasy has 10 possible actions, 200 steps, and real random unknown per-arm Bernoulli payout rates. Eight fixed developer worlds in PR #1519 showed Beta(1,1) Thompson sampling return **+0.6675**, UCB1 **+0.36875**, round-robin **−0.0050**, and fixed arm **−0.10125**, all using solely executed chosen-arm public feedback. Those are not scientific results. The environment's reset and step `info` reveals the TRUE private arm payout probabilities: **never pass `info` to either model, any classical policy, replay storage, or an evaluator's comparison logic.**

## First actual neural integration, keeping original source unchanged

`tam_research/si_pcma_v1_bandit_neural_adapter.py` constructs:
- `OriginalSIHybrid(SIConfig(obs_dim=3, action_dim=10, max_seq_len=16))`;
- `TransformerControl` with **the exact same** ten-action SIConfig;
- the same `nn.Linear(3,1)` payout logit head, modality semantics, causal query masking and Bernoulli cross-entropy training objective.

Both see exactly the same **PUBLIC causal** inputs: last seven real `(chosen action one-hot (10), binary observed payout, normalized time)` feedback tokens and a final query token of a candidate action and normalized remaining horizon. The query reward channel is literally **zero** and modality=1, so it cannot leak its own label; all earlier factual events have modality=0. Full histories are validated for chronological step order, exact actual payout↔reward correspondence, absence of present/future target, duplicates, and incomplete most-recent feedback before truncation. The target is the **actually observed payout of that particular executed query action** (0/1), not privileged posterior arm probability or imagined reward of an unchosen arm.

Data source: one fixed round-robin execution of **four development-only episodes 101–104**, logging 39 chosen-action examples/episode (actual steps 10,15,...,200). A strictly disjoint **two development-only episodes 201–202** supply 78 actual validation examples, on the exact same logged actions for both backbones. The round-robin logging policy is low-cost and provides action support without peeking at unchosen outcomes. Those IDs are **developer fixture integers only**, not available scientific seed reservations. Critically, this dataset is NOT the old POPGym 41–48 baseline comparison and may not be treated as its heldout result.

Both backbones receive exactly **4 AdamW updates**, batch size 4, lr 0.001, clip-norm 1.0, fixed torch developer init/random example ordering, identical training and validation rows, CPU-only, no pretrained weights, no feature from a hidden Markov state, no return-policy rollout. CI runs this bounded neural training and prints **factual Bernoulli validation BCE**, model parameter counts and observed developer CPU training wall-time, after installing `popgym==1.0.7` only on this draft's CI branch. Tests verify actual ten-arm original source, causal query target masking, invalid/replayed/future feedback rejection, data IDs, same shaped inputs, finite predictions and four-update smoke.

**No claim of compute matching:** equal training rows/updates do NOT imply equal FLOPs, active expert execution, adaptive refinement or inference latency. The original hybrid/Transformer parameter counts and CPU wall-time are disclosed; these are not controlled/hardware-reproducible enough for scientific compute parity. Original hybrid may use an attention+recurrent+MoE+refinement path while Transformer remains dense attention. The project already proved this architectural distinction cannot be assumed to beat simple controls.

## What is still missing for a REAL verdict
- **A meaningful training budget** with independently frozen training seeds, network checkpoints/digests, repeatable optimizer data schedule, measured ACTIVE train/inference FLOPs, no post-hoc tuning.
- **Online policy behavior, not just prediction:** an identical ten-arm action decision/feedback interface for hybrid and Transformer, same bounded PCMA factual memory and rank-2 online adapter, exploration/ablation budgets and 200-step episode rollouts. None exists in this new adapter yet. Evaluating a supervised BCE does NOT compare actual acquisition policy return or ability to explore.
- **Strong factual competitors:** Beta Thompson (+0.6675 on development), UCB1, contextual/finite-state/classical beliefs, compute-expanded Transformer and matched-PCMA Transformer with equal information, update budgets and resets.
- **Independent heldout unseen worlds**, protected privileged info, a completed global consumed-seed ledger and durable one-use attempt reservation, prespecified realistic effect sizes, world-level clustered uncertainty and STOP/GO thresholds.
- **Explicit research execution authorization** before a sealed scientific trial; no retries/reshuffling of scientific seeds to rescue an unfavorable result.

## Gate for next iteration
The immediate engineering pass/fail is simply: can the unchanged original SI hybrid and Transformer both process same public ten-arm factual training records, finish the identical frozen four-step developer optimizer check, and yield finite independently collected development-validation BCE? If so, this validates the adapter, NOT architectural advantage. A genuine final PCMA scientific verdict remains **UNKNOWN** and original unchanged v26 remains **NO-GO**. Do not scale GPU compute from this smoke.
