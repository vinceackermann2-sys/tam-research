# AT 64-token state-retention curriculum — frozen stage gate

**Decision:** Keep `ResidualGatedScanRLT` as the working recurrent baseline; **stop claiming success for the coupled adaptive-gain variant**, which catastrophically collapsed to a constant-class predictor in the one-seed toy curriculum. Neither model has established a language-model breakthrough, nor does a solved toy task prove the recurrent pathway itself is needed. Only CPU-only mechanism attribution is justified next.

Scope: `exp/rlt-colab`. Ground-truth immutable result `experiments/rlt/cpu/results/rlt-lastwrite-curriculum-distance-20261009-at.json`. Workflow https://github.com/vinceackermann2-sys/tam-research/actions/runs/37897164667 (success).

## Frozen protocol

- Models: residual gated-scan RLT, adaptive gated-scan RLT (coupled retention/output gate), causal Transformer, **29,504 parameters each**.
- Data: deterministic synthetic binary last-write query with 8-token easy phase and 64-token distance bins 1, 4, 16, 40, 63. An independent oracle verifies the last SET controls the label; each held-out distance bin is exactly balanced: 128 class-zero and 128 class-one examples, with final QUERY rather than answer.
- CPU: 2 threads, AdamW, batch 8, same registered seeds, **20 seconds** at LR 0.003 on easy distance=1, then **90 seconds** at LR 0.001 on mixed-distance 64-token sequences. No step cap; all actual CPU training windows met their budget. One run/seed only.

## AT results

| Model | 8-token easy after phase1 | After 64-token phase: gaps 1, 4, 16, 40, 63 | 8-token easy after phase2 | 64-token train steps | 64-token tokens/s |
|---|---:|---|---:|---:|---:|
| Static residual scan | 100% | **100%, 100%, 100%, 100%, 100%** | 100% | 5,505 | 31,313 |
| Coupled adaptive scan | 100% | **50%, 50%, 50%, 50%, 50%** | 50% | 5,521 | 31,406 |
| Transformer | 100% | **100%, 100%, 100%, 100%, 100%** | 54.30% | 13,841 | 78,740 |

The adaptive model's per-class accuracy on 64-token data is **100% class zero, 0% class one** for all gaps, with binary NLL ~0.69315. That is **collapse**, not long-range recall. Its easy-control accuracy also dropped to 50%. Both residual scan and Transformer reached near-zero long-gap binary NLL. The Transformer, however, trained on **7,086,592 tokens** versus residual scan's **2,818,560** in the same time. Therefore the equal-wall-clock result is not an equal-token/step efficiency result. The Transformer also lost most of its 8-token easy performance despite solving 64-token data: this indicates phase interference/length-specific behavior, and needs separate investigation.

Do not compare AT numerically to AR (AR capped all models at 1,000 steps and failed to learn) or AS (independent easy-task seed) as if they were replicated results. AS established all three models can learn an easy case; AT is one new seed, and the phase1 control again reached 100% for all three.

## Next CPU-only falsifiable mechanism probe

1. Train the **frozen static residual** curriculum with a **new CPU seed** and evaluate under paired inference ablations: normal recurrent state; direct encoder-only state (replace recurrent contribution with zeros); recurrent-only state (remove direct encoder skip). The models' causal encoder, causal cross-attention, local attention and FFN must remain unchanged. If encoder-only retains near-perfect accuracy, the toy result does not require the scan state, and cannot demonstrate a recurrence advantage.
2. Train the **coupled adaptive** architecture under the same registered curriculum and log long-phase gradient/activation/gate statistics and class-balanced accuracy to distinguish saturation/optimization collapse from generalization failure.
3. Any new parameter-neutral gain variant must be individually tested for causality, finite gradients and parameter count **before** CPU training and must remain a new architecture hypothesis, never retroactively labeled as success.
4. No GPU, large-language scientific comparison, replication seeds, 250M/5B scaling, or breakthrough authority is granted by this memo.

Use unique CPU IDs and seeds; don't mutate AR, AS, AT result files or claims, don't rerun their workflows, and don't touch `main`.
