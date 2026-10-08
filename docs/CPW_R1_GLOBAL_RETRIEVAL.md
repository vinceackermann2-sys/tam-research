# CPW-R1: one competitive global retrieval head

Preregistration for engineering-only work: [issue #1314](https://github.com/vinceackermann2-sys/tam-research/issues/1314).

## Why this control exists

CPW-v2 sequence-only shows an encouraging equal-token/equal-H100-screen language-loss and compute signal at 25M tokens, but its dependency horizon is limited by depth. WORLD_LAST1 and AFM_LAST1 both failed the old recall test. Crucially, the old test was later shown to have a perfect query-blind positional shortcut (#1301). The corrected **binding-required** generator (#1307) is therefore the only acceptable synthetic memory test for future candidate evaluations.

CPW-R1 is a deliberately conservative *hybrid control*: one ordinary narrow causal attention block plus fourteen cheap local sequence blocks. It is **not** an original alternative to attention.

## Frozen model parameters

- d_model: 256
- 15 blocks: blocks 0–13 are unchanged sequence-only local predictors
- block 14: replace exactly one low-rank 256 → 96 → 256 SEQUENCE mixer with single-head causal self-attention of QKV inner width 48, head count 1 and output width 48
- replacement mixer: `4 × 256 × 48 = 49,152` weights
- removed mixer: `2 × 256 × 96 = 49,152` weights
- complete model: **21,745,408 parameters**, unchanged from CPW-v2 sequence-only
- baseline Transformer: **24,940,288 parameters**

The attention block uses ordinary scaled dot-product attention with a strict causal mask. It is *not* a fast-weight recurrent matrix and does not have constant-size state; it brings back quadratic sequence cost in exactly one layer. Parameter parity is not FLOP parity.

## Hypothesis

A narrow final global retrieval head could use the local representation of source key/value pairs and the final query key to make a content-dependent selection, without requiring all 15 blocks to be Transformers. This is a test of **competitive selection** and a potentially cheaper hybrid architecture, not a claim that it will beat an equal-compute Transformer.

## Integrity gates

Zero-GPU tests must establish:

1. exactly one global attention mixer at block 14;
2. exactly the same total number of model weights as sequence-only;
3. blocks 0–13, embeddings, FFNs and the output head remain byte-identical under paired initialization;
4. perturbing future tokens cannot affect earlier logits;
5. a token beyond sequence-only's local receptive field can affect the final hybrid prediction;
6. the corrected #1307 generator supports same-prefix, different-query-key counterfactuals;
7. a CPU-only query-target backward pass reaches QKV and output weights with finite gradients.

Passing these says **nothing** about learned retrieval accuracy. It qualifies a candidate for a future separately preregistered H100 paired scientific benchmark with fresh seeds, new namespace, full positivity/negative/shortcut controls, and explicit one-shot authority. No GPU task or result namespace exists in this issue.

`breakthrough_claim_allowed=false`  
`gpu_authorized=false`  
`scale_up_authorized=false`
