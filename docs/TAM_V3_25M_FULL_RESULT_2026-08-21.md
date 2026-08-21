# TAM v3 25M Full Model — Final Result (2026-08-21)

This document freezes the completed end-to-end lineage for the first TAM v3 model that was pretrained and post-trained. It is a recovery/evidence record, not a claim that a 25M model is production- or agent-quality.

## Final status

**Completed and durable on Modal Volume.**

Lineage:

1. TAM v3 25M base pretraining
2. supervised instruction fine-tuning (SFT)
3. direct preference optimization (DPO)
4. final language evaluation and saved generation summary

Final checkpoint:

`/vol/full-model-runs/TAM-v3-25M-Full-seed7400/final_dpo.pt`

Final run directory:

`/vol/full-model-runs/TAM-v3-25M-Full-seed7400/`

Persistent Volume: `tam-research-data`

## Architecture

Model: TAM v3

Approximate scale: 25M parameters

Exact TAM v3 parameter count for the 25M family: 24,941,263.

Core mixer per block:

- reduced-width causal attention
- recurrent learned world-state
- learned scalar mixture gate
- no ATAM associative-memory branch in TAM v3

See `docs/TAM_V3_ARCHITECTURE.md` and `tam_research/models.py` for the frozen implementation.

## Base pretraining

Issue: #73

Seed: 7400

Dataset: FineWeb-Edu, GPT-2 tokenizer

Training-token budget: **300,000,000 tokens**

Context length: 512

Microbatch / gradient accumulation: 64 x 2

Hardware: strict Modal `H100!`

Final held-out base-model metrics:

- NLL: **4.4131**
- perplexity: **82.52**
- steady training throughput: **496,734 tokens/s**
- GPU-function elapsed when pretraining completed: **1195.3 s**

The pretraining checkpoint was committed before the original post-training budget guard fired, so the expensive base-model work was preserved.

Base checkpoint path:

`/vol/full-model-runs/pretrain/25m/ctx512-mb64-ga2/tamv3-25m-compiled-seed7400/latest.pt`

## Post-training data

Source: UltraFeedback-derived deterministic arrays prepared by the repository post-training data pipeline.

Original prepared sets:

- SFT train: 20,000 examples
- SFT eval: 1,000 examples
- DPO train: 5,000 chosen/rejected pairs
- DPO eval: 500 pairs

The completed SFT checkpoint preserved on the Volume reports:

- held-out assistant NLL: **3.6048581835**
- SFT training time: **20.4779 s**
- historical SFT compile time: **201.7430 s**

The large compile overhead was the reason the first post-training-only continuation could hit a platform timeout despite the actual optimizer work being short.

## DPO result

The final DPO checkpoint reports:

- held-out implicit reward accuracy: **0.564**
- DPO training time: **42.2382 s**

Interpretation: the DPO policy moved in the preferred direction on 56.4% of held-out preference pairs according to the reference-relative implicit reward criterion. This is above chance but modest; it should not be interpreted as strong alignment or general capability evidence.

## Final language evaluation

After SFT + DPO:

- FineWeb validation NLL: **4.6162**

The post-trained model therefore has worse raw FineWeb language NLL than the base checkpoint (4.6162 vs 4.4131), which is a common possible tradeoff when optimizing instruction/preference objectives. This must be reported rather than hidden.

The final evaluation/generation summary is stored in:

`/vol/full-model-runs/TAM-v3-25M-Full-seed7400/final_summary.json`

## Verified artifact state

CPU-only persistent-Volume verification in issue #92 confirmed:

- pretrained base: present
- `sft.pt`: present
- `sft_summary.json`: present
- `final_dpo.pt`: present
- `dpo_summary.json`: present
- `final_summary.json`: present

Thus the full lineage is durable independently of chat history.

## Continuation / failure chronology

### Issue #73 — combined pretrain + post-train

Pretraining completed successfully. The hard 1,300-second GPU budget left only ~104.7 seconds after the base stage, so the explicit budget guard refused to start post-training. Correct behavior: preserve the expensive base checkpoint rather than risk an uncheckpointed timeout.

### Issue #82 — post-training-only continuation

A 260-second H100 continuation loaded the saved base. The run was dominated by SFT `torch.compile` setup and hit the Modal platform timeout before GitHub received a normal completion callback.

A subsequent CPU-only Volume inspection initially did not see post-training artifacts while the prior container lifecycle was still resolving; later runs found durable SFT/DPO artifacts. The authoritative final verification is issue #92 after all GPU work completed.

### Issue #88 — budget-v2 eager continuation

To remove the compile bottleneck, SFT compilation was disabled and a deterministic 10k-example SFT budget path was introduced while retaining the full 5k DPO set. The first attempt failed immediately with:

`NotImplementedError: "index_cuda" not implemented for 'UInt16'`

Root cause: compact post-training token arrays are stored as `uint16`; CUDA advanced indexing does not support `uint16` tensors.

### PR #89 / issue #91 — CUDA index fix and completion

PR #89 stages the small post-training arrays into ephemeral local storage as `int64`, leaving canonical compact Volume arrays unchanged.

Issue #91 then loaded the saved model lineage, found existing durable SFT and DPO checkpoints, ran the final evaluation/generation path, and completed in **16.2 s** of that final H100 function.

Completion callback:

- SFT assistant NLL: **3.6049**
- DPO implicit reward accuracy: **0.564**
- final FineWeb NLL: **4.6162**
- final checkpoint: `/vol/full-model-runs/TAM-v3-25M-Full-seed7400/final_dpo.pt`

## What this result means

Supported:

- TAM v3 can be trained end-to-end through base pretraining, SFT and DPO with the repository pipeline.
- The 25M TAM base learned a substantially better language distribution after 300M tokens than the token-poor architecture-screen models.
- Instruction SFT and DPO checkpoints are valid and durable.
- The complete training/post-training plumbing is now exercised on real H100 compute.

Not supported:

- production-quality chat performance
- agent capability
- strong reasoning/coding ability
- superior post-training behavior versus a matched Transformer
- breakthrough claims
- general preference/alignment superiority

A ~25M model is intentionally too small to be a modern general assistant. Its purpose here is to validate the full TAM lifecycle cheaply before applying the same post-training discipline to a much larger, properly pretrained model.

## Recommended next work

Do not spend more compute merely extending this 25M model unless a specific diagnostic is needed.

When budget is available:

1. Evaluate the final 25M checkpoint on fixed chat/instruction prompts and compare base vs SFT vs DPO behavior.
2. Add a parameter- and data-matched Transformer full-training/post-training control if we want to attribute behavioral differences to TAM rather than the training pipeline.
3. Finish/resolve the 100M replicated architecture gate before promoting larger TAM scales.
4. If TAM continues to win at 100M and capability controls remain favorable, move toward a properly pretrained ~300M model, then ~1B.
5. At larger scale, add structured tool-use/agent post-training separately; persistent external memory remains a separate subsystem from TAM's within-sequence recurrent world-state.

## Key references

- Issue #73 — full-model pretraining run
- PR #72 — initial complete pretrain/SFT/DPO pipeline
- PR #77 — post-training-only continuation
- Issue #82 — first post-training continuation
- PR #84 / issue #85 — CPU-only Volume inspector
- PR #87 / issue #88 — budget-v2 eager SFT continuation
- PR #89 — CUDA uint16 indexing fix
- Issue #91 — final post-training completion
- Issue #92 — final persistent-Volume verification
