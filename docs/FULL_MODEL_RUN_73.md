# TAM-v3-25M-Full — run #73

This document records the first end-to-end TAM language-model training pipeline. It is separate from the Transformer-vs-TAM scaling-law experiments.

## Goal

Produce one complete TAM checkpoint lineage that has passed through:

1. causal language-model pretraining,
2. supervised instruction fine-tuning (SFT),
3. preference post-training with Direct Preference Optimization (DPO),
4. held-out evaluation and fixed prompt generation.

This is a research-scale chat model, not an agent-quality model and not evidence of broad capability superiority.

## Frozen configuration

Preregistered in issue #71; launched by issue #73.

- architecture: TAM v3 (attention + recurrent world-state)
- parameters: ~24.94M
- seed: 7400
- tokenizer: GPT-2, unchanged 50,257-token vocabulary
- context: 512
- pretraining corpus: FineWeb-Edu `sample-10BT`
- pretraining budget: 300,000,000 tokens
- pretraining optimizer: established TAM research AdamW/cosine setup
- pretraining microbatch/accumulation: 64 x 2
- pretraining eval/checkpoint cadence for this long run: every 50M tokens
- SFT dataset: `HuggingFaceH4/ultrafeedback_binarized`, `train_sft`
- SFT format: `User:\n{prompt}\nAssistant:\n{response}<eos>`
- SFT loss: assistant tokens only
- SFT target: 20,000 fixed 512-token sequences (~10M sequence tokens before masking)
- DPO dataset: UltraFeedback Binarized `train_prefs`
- DPO pairs: 5,000
- DPO beta: 0.1
- held-out SFT rows: 1,000
- held-out preference pairs: 500
- fixed generations: six prompts covering explanation, Python, troubleshooting, planning, arithmetic correction, and email writing

## Budget safety

The user reported $2.85 compute credit remaining before this launch, while the already-running 100M gate (#67) still existed.

The new full-model H100 function therefore has a hard Modal timeout of **1,300 GPU seconds**. At the verified H100 rate of $0.001097/s, the H100 portion cannot exceed about $1.43 before CPU/RAM charges. An internal reserve check refuses to enter post-training if pretraining consumes nearly the entire ceiling. Pretraining checkpoints are committed before post-training begins.

Only one full-model run is launched. No replica seed or larger scale is part of #73.

## Isolation / storage

Data preparation uses a distinct path so it cannot rewrite the dataset shard being read by the active 100M experiment:

- FineWeb: `/vol/data/fineweb-edu-gpt2-full25m`
- post-training arrays: `/vol/data/ultrafeedback-gpt2-full25m`

Artifacts:

- pretraining root: `/vol/full-model-runs/pretrain`
- full-model root: `/vol/full-model-runs/TAM-v3-25M-Full-seed7400`
- SFT checkpoint: `sft.pt`
- final preference-trained checkpoint: `final_dpo.pt`
- final summary: `final_summary.json`
- full pipeline summary: `pipeline_summary.json`

All code is on `main` after PR #72. Authoritative runtime callbacks are posted to issue #73.

## Evaluation outputs

The run records:

- pretrained FineWeb held-out NLL/PPL,
- SFT held-out assistant-only NLL/PPL,
- DPO train implicit-reward accuracy,
- held-out raw chosen-vs-rejected accuracy,
- held-out reference-relative implicit-reward accuracy/margin,
- final FineWeb NLL/PPL after post-training,
- six generated responses.

Fill this section from issue #73 / `pipeline_summary.json` after the run completes. Do not invent missing values.
