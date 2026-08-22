# TAM v3 100M / 2B Full-Training Protocol

Status: preregistered before the production run is launched.

## Goal

Train the first capability-oriented TAM v3 model at ~100M parameters with an approximately Chinchilla-style 20 tokens/parameter data budget, then perform instruction SFT and preference DPO before judging downstream behavior.

This run is distinct from the earlier 10M-token 25M/50M/100M architecture screens. Those screens measure early learning/sample efficiency; this run asks what TAM v3 can do after a serious pretraining budget.

## Model

- architecture: TAM v3 (reduced-width causal attention + recurrent world-state)
- scale: `100m`
- exact parameters expected: 101,806,616
- tokenizer: GPT-2 BPE, vocab 50,257
- context: 512
- precision: bf16
- seed: 8100
- micro-batch: 64
- gradient accumulation: 2
- global sequences/update: 128
- optimizer: fused AdamW, betas (0.9, 0.95), weight decay 0.1
- peak LR: 3e-4
- warmup: 2%
- cosine decay
- grad clip: 1.0
- compiler: `torch.compile`, existing version-safe persistent Inductor cache

## Pretraining budget

Exactly 2,000,000,000 training tokens plus 5,000,000 held-out validation tokens.

For 101,806,616 parameters this is ~19.64 training tokens per parameter, i.e. the intended ~20x regime.

### Curated mixture

| Source | Train tokens | Fraction | Purpose |
|---|---:|---:|---|
| FineWeb-Edu sample-10BT | 900M | 45.0% | broad educational language/world knowledge |
| FineMath-4+ | 350M | 17.5% | worked mathematics and mathematical reasoning |
| Common-Pile StackV2 Edu filtered, score >=3 | 300M | 15.0% | openly licensed code/software knowledge |
| Cosmopedia OpenStax | 150M | 7.5% | synthetic textbook-style science/education |
| Cosmopedia Stanford | 150M | 7.5% | synthetic higher-education material |
| Common-Pile open-license ArXiv papers | 150M | 7.5% | real scientific text incl. physics/quantitative biology/CS/math |

The dataset builder records exact source metadata and prepares each source as a resumable token shard before final concatenation.

Licensing/provenance choices intentionally avoid non-commercial-only datasets. See upstream dataset cards before any external release of model/data artifacts; FineWeb-derived content can retain source-publisher rights even when the dataset database is ODC-BY.

## Post-training

### SFT

- dataset: `HuggingFaceTB/smol-smoltalk`
- rationale: explicitly curated for small 135M/360M-class models
- train examples: 100,000
- held-out examples: 2,000
- assistant-only causal loss
- LR 5e-5
- micro 64 x grad-accum 2

### Preference training

- dataset: current `HuggingFaceH4/ultrafeedback_binarized`
- train pairs: 10,000
- held-out pairs: 1,000
- DPO beta: 0.1
- LR: 1e-5
- micro 16 x grad-accum 4
- frozen SFT reference model

## Budget safety

The user reported $30 Modal credit before launch.

Hard code-level controls:

- CPU/data-prep function: 8 physical cores, 32 GiB RAM, max 4 hours, no GPU.
- Combined H100 pretrain + SFT + DPO function: exactly `H100!`, 8 cores, 64 GiB RAM, max 11,700 seconds.
- The 11,700-second GPU ceiling is constant in code and cannot be increased through an issue body or workflow-dispatch argument.
- No automatic retry is configured.
- If fewer than 1,500 seconds remain after pretraining, post-training is refused and the completed base checkpoint is preserved.
- SFT is committed before DPO; if fewer than 450 seconds remain after SFT, DPO is refused and the SFT checkpoint is preserved.

At the Modal H100 list rate verified on 2026-08-22 ($0.001097/GPU-second), the maximum H100 charge from the production function is $12.83. Even including the configured CPU/RAM ceilings and the separate CPU-only data-prep ceiling, the code-level maximum remains materially below the reported $30 credit.

## Checkpoint/evaluation policy

- pretraining eval every 200M tokens
- pretraining checkpoint every 200M tokens
- final base eval after 2B tokens
- SFT checkpoint committed before DPO
- DPO checkpoint committed before optional final generation/eval
- final checkpoint path: `/vol/full100m-runs/TAM-v3-100M-2B-seed8100/final_dpo.pt`
- final pipeline summary: `/vol/full100m-runs/TAM-v3-100M-2B-seed8100/pipeline_summary.json`

## Interpretation rule

Do not infer architecture superiority from this model alone. This run establishes whether a seriously pretrained/post-trained 100M TAM develops useful capability. A parameter/data/post-training-matched Transformer should subsequently be trained for the clean architecture comparison if budget permits.

The earlier 25M/50M loss wins remain architecture-screen evidence; this 2B-token run is the first capability-oriented TAM checkpoint.
