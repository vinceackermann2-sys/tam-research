# TAM v3 100M / 2B Full-Training Protocol

Status: preregistered before the production run is launched; amended pre-H100 on 2026-08-22 only to resolve an empirically impossible source quota described below.

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
| Cosmopedia OpenStax | 99M | 4.95% | synthetic textbook-style science/education |
| Cosmopedia Stanford | 201M | 10.05% | synthetic higher-education material |
| Common-Pile open-license ArXiv papers | 150M | 7.5% | real scientific text incl. physics/quantitative biology/CS/math |

The dataset builder records exact source metadata and prepares each source as a resumable token shard before deterministic weighted-fair interleaving.

### Pre-H100 source-capacity amendment — 2026-08-22

The original preregistration assigned 150M training tokens to Cosmopedia OpenStax and 150M to Cosmopedia Stanford. During CPU-only preparation, before any H100 training was started, the complete `HuggingFaceTB/cosmopedia` `openstax` text stream ended at exactly 375,000/375,000 validation tokens and 99,048,429/150,000,000 requested GPT-2 training tokens. Therefore the original 150M text-only OpenStax quota was empirically impossible without duplicating examples or changing the text definition.

The amendment uses 99M unique OpenStax training tokens and reallocates the remaining 51M tokens to the much larger Cosmopedia Stanford config, yielding 201M Stanford tokens. This preserves:

- exactly 2.0B total training tokens;
- exactly 300M total Cosmopedia OpenStax+Stanford training tokens;
- the same HuggingFaceTB Cosmopedia dataset family and Apache-2.0 provenance note;
- the same 5M validation budget and per-source validation allocations;
- exact 1M-token train interleave divisibility;
- all model, tokenizer, seed, optimizer, batch, checkpoint, post-training, and H100-budget settings.

The corpus assembly version is bumped from 2 to 3 so an older final mixture cannot be mistaken for the amended production corpus. Completed source shards whose source metadata is unchanged remain reusable; OpenStax and Stanford are validated/rebuilt against the amended quotas.

This amendment is recorded before H100 allocation and is not evidence for or against TAM. It exists solely because the frozen source quota exceeded the available text stream.

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

- CPU/data-prep function: 8 physical cores, 32 GiB RAM, non-preemptible, max 4 hours, no GPU.
- Combined H100 pretrain + SFT + DPO function: exactly `H100!`, 8 cores, 64 GiB RAM, max 11,700 seconds.
- The 11,700-second GPU ceiling is constant in code and cannot be increased through an issue body or workflow-dispatch argument.
- No automatic retry is configured.
- If fewer than 1,500 seconds remain after pretraining, post-training is refused and the completed base checkpoint is preserved.
- SFT is committed before DPO; if fewer than 450 seconds remain after SFT, DPO is refused and the SFT checkpoint is preserved.

At the Modal H100 list rate verified on 2026-08-22 ($0.001097/GPU-second), the maximum H100 charge from the production function is $12.83. Non-preemptible CPU/RAM is billed at a premium, but the configured CPU-only four-hour ceiling plus the fixed H100 ceiling remains within the reported $30 credit based on the rates checked before launch.

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