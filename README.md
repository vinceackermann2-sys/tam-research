# TAM Research

Reproducible research comparing parameter-matched causal Transformers against TAM architectures under matched data, seeds, optimizer, hardware, token budgets, and explicit compute accounting.

## Current primary architecture

**TAM v3 = reduced-width causal attention + learned recurrent world-state.**

TAM v3 is the successor to ATAM/TAM v1/TAM v2 after mechanism ablations showed that removing the ATAM branch improved general language loss on three matched seeds. The recurrent world-state is a causal diagonal affine recurrence evaluated with a parallel associative scan.

No broad breakthrough claim is made yet.

## Start here

For a complete recovery/continuation package, read:

- [`docs/RESEARCH_HANDOFF_2026-08-21.md`](docs/RESEARCH_HANDOFF_2026-08-21.md) — full research state, chronology, results, failures, storage, next steps
- [`docs/TAM_V3_25M_FULL_RESULT_2026-08-21.md`](docs/TAM_V3_25M_FULL_RESULT_2026-08-21.md) — completed 300M-token base → SFT → DPO lineage and final checkpoint
- [`docs/TAM_V3_ARCHITECTURE.md`](docs/TAM_V3_ARCHITECTURE.md) — architecture and design goals
- [`docs/EXPERIMENT_RESULTS.csv`](docs/EXPERIMENT_RESULTS.csv) — machine-readable results ledger
- [`docs/CONTINUATION_RUNBOOK.md`](docs/CONTINUATION_RUNBOOK.md) — exact recovery/next-run procedure
- [`docs/breakthrough_protocol.md`](docs/breakthrough_protocol.md) — frozen scientific pass/fail criteria

GitHub issue comments are the authoritative callback log for actual Modal/H100 runs.

## Completed end-to-end TAM model

The first fully pretrained + post-trained TAM v3 lineage is complete at ~25M parameters (seed 7400):

- base pretraining: **300M FineWeb-Edu tokens**
- base held-out NLL: **4.4131** (PPL **82.52**)
- SFT held-out assistant NLL: **3.6049**
- DPO held-out implicit reward accuracy: **0.564**
- final post-SFT+DPO FineWeb NLL: **4.6162**
- final checkpoint: `/vol/full-model-runs/TAM-v3-25M-Full-seed7400/final_dpo.pt`

Persistent-Volume verification in issue #92 confirms the base, SFT, DPO and final summary artifacts are all present. This validates the complete TAM lifecycle, not production-quality assistant capability; a 25M model remains intentionally small.

## Current architecture evidence

### ~25M parameters

TAM v3 beat the parameter-matched Transformer on held-out language loss in 3/3 fresh 10M-token seeds. After H100 utilization optimization, TAM maintained roughly 80%+ of Transformer steady-state throughput.

### ~50M parameters

Three matched 10M-token seeds:

| Seed | Transformer NLL | TAM v3 NLL |
|---:|---:|---:|
| 7102 | 7.0969 | **6.8978** |
| 7103 | 7.0550 | **6.8577** |
| 7104 | 7.0404 | **6.9101** |

Mean Transformer NLL: **7.0641**  
Mean TAM v3 NLL: **6.8885**  
TAM won **3/3** seeds, with ~81.8% of Transformer steady training throughput.

### ~100M parameters

Both matched ~101.8M models fit and train stably on one H100 at context 512 / micro 64 / accumulation 2.

The preregistered replicated 100M gate is:

- preregistration: issue **#66**
- run: issue **#67**
- 10M tokens/model
- Transformer + TAM v3
- seeds 7202/7203/7204
- strict `H100!`

Confirmed so far from issue #67: Transformer seed 7202 completed at NLL **6.9840**, 309,905 tok/s, 36.99 GiB peak VRAM. Do not infer the 100M architecture result until all TAM/replica callbacks exist.

## Matched parameter family

| Scale | Transformer | TAM v3 |
|---|---:|---:|
| 25M | 24,940,288 | 24,941,263 |
| 50M | 49,799,808 | 49,801,457 |
| 100M | 101,803,520 | 101,806,616 |

TAM scaling ratios are self-similar:

```text
state_size = d_model / 4
TAM attention inner = 13*d_model / 16
```

## Serious training protocol

- Corpus: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`
- Tokenizer: GPT-2 BPE, vocab 50,257
- Current main context: 512
- Precision: bfloat16
- Optimizer: fused AdamW, betas (0.9, 0.95), weight decay 0.1
- LR: 3e-4 peak, cosine schedule, 2% warmup
- Gradient clipping: 1.0
- Global optimizer batch: 128 sequences
- Current preferred batching: micro 64 × accumulation 2 when stable
- GPU benchmark class: exact `H100!`
- Same token binaries, matched seeds, deterministic batch streams

## Modal storage

Persistent Modal Volume: `tam-research-data`

```text
/vol/data/fineweb-edu-gpt2       tokenized train/validation data
/vol/scaling-runs/...            scaling checkpoints, metrics, summaries
/vol/full-model-runs/...         completed full-model base/SFT/DPO checkpoints
/vol/compile-cache/...           persistent Inductor/Triton compiler artifacts
```

GitHub contains the code, tests, launchers, protocols, run metadata, failure history, and documented evidence. Large checkpoint binaries remain on the Modal Volume.

## Verification

```bash
pip install -e '.[test]'
pytest -q
```

CI additionally syntax-checks the root `modal_*_app.py` launchers before GPU changes are merged.

## Scientific rule

Equal-token loss is only one claim. Before calling TAM a breakthrough, the frozen protocol requires multi-scale reproducibility, competitive equal-compute behavior, mechanism/residual controls, long-context/state evidence, downstream capability gains, modern recurrent/hybrid baselines, and a second data mixture.

The completed 25M full model proves the training/post-training pipeline works; it does **not** prove TAM is a superior chat/agent architecture.

## Security

Never commit GitHub/Modal credentials, `.env` files, tokenized corpora, or model checkpoints directly to ordinary Git history.
