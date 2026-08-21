# TAM Research

Reproducible research comparing parameter-matched causal Transformers against TAM architectures under matched data, seeds, optimizer, hardware, token budgets, and explicit compute accounting.

## Current primary architecture

**TAM v3 = reduced-width causal attention + learned recurrent world-state.**

TAM v3 is the successor to ATAM/TAM v1/TAM v2 after mechanism ablations showed that removing the ATAM branch improved general language loss on three matched seeds. The recurrent world-state is a causal diagonal affine recurrence evaluated with a parallel associative scan.

No broad breakthrough claim is made yet.

## Start here

For a complete recovery/continuation package, read:

- [`docs/RESEARCH_HANDOFF_2026-08-21.md`](docs/RESEARCH_HANDOFF_2026-08-21.md) — full research state, chronology, results, failures, storage, next steps
- [`docs/TAM_V3_ARCHITECTURE.md`](docs/TAM_V3_ARCHITECTURE.md) — architecture and design goals
- [`docs/EXPERIMENT_RESULTS.csv`](docs/EXPERIMENT_RESULTS.csv) — machine-readable results ledger
- [`docs/CONTINUATION_RUNBOOK.md`](docs/CONTINUATION_RUNBOOK.md) — exact recovery/next-run procedure
- [`docs/breakthrough_protocol.md`](docs/breakthrough_protocol.md) — frozen scientific pass/fail criteria

GitHub issue comments are the authoritative callback log for actual Modal/H100 runs.

## Current evidence

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
- active training: issue **#67**
- 10M tokens/model
- Transformer + TAM v3
- seeds 7202/7203/7204
- strict `H100!`

Do not launch a duplicate while #67 may have detached GPU calls alive.

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
/vol/scaling-runs/...            checkpoints, metrics, summaries
/vol/compile-cache/...           persistent Inductor/Triton compiler artifacts
```

GitHub contains the code, tests, launchers, protocols, run metadata, and documented evidence. Large checkpoint binaries remain on the Modal Volume.

## Launching scaling runs

Owner-authored issues whose title starts with `[modal-scale]` trigger `.github/workflows/modal-scaling.yml`.

Example body:

```json
{
  "model_scale": "100m",
  "token_budget": 10000000,
  "architectures": "transformer,tamv3",
  "seeds": "7202,7203,7204",
  "micro_batch_size": 64,
  "grad_accum_steps": 2
}
```

Do not include secrets in issues.

GitHub Actions requires repository secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Modal requires secret `github-secret` containing `GITHUB_TOKEN` for remote result callbacks.

## Verification

```bash
pip install -e '.[test]'
pytest -q
```

Tests cover architecture causality/shape behavior, affine-scan correctness, parameter matching, scaling configurations, and compiler-cache isolation/reuse logic.

## Scientific rule

Equal-token loss is only one claim. Before calling TAM a breakthrough, the frozen protocol requires multi-scale reproducibility, competitive equal-compute behavior, mechanism/residual controls, long-context/state evidence, downstream capability gains, modern recurrent/hybrid baselines, and a second data mixture.

## Security

Never commit GitHub/Modal credentials, `.env` files, tokenized corpora, or model checkpoints directly to ordinary Git history.

If this repository is public and the research should be confidential, change the repository visibility to **Private** in GitHub settings. The current connector cannot change repository visibility automatically.
