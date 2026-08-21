# TAM Research

Reproducible experiments comparing a standard causal Transformer against TAM architectures under matched data, parameter scale, optimizer, token budget, and seeds.

## Current question

**Does TAM v2 retain its early-learning advantage over a parameter-matched Transformer when trained on a real corpus at meaningful token counts?**

No breakthrough claim is made unless an advantage survives larger token budgets, multiple seeds, general language evaluation, downstream tasks, and compute/throughput accounting.

## Architectures

TAM v2 routes every token/layer across three causal pathways: self-attention, ATAM temporal associative memory, and a recurrent world-state. The world-state uses a differentiable O(log T) parallel affine scan rather than the CPU screening run's Python token loop. The router starts exactly neutral and must learn any specialization.

25M parameter counts:

- Transformer: 24,940,288
- TAM v1: 24,971,143
- TAM v2: 24,986,548

## Serious 25M gate

- Corpus: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`
- Tokenizer: GPT-2 BPE, 50,257 vocabulary
- Default context: 512
- Precision: bf16
- Optimizer: fused AdamW, betas (0.9, 0.95), cosine schedule
- Default comparison: Transformer vs TAM v2, three seeds each
- First gate: 100M tokens/model
- Validation every 5M tokens
- Checkpoints every 10M tokens
- Same pre-tokenized binary stream and deterministic batch offsets for matched seeds
- Persistent Modal Volume: `tam-research-data`

If the 100M-token signal survives, resume the same checkpoints to 250M and then 500M before increasing model size.

## One-time connection setup

The repository contains **no credentials**. For GitHub to launch Modal, add these GitHub repository Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Your existing Modal Secret named `github-secret` must contain `GITHUB_TOKEN`; the Modal training jobs use it only to post run results back to the GitHub run-request issue.

After the workflow is merged, there are two launch paths:

1. GitHub Actions → **Modal TAM research** → **Run workflow**.
2. Create an owner-authored GitHub issue whose title starts with `[modal-run]` and whose body is JSON, for example:

```json
{
  "token_budget": 100000000,
  "architectures": "transformer,tamv2",
  "seeds": "7025,7026,7027",
  "seq_len": 512
}
```

The issue-trigger path is intentionally restricted to issues authored by the repository owner and validates architectures, seeds, token budget, and sequence length before invoking Modal.

## Direct CLI

```bash
pip install modal
modal run --detach modal_app.py \
  --action suite \
  --token-budget 100000000 \
  --architectures transformer,tamv2 \
  --seeds 7025,7026,7027 \
  --seq-len 512
```

## Local verification

```bash
pip install -e '.[test]'
pytest -q
```

The current unit tests verify causal behavior, exact parallel-scan equivalence, forward shapes, and <1% parameter mismatch.

## Security

Do not commit Modal or GitHub credentials. `.modal.toml`, `.env`, model checkpoints, and tokenized datasets are ignored.

**Important:** this repository was public when this stack was created. If you want the research code and run results private, change the repository visibility to **Private** before launching experiments.
