# Recurrent Looped Transformer (RLT) isolated experiment lane

This directory is an isolated, non-historical experiment lane for testing the
Recurrent Looped Transformer idea described by
`yifanzhang-pro/recurrent-looped-tranformer`.

## Architecture

The implementation keeps the existing TAM research Transformer primitives and
data protocol, but changes depth parameterization:

```text
h0 = token_embedding(tokens) + position_embedding(positions)
h(t+1) = F_theta(h(t))          # same Transformer block at every loop
logits = LM_head(LN(hT))
```

`ModelConfig.n_layers` is interpreted as **effective recurrent depth** for RLT.
Only one physical `Block` is instantiated, so increasing loop count does not add
block parameters.

Nothing in `tam_research/`, historical result directories, existing workflow
files, or the default branch is modified by this lane.

## Local CPU preflight

```bash
python -m experiments.rlt.smoke
pytest -q tests/test_rlt_experiment.py
```

The smoke test checks:
- parameter count is invariant to recurrent loop count,
- causal masking survives recurrence,
- gradients reach the shared block,
- a tiny deterministic sequence task learns.

## Colab bridge

Open `colab/RLT_Colab_Bridge.ipynb` in Google Colab, choose a GPU runtime, add a
Colab secret named `GITHUB_TOKEN`, and run the cells. The token should be a
fine-grained GitHub token scoped only to `vinceackermann2-sys/tam-research` with
repository **Contents: Read and write** permission.

The notebook watches only:

- jobs: `experiments/rlt/bridge/jobs/*.json`
- terminal results: `experiments/rlt/bridge/results/*.json`

The bridge does **not** execute arbitrary commands from queue files. Schema 1
allows only `smoke` and bounded `compare` jobs. It currently caps comparisons at
the 25m width/depth shape and at 5,000,000 training tokens.

Training data and checkpoints live outside the repository by default:
`/content/rlt-data` and `/content/rlt-runs`. This prevents collisions with
existing research outputs. Set `RLT_DATA_DIR` / `RLT_RUN_ROOT` in Colab if you
want a mounted Drive location instead.

Result JSON files are create-once terminal records. A job whose result already
exists is never rerun by the worker.

## Comparison semantics

`compare` trains:
1. RLT: same width and effective depth as the selected Transformer scale, one
   shared physical block;
2. Transformer: the repository's ordinary Transformer at that same shape.

Both get the same FineWeb-Edu token stream, GPT-2 tokenizer, seed, token budget,
sequence length, optimizer hyperparameters, and sampled training offsets.

This is a **same-width / same-effective-depth** comparison, not a
parameter-matched comparison. RLT is intentionally much smaller in parameter
count because it shares block weights.
