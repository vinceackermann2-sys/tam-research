# Recurrent Looped Transformer (RLT) isolated experiment lane

This directory is an isolated, non-historical experiment lane for testing the
public architecture described by `yifanzhang-pro/recurrent-looped-tranformer`.
The upstream repository currently publishes a technical report/project page, not
an executable training implementation, so this is a clean-room **public-spec
prototype**, not a claim of bit-for-bit reproduction of unpublished code.

Nothing in `tam_research/`, historical result directories, existing workflow
files, or the default branch is modified by this lane.

## Public architecture implemented here

The current upstream description defines a causal encoder plus a token-recurrent
decoder. The complete decoder state is the previous final hidden state plus
layerwise decoder sliding-window-attention (SWA) KV caches:

```text
encoder: known tokens -> causal encoder representations + global prefix memory
H_t = (s_t, C_t^D)
H_t = D_phi(Merge(e_t, s_(t-1)); M_(<=t), C_(t-1)^D, t)
logits_t = W_o RMSNorm(s_t)
```

This prototype implements:

- causal encoder processing over the known token sequence;
- token-by-token recurrent decoding through every prompt token;
- previous final decoder output carried into the next token;
- layerwise decoder SWA KV caches with a hard window bound;
- decoder cross-attention restricted to encoder memory through the current token;
- compatible self-attention and FFN weights shared between corresponding encoder
  and decoder stages; decoder cross-attention remains decoder-specific;
- no detach of recurrent state, decoder KV, or encoder memory during training, so
  the reference path is differentiable for full BPTT;
- RMSNorm output and a tied token/output embedding.

The public project page leaves `Merge(e_t, s_(t-1))` abstract. This prototype
makes that choice explicit: concatenate the two vectors and apply a learned linear
projection followed by RMSNorm. That is an experimental implementation choice,
not an upstream claim.

The upstream concrete configuration describes 48 encoder stages + 48 decoder
stages. The code supports `n_stages=48`, while the first Colab training lane is
intentionally much smaller so architectural correctness is established before
spending meaningful GPU time.

## Local preflight

```bash
python -m experiments.rlt.smoke
pytest -q tests/test_rlt_experiment.py
```

The smoke test checks:

- future tokens cannot affect earlier logits;
- decoder SWA caches never exceed the configured window;
- a late-token loss backpropagates through recurrent state;
- a deterministic next-token toy task can be optimized;
- a 48-stage tiny-width depth probe executes with the expected output shape.

## Real-data training

`experiments/rlt/train_rlt.py` reuses the repository's existing FineWeb-Edu /
GPT-2 token data path, deterministic `TokenBin` sampler, seed utility, and cosine
learning-rate schedule. It writes checkpoints and summaries only under its
explicit `run_root`.

Initial profiles are:

- `tiny`: 128 width, 4 heads, 2 encoder/decoder stages, SWA 32;
- `small`: 192 width, 6 heads, 4 encoder/decoder stages, SWA 64;
- `paper-depth-probe`: 128 width, 4 heads, 48 encoder/decoder stages, SWA 64.

The Colab bridge initially permits only the bounded `tiny` profile. Expanding the
training envelope should happen only after the smoke and first real-data run are
recorded successfully.

## Colab bridge

Open `colab/RLT_Colab_Bridge.ipynb` in Google Colab, choose a GPU runtime, add a
Colab secret named `GITHUB_TOKEN`, and run the cells. The token should be a
fine-grained GitHub token scoped only to `vinceackermann2-sys/tam-research` with
repository **Contents: Read and write** permission.

The notebook watches only:

- jobs: `experiments/rlt/bridge/jobs/*.json`
- terminal results: `experiments/rlt/bridge/results/*.json`

The bridge does **not** execute arbitrary commands from queue files. Schema 1
allows only `smoke` and bounded `train` jobs. Initial training is limited to the
`tiny` profile, at most 500,000 training tokens, sequence length at most 128,
micro-batch at most 4, and gradient accumulation at most 16.

Training data and checkpoints live outside the repository by default:
`/content/rlt-data` and `/content/rlt-runs`. This prevents collisions with
existing research outputs. Set `RLT_DATA_DIR` / `RLT_RUN_ROOT` in Colab if you
want a mounted Drive location instead.

Result JSON files are create-once terminal records. A job whose result already
exists is never rerun by the worker. Queue job IDs should therefore never be
reused.
