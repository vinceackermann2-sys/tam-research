# Physics Tokenizer Phase 0 — raw wave fields → learned tokens → dynamics

Issue: #950  
Engineering seed: `74193`  
Status: **FAIL (primary forecast gate)**

## Question

Can a model learn discrete non-text tokens directly from scientific measurements and use those tokens as the native state for a predictive world model?

This Phase-0 experiment does **not** serialize physics into English, SMILES-like strings, or symbolic equations. Its input is a numeric tensor representing a physical field:

- channel 0: displacement field `u(x,y,t)`
- channel 1: velocity field `du/dt`

The synthetic system is a periodic 2-D wave equation. The purpose is to validate the representation/training path cheaply before considering expensive compute or real detector/simulation data.

## Pipeline

`raw float fields -> convolutional encoder -> 256-entry learned K-means/VQ codebook -> discrete 16x16 token grid -> spatial token dynamics model -> decoded future field`

The tokenizer and predictor use PyTorch + NumPy only (the repo's core dependencies). The predictor is intentionally a small 2-D convolutional token model for this first field-dynamics test; this is not an LLM and no text is used.

## Preregistered setup

- Grid: `16x16`
- Sequence length: `8`
- Integration step: `dt=0.12`
- Train/ID wave speed: `c in [0.6, 1.0]`
- OOD wave speed: `c in [1.15, 1.35]`
- Train sequences: `128`
- ID validation sequences: `48`
- OOD validation sequences: `48`
- Codebook: `256`
- Context: `3` token grids
- Rollout horizon: `3`
- Engineering seed: `74193`

### Gates

1. Codebook usage: >= 25% active codes and perplexity >= 4.
2. Tokenizer reconstruction: normalized MSE <= 0.20.
3. Primary predictive gate: decoded 3-step ID rollout MSE must be lower than a persistence baseline that simply repeats the last observed field.

No new-physics, AGI, or scientific-discovery claim is authorized by this engineering experiment.

## Canonical CPU result

Command:

```bash
python -m tam_research.physics_tokenizer_phase0 \
  --device cpu \
  --seed 74193 \
  --out physics_tokenizer_phase0_seed74193.json
```

The run was repeated with the same code/seed and reproduced the metrics below.

| Metric | Result | Gate |
|---|---:|---|
| Active codes | 256 / 256 | PASS |
| Codebook perplexity | 198.6584 | PASS |
| Normalized reconstruction MSE | 0.060512 | PASS |
| Physical reconstruction MSE | 0.019935 | diagnostic |
| ID 1-step token accuracy | 69.995% | diagnostic |
| ID 3-step decoded MSE | 0.028167 | — |
| ID 3-step persistence MSE | 0.013432 | — |
| ID forecast / persistence | 2.09698x | **FAIL** |
| OOD 3-step decoded MSE | 0.054499 | diagnostic |
| OOD 3-step persistence MSE | 0.052762 | diagnostic |
| OOD forecast / persistence | 1.03291x | diagnostic |
| Phase-0 gate | false | **FAIL** |

The exact machine-readable result is checked in as `docs/PHYSICS_TOKENIZER_PHASE0_SEED74193.json`.

## Interpretation

The representation part is viable: the codebook is not collapsed and the decoder can reconstruct held-out raw fields well enough to pass the preregistered tokenizer gate. The current discrete-token dynamics path, however, loses enough state precision that a trivial short-horizon persistence predictor remains stronger on ID data.

That is a substantive negative result. Scaling the same architecture on paid GPU compute would not answer the important question; it would mostly spend more compute behind a failed baseline.

## Phase 0b hypothesis

The next controlled change should target **information loss**, while preserving the same data split, seed policy, and persistence gate. Candidates, in order:

1. residual/product VQ (multiple codes per spatial location),
2. hybrid state = discrete semantic token + continuous residual,
3. delta prediction in latent/field space rather than absolute-token classification,
4. only after the above, increase dynamics-model capacity.

A Phase 0b run should be separately preregistered with a fresh engineering seed and must still beat the same persistence baseline before any expensive scaling or scientific interpretation.
