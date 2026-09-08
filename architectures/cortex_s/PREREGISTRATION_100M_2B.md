# CORTEX-S v0 — 100M / 2B paired historical-control preregistration

Status: **frozen before paid execution**.

This protocol supersedes the unexecuted 25M paid seed-1 plan only for the immediate scale test requested on 2026-09-08. No reserved fresh CORTEX-S scientific seed has been consumed.

## Classification

This is an **adaptive paired historical-control experiment**, not a blind fresh-seed trial.

The repository's matched 100M Transformer was completed earlier and its outcome was available before CORTEX-S was scaled. Therefore even a strong CORTEX-S result cannot by itself be called a breakthrough. The purpose of this run is to determine whether the architecture is worth an independent replication.

## Frozen comparison

Historical control: repository issue `#140`, Transformer 100M, seed `8100`.

CORTEX-S paired run:

- seed: `8100` so the random-window stream is paired with the historical control;
- training tokens: `2,000,000,000`;
- validation tokens: `5,000,000`;
- context: `512`;
- microbatch: `64`;
- gradient accumulation: `2`;
- global batch: `128`;
- optimizer: AdamW, betas `(0.9, 0.95)`;
- peak LR: `3e-4`;
- weight decay: `0.1`;
- warmup fraction: `0.02`;
- precision: CUDA bf16 autocast;
- same immutable pretraining directory and exact corpus fingerprints as the Transformer;
- same final validation seed/batch convention as the repository trainer.

No second Transformer will be trained. Reusing the already completed control is both scientifically cleaner and materially cheaper.

## Frozen 100M CORTEX-S architecture

- vocabulary: `50,257`;
- width: `512`;
- layers: `24`;
- heads: `16`;
- recurrent state: `128` per layer;
- experts: `8`;
- top-k: `2`;
- expert hidden width: `338`;
- full attention every `6` layers, therefore `4/24` attention layers;
- total trainable parameters: `101,778,112`;
- historical Transformer parameters: `101,803,520`;
- absolute mismatch: `25,408` parameters, about `0.025%`;
- theoretical selected expert-token assignments: `25%` of dense expert assignments.

The models are matched on **total trainable parameters**, not active FLOPs. Any speed/efficiency advantage must be measured; it is not assumed from sparsity.

## State policy limitation

Training uses the repository's random 512-token windows. Recurrent state is reset at each independently sampled training sequence. This run tests the CORTEX-S architecture inside a normal language-model training regime; it does **not** train or establish continual learning across batches and cannot support a continual-learning claim.

## Zero-credit gates before H100

Before any paid function is permitted:

1. instantiate the real 100M CORTEX-S on CPU and measure its actual parameter count;
2. require parameter mismatch <= `0.5%`;
3. run a real 100M forward pass on CPU and require finite logits;
4. run same-mechanism forward/backward on a reduced CPU model and require finite gradients;
5. syntax-check the paid launcher and trainer;
6. require normal repository CI and the dedicated CORTEX-S 100M/2B CPU workflow to pass.

## Zero-GPU corpus gate

The Modal app must first use CPU only to check the existing corpus metadata, byte lengths, and full SHA-256 fingerprints. A corpus mismatch stops before H100 allocation.

## H100 calibration gate

Calibration seed `910001` is engineering-only and becomes consumed by the first H100 preflight. It is not scientific evidence.

The preflight uses the exact production 100M graph, context, microbatch, gradient accumulation and optimizer. It may try the historical Transformer's compile mode; if compilation fails it may make one eager fallback measurement. It may never start the 2B run itself.

Full progression requires all of:

- finite loss and gradients;
- exact expected parameter count;
- positive measured throughput;
- peak H100 allocation <= `70 GiB`;
- conservative projected full compute <= `8,500 seconds`;
- exact source SHA unchanged between preflight and full launch.

## Credit guard

Pricing snapshot recorded in the protocol code:

- H100: `$0.001097/s`;
- conservative Starter CPU assumption: `$0.00003942/core/s`;
- conservative Starter memory assumption: `$0.00000667/GiB/s`.

The full function has a separate hard timeout of `10,000 seconds`. The code validates that even this conservative hard-timeout estimate stays below the stated `$29` credit envelope. Live billing can differ, so the primary fail-safe is a wall-clock ceiling rather than a dollar estimate.

The H100 preflight and full namespaces are single-use: consumed markers prevent accidental duplicate launches.

## Primary result

Primary quality comparison is held-out language NLL after exactly 2B token exposures. Historical Transformer reference:

- NLL: `2.7115590302149455`;
- perplexity: `15.053722884190283`;
- training throughput: `321151.5755348581 tok/s`;
- training seconds: `6227.609573988244`;
- total compute seconds including compile: `6478.327432424761`;
- peak VRAM: about `11.896 GiB`.

Report CORTEX-S final NLL/PPL, training throughput, total compute time, peak VRAM, actual router utilization and parameter mismatch.

## Interpretation gate

A CORTEX-S result is interesting enough for independent replication if it is finite and its 2B-token held-out NLL is strictly below the historical Transformer's `2.711559...` while the measured systems cost remains practical.

Even then, label it **breakthrough candidate evidence at most** only after later fresh-seed work. This paired adaptive run itself remains `PAIRED_HISTORICAL_CONTROL_ADAPTIVE_EXPERIMENT` and has `breakthrough_claim_allowed = false` in machine-readable results.

If it loses on 2B-token NLL or the preflight fails the budget/systems gate, stop paid progression. Do not spend reserved fresh seeds just to rescue the result.

## Fresh replication remains untouched

Reserved seeds `48131`, `48132`, `48133` remain unused. They may only be used after this development comparison justifies a separately frozen independent protocol, including a modern recurrent/state-space baseline and additional data/domain checks.
