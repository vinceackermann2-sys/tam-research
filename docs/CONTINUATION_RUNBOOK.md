# TAM Continuation Runbook

Use this file when resuming the research from a new machine, ChatGPT session, or researcher.

## 1. Canonical sources of truth

Read in this order:

1. `docs/RESEARCH_HANDOFF_2026-08-21.md`
2. `docs/breakthrough_protocol.md`
3. `docs/TAM_V3_ARCHITECTURE.md`
4. `docs/EXPERIMENT_RESULTS.csv`
5. `tam_research/models.py`
6. `tam_research/train.py`
7. `tam_research/train_scaled.py`
8. `modal_scale_app.py`
9. `.github/workflows/modal-scaling.yml`

GitHub issue comments are the authoritative callback log for actual Modal runs.

## 2. Current run to inspect first

At the time this runbook was created:

- preregistration: issue #66
- active 100M run: issue #67

Issue #67 configuration:

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

Do not launch a duplicate run unless #67 is conclusively failed and the failure cannot leave detached GPU calls alive.

## 3. Budget rule

At this handoff the user reported roughly $3.5 Modal credit remaining. Issue #67 was launched as the final intended compute spend under that budget.

Before any additional compute:

1. inspect #67
2. inspect remaining Modal balance manually if available
3. do not launch 300M/1B with the old credit assumption

## 4. How the scaling launcher works

Opening an issue whose title starts with `[modal-scale]` triggers `.github/workflows/modal-scaling.yml`.

The issue body is JSON. The workflow validates:

- allowed scale
- architecture set
- numeric seeds
- token budget bounds
- allowed microbatch/accumulation
- global batch invariant `micro * accum == 128`

It then runs Modal detached. Remote callbacks post to the issue.

Do not put secrets in issue bodies or repository files.

Required GitHub Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Required Modal secret:

- `github-secret` containing `GITHUB_TOKEN`

Never commit or print secret values.

## 5. Checkpoints and resume

Modal Volume: `tam-research-data`

Important paths:

```text
/vol/data/fineweb-edu-gpt2
/vol/scaling-runs/<scale>/ctx<context>-mb<micro>-ga<accum>/...
/vol/compile-cache/...
```

`train.py` checkpoints include:

- model state_dict
- optimizer state_dict
- step
- tokens_seen
- CPU batch generator state
- compile metadata

Run roots are isolated by scale/context/microbatch/accumulation to avoid accidental resume collisions.

## 6. Compiler cache

Purpose: avoid paying hundreds of H100-seconds for the same TorchInductor graph on every random seed.

Current cache identity intentionally excludes random seed and includes graph/runtime identity. See `tam_research/compile_cache.py`.

The launcher warms the first seed for each architecture synchronously, commits the Volume, then spawns remaining seeds.

Historical cache bugs already fixed:

- status reporting killed launcher in GitHub runner → status is now non-fatal
- `volume.reload()` after Torch opened cache files → reload moved before compiler/cache initialization
- cache namespace insufficiently versioned → schema/runtime/compiler isolation added

If a new cache bug appears, preserve the failed issue and do not delete evidence.

## 7. How to score issue #67

Wait until all six final callbacks exist or a failure is conclusive.

For each matched seed compute:

```text
delta_i = Transformer_NLL_i - TAM_NLL_i
```

Then report:

- all six NLLs
- mean NLL per architecture
- 3 paired deltas
- mean delta
- standard deviation of paired deltas
- Transformer mean training tok/s
- TAM mean training tok/s
- throughput ratio TAM / Transformer
- mean peak VRAM
- cold/warm compile seconds separately

Pass rule from issue #66:

- TAM lower mean NLL
- TAM wins >=2/3 seeds
- finite/stable runs
- TAM throughput >=80% Transformer mean, otherwise optimize before scaling onward

Do not change this rule after seeing results.

## 8. If 100M passes

Do NOT immediately claim breakthrough.

Next actions:

1. Update `docs/EXPERIMENT_RESULTS.csv` and handoff/evidence ledger with the 100M results.
2. Fit the first 25M/50M/100M equal-token trend.
3. Estimate uncertainty and whether the TAM advantage is shrinking, flat, or widening.
4. Verify unresolved residual/amplitude controls.
5. Add at least one second corpus/data mixture before broad claims.
6. Add modern recurrent/hybrid baselines.
7. Run long-context/state tests.
8. Only then promote the staged ~300M configuration.

The 300M/1B family should be reconstructed/merged on top of current main rather than reviving stale PR #57 blindly.

## 9. If 100M fails

Keep every result.

Diagnose which failure class occurred:

- lower quality at equal tokens
- throughput below gate
- numerical instability
- seed inconsistency
- cache/launcher/runtime failure

Do not confuse an infrastructure failure with an architecture failure. Conversely, do not rerun an unfavorable scientific result merely because it is unfavorable.

## 10. Agent path

The base LM scaling track and agent track are separate.

If TAM survives 100M → 300M → ~1B, begin agent post-training with:

- instruction/SFT data
- structured function/tool calls
- terminal/file/browser/code tools
- multi-step observation/action trajectories
- outcome/reward training
- persistent external memory
- recovery from failed actions
- agent task benchmarks

First useful narrow-agent target: roughly 1–3B after adequate pretraining and dedicated agent post-training, not a raw 100M model.

## 11. Breakthrough rule

Never infer a breakthrough from perplexity alone. The frozen criteria require multi-scale reproducibility, compute competitiveness, mechanism controls, downstream capability/state improvements, modern baselines, and a second data mixture.
