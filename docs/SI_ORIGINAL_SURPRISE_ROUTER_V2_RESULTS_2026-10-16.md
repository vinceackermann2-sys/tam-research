# SI Original — Causal Surprise Router v2: CPU results (2026-10-16)

**Status: FAILED primary dual-endpoint gate.** We have neither an SI breakthrough nor grounds to spend the user's reported ~$25 Modal compute. This is a separate *mechanism test* belonging to the original screenshot-inspired SI hybrid program. It does **not** complete or supersede the actual recurrent/MoE/memory/continual-learning/multimodal architecture.

## Immutable source / execution provenance

- Preregistration: `docs/SI_ORIGINAL_SURPRISE_ROUTER_V2_PROTOCOL_2026-10-16.md` committed before held-out experiments.
- Implementation: `scripts/si_original_surprise_router_v2.py`; frozen GitHub blob **`1797500667ea7bdda98b558f7f5f2ff00d90bea6`**, byte-identical to locally executed file (local git hash-object also `1797500667ea7bdda98b558f7f5f2ff00d90bea6`; local SHA256 `188421d89c3d38edda443bf6b2e98d2d9550ee843392d8d86373ec59bdf28561`).
- Initial smoke/development seed **96300** and causality check, including diagnostics on training-generator conditions, were excluded from confirmation.
- **Fresh frozen CPU seeds 96311, 96312, 96313 all completed exactly once**, no infrastructure retries, no scientific expensive attempts. Python 3.13, PyTorch 2.10+cpu, 2 threads. Each of four modes: 150 AdamW steps, batch16, 24 timesteps, width32, 2 blocks, 4 attention heads, LR=.001, WD=.01, clip=1.0. All four instantiate **18,151 parameters** and start from identical seeded initialization/data. Fixed short window=6, detector threshold=.09, switch-route decision after >=12 observations. Persistence and zero-target were computed for secondary no-learning sanity checks.
- Models take same causal observation/action histories; the prediction-error detector at step t may only access state observations up to x_t (past completed transitions), never x_(t+1) or true change labels. **No oracle control leakage** detected by strict causal prefix perturbation test. Local checks passed for finite gradients, model causality, exactly one SDPA call per layer, deterministic alternate-family data, and no future access.
- Evaluation per (simulator family × regime × seed): 3 batches ×32 previously unseen worlds. World family #2 **delayed** introduces strong dependence on preceding action and is never used in training.

## Held-out results (mean squared error; lower better)

Primary on original simulator is stable **full-sequence** MSE and switch **late-half** MSE. All values are three-seed averages.

| Model | Stable primary | Switching late primary | Delayed-world stable | Delayed-world switch late | Mean CPU train seconds |
|---|---:|---:|---:|---:|---:|
| No-history | 0.323415 | 0.311912 | 0.318977 | 0.327804 | 1.227 |
| Fixed local (6 tokens) | 0.293461 | **0.296859** | 0.299161 | **0.321589** | 1.250 |
| Fixed global (full prefix) | 0.290810 | 0.312565 | **0.292742** | 0.337941 | 1.270 |
| Surprise-triggered | **0.290405** | 0.298704 | 0.297770 | 0.324738 | **1.609** |

Per-seed original simulator:

| Seed | Surprise stable | Global stable | Surprise switch late | Local switch late |
|---|---:|---:|---:|---:|
| 96311 | **0.290230** | 0.291978 | **0.308281** | 0.308774 |
| 96312 | 0.298766 | **0.297392** | 0.288876 | **0.285782** |
| 96313 | **0.282220** | 0.283059 | 0.298954 | **0.296022** |

**Predeclared primary gate FAILED.** Surprise has a tiny mean stable advantage versus full prefix (0.000405 absolute, ~0.14%), but **loses the changing-world endpoint to fixed local** (0.298704 vs 0.296859); it wins only 1/3 switch seeds and 2/3 stable seeds. It is **28.7% slower in CPU training than fixed local** (single SDPA per block does not make prefix ridge fitting free).

## Detection is not general reasoning

The causal detector's fraction of late timesteps choosing local attention:
- Original-world stable **0.1209**, switched **0.7263** — meaningfully different on this simulator.
- **Unseen delayed-action simulator:** stable **0.7992**, switched **0.9010** — nearly always local, even without a regime change.

This is clear **out-of-distribution detector failure**. It overreacts to perfectly stable worlds whose dynamics do not match its linear/instantaneous system-identification assumption. Despite correctly responding to regime switches on the development simulator, it does not yield superior predictions. A detector/route statistic is not a downstream utility metric.

## Stop/continue decision

**STOP this ridge-surprise hard router as a paid-compute candidate.** The original user's broader hybrid SI idea is **not falsified** by this negative synthetic experiment. Fixed local/global attention and history-aware prediction remain worthwhile baselines, but no benefit justifies GPUs yet.

The next zero-GPU step, if continuing, is to test a *learnt or Bayesian uncertainty/change-point module* against strong classical change detectors **under a second dynamics family included in development and a third fully held out**, and to measure *actual action-planning success* instead of only one-step prediction error. Further work also must reconnect to and ablate the complete original attention/recurrent/MoE/memory/refinement model; this isolated 2-layer dense mechanism is not a substitute for that architecture.

Keep PR draft, require original exact-head CI and new drift review before any merge. No compute credit, no GPU, no automation, no rerun of consumed seeds. Code/experiment records are intentionally not written to main.
