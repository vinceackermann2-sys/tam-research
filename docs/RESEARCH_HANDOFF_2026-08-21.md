# TAM Research Handoff — 2026-08-21

This document is the canonical recovery point for the TAM research program as of 2026-08-21. It is intentionally self-contained enough that a fresh researcher or ChatGPT session can reconstruct what was attempted, what worked, what failed, what is running, and what should happen next without relying on chat history.

## 1. Research goal

Primary question:

> Can an attention + learned recurrent world-state language model achieve a reproducible quality/compute frontier that is better than a parameter-matched Transformer, and does that advantage persist or improve with scale?

Longer-term product question:

> If the architecture survives scaling and capability tests, can it become the core of a tool-using, persistent-memory agent that completes real computer work more efficiently than a comparable Transformer?

The strong word **breakthrough** is reserved for the preregistered criteria in `docs/breakthrough_protocol.md`. Current evidence is promising architecture/scaling evidence, not yet a broad breakthrough claim.

## 2. Current status at handoff

Completed:

- ATAM associative-memory prototype and synthetic recall experiments.
- Parameter-matched Transformer / ATAM / TAM v1 screens.
- TAM v2 three-path architecture and 3-seed 25M screen.
- Mechanism ablations showing the ATAM path is not responsible for the language-loss gain.
- TAM v3: attention + recurrent world-state, nearly exact parameter match to Transformer.
- H100 profiling and substantial systems optimization.
- 25M TAM v3 replicated equal-token win: 3/3 seeds.
- 50M TAM v3 replicated equal-token win: 3/3 seeds.
- 100M model-fit/runtime smoke: both models fit on one H100 and remain finite.
- Persistent compiler-cache infrastructure, launcher hardening, version-safe cache namespace.
- Preregistered 100M replicated gate in issue #66.

Active at handoff:

- **Issue #67**: `[modal-scale] 100M replicated gate — Transformer vs TAM v3`
  - scale: 100M
  - token budget: 10M per model
  - architectures: Transformer, TAM v3
  - seeds: 7202, 7203, 7204
  - context: 512
  - micro-batch: 64
  - grad accumulation: 2
  - H100 pinned via `H100!`
- **Issue #65**: compiler-cache cold→warm proof v3. Cold seed 7304 completed; the warm replica was spawned by the launcher and may still be pending/running.

Budget guard:

- At this handoff the user reported about **$3.5 Modal compute credit remaining**.
- Issue #67 is the only new compute experiment intentionally launched after that statement.
- Do not start 300M/1B or additional sweeps until the remaining Modal balance is checked.

## 3. Architecture evolution

### ATAM — Adaptive Temporal Associative Memory

Original specialized memory architecture. It learned predecessor→successor bindings with content lookup, write gates, and temporal bias. It was extremely strong on purpose-built associative recall but did not demonstrate a decisive general language advantage.

Synthetic recall highlights from early local experiments:

- 8 pairs: ATAM ~100%, Transformer ~27%
- 16 pairs: ATAM ~100%, Transformer ~15%
- 32 pairs: ATAM ~99.7%, Transformer ~8%
- 96-pair overwrite stress: ATAM-15K ~83.2%, Transformer ~6.64%

Interpretation: strong specialized memory mechanism, not evidence of general LM superiority.

### TAM v1

Parallel half-width causal attention + half-width ATAM with learned scalar gate. Parameter-matched to Transformer. Mixed results on tiny byte-language screens; useful as evidence that hybridization was plausible, but not a stable general winner.

### TAM v2

Three causal paths:

1. causal self-attention
2. ATAM associative memory
3. recurrent world-state

A learned token-wise router selected among the paths. In a 25M local screen it showed a strong language-loss signal, but serious H100 profiling showed the ATAM and world-state paths were both expensive in eager execution.

Mechanism ablations at 10M tokens, seed 7042:

- full TAM v2: NLL 7.2366
- fixed 1/3 router: 7.1950
- no-world: 7.2081
- no-memory: **7.1423**

No-memory was then replicated on two fresh seeds and won again. Across seeds 7042/7043/7044, the no-memory variant beat full TAM v2 3/3. This changed the main hypothesis: the language benefit appeared to come primarily from **attention + recurrent world-state**, not ATAM.

### TAM v3 — current primary architecture

TAM v3 removes ATAM from the main language mixer and keeps:

- reduced-width causal attention
- recurrent learned world-state
- one scalar learned mixture gate

For each block:

```text
h = LayerNorm(x)
attention = causal_attention(h)
candidate = tanh(W_candidate h)
keep = sigmoid(W_keep h)
state_t = keep_t * state_(t-1) + (1 - keep_t) * candidate_t
world = W_out state

g = sigmoid(gate_logit)
mixer = 2 * ((1-g) * attention + g * world)

x = x + mixer
x = x + MLP(LayerNorm(x))
```

The recurrence is evaluated with a parallel diagonal affine scan, preserving causality.

Exact implementation: `tam_research/models.py` (`RecurrentWorldState`, `TAMV3Mixer`, `Block`, `ResearchLM`).

## 4. Current model family

Shared:

- tokenizer vocabulary: GPT-2, 50,257 tokens
- learned token embeddings, tied output head
- learned positional embeddings
- LayerNorm + GELU MLP
- FF multiplier: 4
- max configured sequence length: at least 1024; main experiments use context 512

Matched scale family:

| Label | Transformer params | TAM v3 params | d_model | layers | Transformer heads | TAM state | TAM attention inner |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25M | 24,940,288 | 24,941,263 | 256 | 15 | 8 | 64 | 208 |
| 50M | 49,799,808 | 49,801,457 | 384 | 17 | 12 | 96 | 312 |
| 100M | 101,803,520 | 101,806,616 | 512 | 24 | 16 | 128 | 416 |

Structural ratios are self-similar:

- world-state width = d_model / 4
- TAM attention inner width = 13/16 * d_model

A 300M/1B extension has been designed but should remain gated behind the 100M result.

## 5. Training/data protocol

Serious Modal runs use:

- dataset: FineWeb-Edu, Hugging Face `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- tokenizer: GPT-2
- deterministic train/validation token binaries on Modal Volume
- GPU: strict `H100!`
- precision: bfloat16
- optimizer: fused AdamW
- betas: 0.9 / 0.95
- weight decay: 0.1
- peak LR: 3e-4
- cosine decay
- warmup: 2%
- gradient clipping: 1.0
- default global sequences per update: 128
- current preferred micro/accum at 25M/50M/100M: 64 × 2 when stable
- current main context: 512

Training code: `tam_research/train.py`, `tam_research/train_scaled.py`.

Data code: `tam_research/data.py`.

Scaling launcher: `.github/workflows/modal-scaling.yml`, `modal_scale_app.py`.

## 6. Key quantitative results

### 25M TAM v3, 10M tokens, three fresh seeds

Transformer mean NLL: **7.2284**
TAM v3 mean NLL: **7.1876**

TAM v3 won 3/3 matched seeds.

Later 64×2 confirmation:

- Transformer mean NLL ~7.2070
- TAM v3 mean NLL ~7.1621
- Transformer training throughput ~572.9k tok/s
- TAM v3 ~492.6k tok/s
- TAM throughput ratio ~86.0%

Interpretation: reproducible 25M equal-token quality win after systems optimization.

### 50M, 10M tokens, three fresh seeds

| Seed | Transformer NLL | TAM v3 NLL | Delta (T - TAM) |
|---:|---:|---:|---:|
| 7102 | 7.0969 | 6.8978 | +0.1991 |
| 7103 | 7.0550 | 6.8577 | +0.1973 |
| 7104 | 7.0404 | 6.9101 | +0.1303 |

Means:

- Transformer NLL: **7.0641**
- TAM v3 NLL: **6.8885**
- mean NLL advantage: **0.1756 (~2.49%)**
- approximate perplexity reduction: **~16%**
- Transformer throughput: **~452.9k tok/s**
- TAM throughput: **~370.4k tok/s**
- throughput ratio: **~81.8%**

TAM v3 won 3/3 seeds. The observed 50M gap is larger than the 25M gap, but two scaling points are insufficient to claim a superior scaling exponent.

### 100M runtime smoke

At 64×2, 1M tokens, seed 7201:

- Transformer: NLL 8.9265, 280.4k training tok/s, 36.98 GiB peak VRAM, compile 284.32s
- TAM v3: NLL 8.9263, 225.9k training tok/s, 43.04 GiB peak VRAM, compile 564.28s
- throughput ratio: **~80.6%**

Interpretation: both ~101.8M models fit and train stably on one H100; 1M-token loss is a runtime smoke, not scientific quality evidence.

### Current 100M replicated gate

Issue #67 was launched with the exact preregistered configuration from issue #66. Results are pending at the time this handoff document was created. Update this section after all six model/seed callbacks finish.

## 7. Systems-performance findings

Initial TAM v2 eager block profiling showed a ~5x block penalty. The main costs were ATAM and world-state paths.

`torch.compile` dramatically reduced the penalty. A representative profile reduced full TAM v2 from ~5.99 ms to ~1.45 ms per block forward+backward, while a compiled Transformer block was ~0.74 ms.

TAM v3 simplified the architecture and microbatch utilization produced the biggest practical throughput improvement:

- 16×8 TAM v3: ~184k tok/s
- 32×4: ~389k tok/s
- 64×2: ~430k tok/s
- 128×1: ~455k tok/s but NaN; rejected

Thus 64×2 became the fastest stable small-scale configuration.

A chunked exact affine scan improved the world-state branch but only modestly improved whole-block time; it was not the dominant remaining bottleneck.

Projection fusion was tested but did not produce a sufficiently clean end-to-end win to promote as the production TAM v3 path.

## 8. Compiler-cache engineering

Problem: at 100M, cold `torch.compile` setup consumed hundreds of H100-seconds while the actual 1M-token training phase took only seconds.

Implemented:

- persistent `TORCHINDUCTOR_CACHE_DIR` on shared Modal Volume
- persistent Triton cache
- FX graph cache
- AOTAutograd cache
- seed-independent graph keys
- cache shape isolation
- version-safe namespace containing cache schema, PyTorch build, compile mode, accelerator, architecture, scale, context, microbatch, and accumulation
- warm-first-seed then replica-seed launch strategy

Important failures retained:

1. First cache proof failed before H100 because local GitHub runner status reporting imported PyGithub without a token/dependency. Fixed by making all status reporting best-effort/non-fatal.
2. Second proof demonstrated a warm cache but failed because `volume.reload()` was called after Torch opened cache files (`RuntimeError: there are open files preventing the operation`). Fixed by reloading the Volume before cache/Torch setup.
3. Corrected v3 proof (#65): cold seed 7304 completed with compile=294.94s and stable training; warm replica was spawned afterward.

Relevant files:

- `tam_research/compile_cache.py`
- `modal_scale_app.py`
- `tests/test_compile_cache.py`

## 9. Storage / recovery

GitHub contains:

- model architecture code
- data-prep code
- training code
- H100 launchers/workflows
- tests
- compiler-cache logic
- breakthrough protocol
- research handoff/results documentation
- GitHub issues containing authoritative run callbacks and metrics

Modal Volume `tam-research-data` contains the large runtime artifacts that are not appropriate for ordinary Git commits:

- `/vol/data/fineweb-edu-gpt2` — tokenized dataset
- `/vol/scaling-runs/...` — scale/context/batch-specific checkpoints, metrics, summaries
- `/vol/compile-cache/...` — persistent Inductor/Triton artifacts

The code supports checkpoint resume. Checkpoint files include model state, optimizer state, step, tokens seen, batch generator state, and compile metadata.

Important limitation: checkpoint binaries on the Modal Volume have not been copied into GitHub. GitHub is the canonical code/metadata/evidence record; Modal Volume is the current binary checkpoint store.

## 10. Scientific interpretation

What is supported:

- TAM v3 has a reproducible equal-token language-loss advantage at 25M.
- TAM v3 has a reproducible equal-token language-loss advantage at 50M.
- The observed advantage is larger at 50M than at 25M.
- Stable optimized throughput is ~80%+ of the matched Transformer at tested scales.
- 100M models fit and train on a single H100.

What is NOT yet supported:

- a broad architectural breakthrough claim
- superior compute scaling at 100M+
- superior reasoning/coding capability
- superior long-context behavior
- persistent lifelong internal memory
- agent superiority
- superiority over Mamba/DeltaNet/RWKV/other modern recurrent/hybrid baselines

## 11. 100M decision rule

The next gate is frozen in issue #66.

Advance beyond 100M only if the 10M × 3-seed gate satisfies:

1. TAM v3 lower mean held-out NLL.
2. TAM wins at least 2/3 matched seeds.
3. No numerical instability.
4. TAM steady-state throughput remains >=80% of matched Transformer mean, otherwise profile/optimize before scaling.

Record all seeds, including unfavorable results.

## 12. After 100M

If 100M passes:

1. Fit first empirical 25M/50M/100M loss-vs-parameter and loss-vs-compute trend.
2. Run residual/amplitude controls if any unresolved control remains.
3. Add strong modern baselines before claiming novelty: Mamba-family, Gated DeltaNet/delta-rule recurrence, RWKV-class recurrence, Griffin/Hawk-style hybrids; long-memory comparison where practical against Titans/TTT-style approaches.
4. Validate on a second corpus/data mixture.
5. Run long-context/state tests at 512 → 1K → 2K → 4K+.
6. If the scaling/capability frontier remains favorable, move to ~300M.
7. Only after 300M validates extrapolation should ~1B pretraining be considered.

Agent development should begin as a separate post-training track around 300M–1B:

- instruction/SFT
- structured tool calls
- browser/terminal/files/code environment
- multi-step trajectories
- outcome-based training
- persistent external episodic/semantic/procedural memory
- recovery/error correction
- agent benchmarks

A raw base model is not an agent.

## 13. Important issue/PR landmarks

Core proof / infrastructure history includes:

- PR #1: initial Modal TAM v2 training stack
- PR #3/#5/#7/#11: instrumentation/auth/diagnostic hardening
- issue #14: first confirmed real TAM v2 H100 training
- issue #16: early Transformer/TAM calibration
- issues #19/#20: equal-compute gate
- issue #24: H100 component profiler
- issue #26: TAM v2 mechanism ablations
- PR #30: TAM v3 architecture
- issue #51: optimized 25M confirmation
- issue #56: 50M replicated gate
- issue #58: 100M 64×2 smoke
- PR #59: persistent compiler cache
- PR #61: non-fatal status reporting
- PR #63: version-safe compiler cache
- PR #64: observed 50M evidence ledger
- issue #65: corrected compiler-cache proof
- issue #66: preregistered 100M gate
- issue #67: active 100M replicated training gate

Use GitHub issue comments as the authoritative immutable run-callback trail.

## 14. Safety rule for future continuation

Do not silently redefine the architecture or pass criteria after results arrive. Any TAM v4 or major training-protocol change starts a new preregistered experiment family. Preserve negative runs. Compare at matched parameters/data/hardware and report both token efficiency and actual compute efficiency.
