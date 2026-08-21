# TAM Breakthrough Evaluation Protocol

Status: preregistered research protocol. Results do not change the pass/fail criteria below.

## 1. Question

Does an attention + learned recurrent world-state architecture provide a reproducible improvement over a parameter-matched Transformer that survives fair token, hardware, and compute comparisons, scales with model/data size, and generalizes beyond next-token loss?

The current primary candidate is **TAM v3**. ATAM remains a separate associative-memory research path; the 25M/10M mechanism ablation showed that removing ATAM from TAM v2 improved held-out language loss on three fresh matched seeds.

## 2. Claims are separated

We will not collapse different notions of "better" into one claim.

1. **Sample efficiency**: lower held-out loss at equal parameters and equal training tokens.
2. **Training efficiency**: lower held-out loss at equal H100-seconds / measured compute budget.
3. **Hardware efficiency**: throughput, peak VRAM, and long-context scaling.
4. **Capability**: downstream language, reasoning, code, retrieval, state tracking, and continual-update performance.
5. **Scaling**: whether any advantage persists or grows as parameters, data, and context increase.

A sample-efficiency win alone is not a breakthrough claim.

## 3. Frozen fairness rules

Unless an experiment is explicitly labeled an ablation:

- Same tokenizer and exact token files.
- Same train/validation split and deterministic data stream.
- Same seed per matched pair.
- Same context length and global tokens per optimizer step.
- Same optimizer family, betas, weight decay, LR schedule, warmup policy, precision, gradient clipping, and evaluation protocol.
- Same exact GPU class (`H100!`) for hardware comparisons.
- Parameter mismatch target: <=0.1%; preferred <=0.01%.
- Report both steady-state training time and total cold-start compute including compilation.
- Failed or unfavorable runs are retained in the record.
- Seeds and budgets are declared before launch.
- No architecture change is evaluated under a larger token budget before it passes the smaller preregistered gate.

## 4. Primary metrics

For each run record:

- held-out NLL (primary quality endpoint)
- perplexity
- parameter count
- tokens seen
- seed
- context length
- H100 model
- steady-state tokens/sec
- compile/setup seconds
- total H100-seconds
- peak training VRAM
- learning curve checkpoints at matched token counts
- route/gate statistics when applicable

At larger scale also estimate training FLOPs and report loss vs FLOPs in addition to loss vs wall-clock/dollars.

## 5. Gate A — 25M mechanism validation

Already motivated by TAM v2 experiments.

Required:

- >=3 matched seeds at 10M tokens.
- Component ablations for memory, world-state, and routing/gating.
- Residual-output-scale control so mixer amplitude cannot masquerade as an architecture improvement.
- Compiler/kernel profile so eager framework overhead is not confused with unavoidable architecture cost.

Interpretation:

- If a component can be removed and quality improves, it is not credited with the gain.
- The simplest surviving mechanism becomes the next candidate.

## 6. Gate B — TAM v3 25M validation

### B1. Smoke

1M tokens, one fresh seed, Transformer vs TAM v3. Purpose: correctness, compilation, checkpointing, throughput, VRAM. Do **not** interpret 1M NLL as decisive capability evidence.

### B2. Equal-token replication

10M tokens, three fresh preregistered seeds:

- Transformer
- TAM v3

Pass condition for further scaling:

- TAM v3 lower mean held-out NLL, and
- wins >=2/3 seeds, preferably 3/3, and
- no numerical instability.

### B3. Residual/gate controls

At the same 25M/10M setting, run controls sufficient to rule out a trivial residual-amplitude explanation, including a Transformer residual-gain control and a variance-preserving TAM v3 mixture or equivalent normalization control.

TAM v3 must retain a meaningful advantage after this control to attribute the gain to world-state structure.

### B4. Equal-compute comparison

Using measured steady-state throughput, allocate token budgets so Transformer and TAM v3 receive approximately equal H100 training time. Report two versions:

- amortized equal compute: compilation excluded as a one-time setup cost
- cold-start equal compute: compilation included

TAM v3 does not need to win cold-start cost at tiny token budgets, but a serious architecture claim requires a competitive amortized compute frontier.

## 7. Gate C — scaling laws

Do not jump directly to one giant run. Build paired curves.

Initial grid:

| Approx params | Seeds | Token horizons |
|---:|---:|---|
| 25M | 3 | 10M, 25M, 50M, 100M |
| 50M | 3 | 25M, 50M, 100M, 200M |
| 100M | 3 | 50M, 100M, 200M, 400M |
| 200-300M | 2-3 after prior gates | compute-optimal horizon chosen from fitted curves |

Fit separate empirical relationships for loss vs:

- parameter count N
- training tokens D
- measured training compute C

Candidate form:

`L(N,D) = L_inf + A*N^(-alpha) + B*D^(-beta)`

and a direct compute frontier `L(C)`.

Evidence becomes substantially stronger if TAM v3's gap is stable or grows with scale. A one-scale crossover is not sufficient.

## 8. Gate D — context and state scaling

Train/evaluate matched models across context lengths as resources permit:

512 -> 1K -> 2K -> 4K -> 8K -> 16K+

Measure:

- NLL
- training throughput
- inference throughput
- peak VRAM
- retrieval/state-tracking accuracy
- degradation as distance between write and query increases

The world-state hypothesis predicts an advantage on information that must be retained or updated over long spans, not merely on local next-token statistics.

## 9. Gate E — capability suite

A breakthrough claim requires generalization beyond one language-loss benchmark.

Minimum categories:

- ordinary language modeling
- common-sense / factual completion
- reasoning
- code
- associative retrieval
- repeated-key overwrite / latest-value retrieval
- state tracking
- interference resistance
- long-document retrieval
- continual update / stale-belief replacement

A canonical continual-update test should explicitly contain:

1. early fact or strategy A is correct
2. long unrelated interval
3. A becomes stale and B becomes correct
4. another long interval
5. query asks for the current answer

Score both correct updating and resistance to resurrecting stale state.

## 10. Strong architecture baselines

Transformer is the first baseline, not the final one. Before a broad novelty/breakthrough claim, compare against representative efficient recurrent/hybrid families when implementation quality permits, including:

- Mamba-2
- Gated DeltaNet / strong delta-rule recurrent baseline
- Griffin/Hawk-style gated recurrence + attention baseline
- RWKV-7-class dynamic-state baseline

For long-context/persistent-memory claims, discuss and where feasible compare against newer memory/test-time-learning approaches such as Titans and end-to-end TTT. These baselines answer whether TAM discovered a genuinely better frontier rather than merely rediscovering the known benefit of adding recurrent state to attention.

References for protocol planning:

- Mamba-2 / structured state-space literature
- Gated Delta Networks, arXiv:2412.06464
- Griffin, arXiv:2402.19427
- RWKV-7, arXiv:2503.14456
- Titans, arXiv:2501.00663
- End-to-End Test-Time Training for Long Context, arXiv:2512.23675

## 11. Statistical reporting

For every multi-seed gate:

- report every seed
- mean and standard deviation
- paired per-seed delta when seeds/data are matched
- bootstrap or paired confidence interval once sample count is sufficient
- do not report only the best checkpoint unless checkpoint selection is identical and preregistered

At the scaling stage, fit both architectures jointly where useful and report uncertainty on fitted exponents/frontiers.

## 12. Breakthrough threshold

We may describe TAM as an **interesting architecture signal** before all criteria pass. We reserve a strong "breakthrough" claim for evidence approximately this strong:

1. Reproducible advantage across >=3 seeds.
2. Advantage across >=3 parameter scales, including 100M+.
3. Competitive or superior amortized equal-compute frontier.
4. Advantage is not explained by residual scaling, parameter mismatch, optimizer choice, or framework/kernel inefficiency.
5. Improvements extend beyond next-token NLL into relevant capability/state tasks.
6. Long-context/state behavior demonstrates a mechanism-level advantage consistent with the world-state hypothesis.
7. Competitive comparison against strong modern recurrent/hybrid baselines.
8. Ablations identify which component causes the improvement.
9. No catastrophic training/inference throughput or memory penalty after reasonable optimization.
10. Results remain when the protocol is repeated on a second corpus or data mixture.

Only then should we spend substantially more compute on large-scale confirmation or make a broad architectural claim.

## 13. Stop / redirect conditions

Stop blind scaling and redesign if any of these occur:

- TAM loses consistently at equal tokens after mechanism controls.
- The advantage vanishes under residual-scale control.
- The compute frontier is materially worse and kernel optimization cannot close it.
- The gain disappears at 50M/100M.
- A standard modern recurrent/hybrid baseline matches or exceeds TAM with simpler machinery.

Negative results are useful: they narrow the mechanism and prevent expensive scaling of the wrong architecture.

## 14. Observed evidence ledger

This section records results after the preregistration above. It does **not** change any pass/fail criterion.

### 2026-08-21 — 50M equal-token replication at 10M tokens

Configuration: context 512, micro-batch 64, gradient accumulation 2, matched ~49.8M parameters, same FineWeb-Edu/GPT-2 token stream and H100 class.

| Seed | Transformer NLL | TAM v3 NLL | Winner |
|---:|---:|---:|---|
| 7102 | 7.0969 | 6.8978 | TAM v3 |
| 7103 | 7.0550 | 6.8577 | TAM v3 |
| 7104 | 7.0404 | 6.9101 | TAM v3 |

Observed means:

- Transformer NLL: **7.0641**
- TAM v3 NLL: **6.8885**
- Mean NLL delta: **-0.1756** for TAM v3 (~2.49% lower)
- TAM v3 wins: **3/3 seeds**
- Transformer steady-state throughput: ~**452.9k tok/s**
- TAM v3 steady-state throughput: ~**370.4k tok/s**
- TAM v3 throughput ratio: ~**81.8%** of Transformer
- Peak VRAM at this batch: Transformer ~29.75 GiB; TAM v3 ~32.98 GiB

Interpretation: the 50M equal-token replication passes the current quality/stability scaling gate. The gap is larger than the earlier 25M signal, which is encouraging for the scaling hypothesis, but two parameter scales are not enough to establish a scaling law.

### 2026-08-21 — 100M smoke only

At ~101.8M parameters, context 512, micro-batch 64, accumulation 2, seed 7201, both models trained stably for 1M tokens. Transformer NLL was 8.9265 and TAM v3 NLL was 8.9263; the difference is effectively zero at this intentionally undertrained smoke horizon. Steady-state throughput was ~280.4k tok/s for Transformer and ~225.9k tok/s for TAM v3 (~80.6% ratio), with TAM v3 peak VRAM ~43.04 GiB.

**Status:** implementation/hardware gate passed; scientific 100M replication remains pending. Do not treat the 1M-token smoke as evidence for or against the architecture-quality claim.
