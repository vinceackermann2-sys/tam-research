# Physics tokenizer Phase 10 — tail-aware companded innovation screen

Date: 2026-09-13

Status: **COMPLETE — GO (engineering screen only)**

Branch: `experiment/physics-tokenizer-phase10-companded-innovation-20260913`

Parent Phase-9 HEAD: `f66c27a9fbf81b1752d759cc058be801e077ebee`

## Question

Phase 9 showed that a learned causal innovation RVQ helped spectral-only OOD but failed to robustly improve the harder wave speed+spectral OOD split. Its decoded innovation context was substantially more distorted than absolute-state RVQ on that combined shift. Phase 10 tested the preregistered hypothesis that this failure was substantially a support/tail mismatch in the learned innovation codebook.

The intervention was deliberately representation-only. The downstream context-fitted identifier, context length, feature basis, ridge, raw trajectories, and primary metric were frozen.

## Preregistered design

The protocol was committed before any registered Phase-10 execution at `5e8db2e7c7767735a8cfe951f0ccade81c74facb`.

Three paired 18-bit-per-cell-per-frame arms were compared on identical raw trajectories:

1. **State RVQ:** 2048-way coarse token + 128-way residual token.
2. **Plain innovation RVQ:** the Phase-9 representation class, also 2048+128.
3. **Tail-aware companded innovation:** two 9-bit channel indices packed into one 18-bit integer token.

For the companded arm, innovations are causal: `d_t = x_t - xhat_{t-1}`. Training-mixture innovation statistics define `z=(d-mu)/sigma`; then `y=asinh(z)`, clipped to `[-asinh(16), +asinh(16)]`. Each channel is uniformly quantized to exactly 512 reconstruction levels including the endpoints. Decoding applies `sinh` and reconstructs `xhat_t=xhat_{t-1}+dhat_t`. No OOD data, law labels, hidden parameters, governing equations, or future states select the range or token.

The downstream identifier remained the Phase-7/9 three-frame, 16-feature D4/local polynomial physical-coordinate ridge model with per-sequence RMS feature scaling and ridge `0.001`.

Primary evaluation was wave combined OOD: 48 sequences, `c in [1.15,1.35]`, initial `max_mode=6`, horizon 3, evaluated as rollout MSE / raw-state persistence MSE.

## Implementation validation

The exact implementation was sealed before registered execution. The final validated source commit is `13f58b72d1e7430c46d0e4397f0ccbf85a6c34be`; source blob SHA is `1a61b8d0737b8c6beacfc2eabff3fae70140c0fb`. The invariant test commit is `1d06de2fd2e3444f258223e9397a15efe6a7e733`.

The implementation verified:

- packed tokens span exactly `0..262143`, i.e. 18 bits;
- endpoint levels decode to standardized innovations `-16` and `+16` within floating-point tolerance;
- the encoder is prefix-causal under future-frame perturbation;
- static zero innovation remains static within the expected central-bin quantization tolerance;
- all three arms receive identical raw trajectories;
- the optimized 2-D KD-tree nearest-centroid assignment matches brute-force nearest-centroid argmin exactly in randomized validation.

Two separate pre-registered-seed full-cost smoke attempts timed out before durable output while using the original brute-force distance path. Those throwaway smoke seeds were retired and are not evidence. The KD-tree change only accelerates the same exact nearest-centroid Lloyd assignment; it does not change the frozen representation or metric. A fresh non-registered full-cost smoke then completed successfully. No registered seed was touched before the final code/test seal.

## Registered primary results

| Replicate | State RVQ / persistence | Plain innovation / persistence | Companded / persistence | Companded / plain | Companded context MSE | Plain context MSE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.891729 | 1.043714 | 0.332611 | 0.318680 | 0.007789 | 0.045639 |
| 2 | 1.414126 | 0.951328 | 0.353958 | 0.372067 | 0.005606 | 0.030939 |
| 3 | 0.687846 | 0.732756 | 0.333205 | 0.454729 | 0.003434 | 0.023902 |
| Median | 0.891729 | 0.951328 | 0.333205 | 0.372067 | 0.005606 | 0.030939 |

The three registered result commits are:

- replicate 1: `02e8ca223c1a9fd8352a9a604a11715a6aa6fd81`
- replicate 2: `c9b6b641891f8a1e98b09c374b577950dfc37a68`
- replicate 3: `2704cfcc8c273e16081e3afacf09925b6427cf0c`

No registered replicate was retried or replaced.

## Frozen GO rule

All preregistered conditions passed:

1. companded primary ratio was strictly lower than plain innovation in all 3 pairs;
2. companded primary ratio was strictly lower than state RVQ in all 3 pairs;
3. median `companded/plain = 0.372067`, below the frozen `<=0.80` threshold;
4. companded beat raw persistence in all 3 pairs;
5. companded combined-context reconstruction MSE was lower than plain innovation in all 3 pairs.

**Frozen Phase-10 decision: `GO`.**

The aggregate result is sealed in `results/physics_tokens/phase10_companded_innovation_summary.json` at commit `026eb89ae8de2391d2ed4fa89b9b9d25b70ff394`.

## Mechanistic diagnostics

The improvement is consistent with the preregistered tail/support hypothesis, although Phase 10 does not prove that this is the unique mechanism.

On combined OOD, median target-oracle ratio fell from `0.752657` for plain innovation and `0.551059` for state RVQ to `0.024473` for the companded representation. Median raw-context-fit recursive projection ratio for the companded arm was `0.051744`. Median companded context clipping was only `4.29%`, and median recursive-rollout clipping was `3.40%`.

The large oracle and raw-fit projection gains show that the fixed companded coordinate system greatly reduced representation/projection error under this combined shift; downstream context identification then remained usable from those reconstructed states.

## Secondary results and important caveat

Spectral-only OOD was strongly favorable in all three replicates. Median rollout ratios were state `0.198341`, plain innovation `0.165431`, and companded `0.019425`; median companded/plain was `0.086277`.

Long-horizon in-distribution behavior was **not uniformly stable**. Companded ID-h8 ratios were `0.058112`, `1.016651`, and `0.307927`. Replicate 2 was slightly worse than raw persistence, and the median companded/plain ratio on ID was `1.403784`. Therefore Phase 10 must not be summarized as a universally superior codec. The positive preregistered result is specifically the combined-OOD tail-aware representation screen, with strong spectral-OOD support and a real long-horizon-ID robustness issue that must be carried into any next stage.

## Execution provenance and claim boundary

All registered Phase-10 work ran on local CPU. No Modal job, GPU job, or paid scientific workflow was used. `main` was not modified and unrelated research was left untouched.

This GO is an **engineering representation-screen result on known synthetic PDEs**. It authorizes only a separately preregistered follow-up testing robustness/generalization of the frozen representation. It is not evidence of novel physics, a universal tokenizer, unrestricted scientific-law discovery, AGI, or SSI.
