# Physics Tokens Phase 6 — Shared Context System Identification (2026-09-13)

## Status

**FAIL under the preregistered 3/3 shared-law gate.**

Phase 6 used one shared 1024+64 residual-VQ representation trained jointly on wave and Gray-Scott trajectories. The transition law was fitted directly from three quantized context frames with the same generic 25-feature local polynomial basis for every sequence. No law identifier, family router, governing equation, or hidden parameter label was supplied.

## Primary results

| Replicate | Wave ID h8 | Wave spectral h3 | Wave combined h3 | Gray ID h8 | Gray sharp h3 | Gray parameter h3 | Gray combined h3 | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.3827 | 0.5865 | 1.3475 | 0.4977 | 0.2004 | 0.0752 | 0.3918 | no |
| 2 | 0.7427 | 0.1896 | 0.5434 | 0.7179 | 0.6575 | 0.0603 | 0.3257 | yes |
| 3 | 0.5198 | 0.2883 | 1.1214 | 0.4761 | 0.3196 | 0.1275 | 0.3907 | no |

Values are MSE divided by raw-state persistence MSE. The frozen rule required every replicate to remain below 1.0 on all seven tests. Replicates 1 and 3 fail the same combined wave OOD condition at 1.3475× and 1.1214×; replicate 2 passes it at 0.5434×. All Gray-Scott gates pass in all three replicates.

## Interpretation

The result rejects the stronger claim that a single unconstrained dense local map inferred from only three quantized frames is robust across both law families and all tested shifts. The shared token representation remains viable; the repeated failure is isolated to system identification for the hardest wave regime.

Exploratory follow-up shows that sparse term selection improves that wave regime but can destabilize long Gray-Scott rollouts. This motivates context-based model selection between generic identifiers rather than a fixed dense or fixed sparse rule.

This remains synthetic systems research, not evidence of new fundamental physics.
