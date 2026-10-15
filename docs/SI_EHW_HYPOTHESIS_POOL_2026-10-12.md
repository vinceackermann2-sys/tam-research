# SI EHW hypothesis pool — frozen exploratory confirmation

Date: 2026-10-12. Independent of TAM/CPW/RLT research. No GPU or scientific seed spent.

## Motivation

Phase-0 EHW used an episode-local ridge transition hypothesis fit to all observations. This assumes the dynamics are stationary. On a world where transition rules change partway through the sequence, that approximation is wrong and EHW lost to the Transformer.

## Hypothesis pool variant

Fixed candidate context windows (12, 8, 6 observations). For each candidate, fit a ridge transition operator on its observations excluding the final two; score on those two held-out observations plus a small penalty for shorter windows; refit the selected candidate on its entire window; predict the query. The query target is **not** used to select hypotheses. This is **not a learned compiler** and the rule family remains almost linear.

The implementation `experiments/si_ehw_hypothesis_pool.py` was committed with the fixed windows (12/8/6), two-observation validation, ridge lambda=0.14 and penalty=0.001 **before** checking confirmation seeds 8401–8403. Earlier seeds 8301–8303 were exploratory and cannot be treated as blinded evidence.

Reproduce from repo root:

```bash
for seed in 8401 8402 8403; do
  python experiments/si_ehw_hypothesis_pool.py --seed "$seed"
done
```

## Fresh seed confirmation: 8401/8402/8403

MSE, mean across 3 seeds × 8 batches × 128 worlds (lower is better). Same random world streams for hypothesis-pool and ridge-all within each seed.

| Task | Hypothesis pool | Ridge-all | Difference |
|---|---:|---:|---|
| Stable unseen laws | 0.005954 | **0.005395** | Pool 10.4% worse |
| Stronger transition weights | 0.012632 | **0.011585** | Pool 9.0% worse |
| New quadratic effects | 0.046520 | **0.040005** | Pool 16.3% worse |
| Abrupt change in laws | **0.056401** | 0.074529 | Pool 24.3% better |

Switch per-seed hypothesis-pool MSE: **0.054805, 0.057837, 0.056562**.

The pool recovers a nontrivial amount of switch adaptation, but sacrifices stable-world accuracy. There is no claim this outperforms a well-trained Transformer, learned change-point detector, classical regime-switching filter, or fully parameter/compute-matched adaptive baseline.

## Interpretation and future falsification

What appears useful is **selecting among executable hypotheses on predictive evidence**. What is not yet supported is the idea that an AI can *learn to invent the right hypothesis class*, adapt causally to unseen nonlinear dynamics, choose informative interventions, or retain general long-term skills.

Before claiming architecture progress: replace hand-coded linear candidates with learned compositional causal programs, jointly evaluate stable-vs-switch tradeoff, compare existing change-point models and adaptive Kalman filters, and use preregistered non-oracle held-out worlds. Test actual downstream action outcomes, not merely one-step MSE.

Status: exploratory result with fixed fresh-seed confirmation; no breakthrough claim, no paid compute authority.
