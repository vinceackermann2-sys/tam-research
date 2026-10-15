# SI original hybrid action-world Gate 1 — fixed-seed CPU results

Date 2026-10-15. Source: `scripts/si_original_action_probe.py` as frozen at `b93a1ab94c8a6df4d88e9d24a0060180deac6b98`; protocol at `docs/SI_ORIGINAL_ACTION_GATE1_PROTOCOL_2026-10-15.md`. Paired new engineering seeds 94111 / 94112 / 94113; 300 AdamW updates, batch 16, length 24, 4×32 held-out independent worlds per evaluation mode, identical streams per model. CPU-only; no Modal compute, no claims of scientific breakthrough.

**MSE lower better.** Hybrid 64,614 parameter / Transformer 64,637. Stable primary is full-sequence held-out MSE, with per-seed confirmation:

| Seed | Hybrid stable | Transformer stable |
|---|---:|---:|
| 94111 | 0.293625 | **0.290338** |
| 94112 | 0.287531 | **0.283252** |
| 94113 | 0.288743 | **0.283988** |
| Mean | 0.289967 | **0.285860** |

Secondary OOD, means over three matched seeds:

| Mode / metric | Original hybrid | Transformer | Persistence | Zero |
|---|---:|---:|---:|---:|
| Stable late (t>=12) | 0.279720 | **0.273679** | 0.425672 | 0.422978 |
| **Abrupt-switch late** | **0.324813** | 0.353960 | 0.441883 | 0.415060 |
| Nonlinear late | 0.281543 | **0.276744** | 0.422602 | 0.433853 |

Mean switch whole-sequence MSE: hybrid **0.312513**, Transformer **0.326000**. Switch late per-seed hybrid [0.328818, 0.322039, 0.323583], Transformer [0.357745, 0.345326, 0.358809]. Neither baseline won the switch late task. The switch improvement is therefore **about 8.2% lower late MSE than Transformer**, but switch was secondary, not the primary gate and **not** evidence of a breakthrough or causal mechanism on its own.

Train-only reference CPU wall time averaged **4.631 s** hybrid vs **2.850 s** Transformer, 1.63× slower hybrid. Dense Tensor/loop implementations are not kernel-optimized and this is not an equal-FLOPs comparison; CPU performance is a real practical warning.

**Verdict on the frozen gate: FAIL for advancement to paid compute** because Transformer wins mean stable primary and all 3 paired seeds. **Exploratory signal:** investigate why the hybrid is better under abrupt nonstationarity and which component is responsible. Do not claim the overall model is better. No automatic trigger, rerun, or extra GPU authority. Independent ablations on fresh seeds and an optimized compute comparison are required.

### Next bounded zero-GPU test

Run full hybrid vs `no_recurrence` (replace recurrent layers with local attention), `no_moe` (dense FFN), `no_refine` (no optional iterative refinement), and a Transformer baseline, with identical data but **different parameter counts for ablations**, on fresh development seeds. Assess switch late error **and** stable error and wall time. These are mechanistic screens, not inferentially clean matched-parameter ablations. If none of the components explains the secondary switch gain, revisit evaluation leakage/architecture interactions. No reason yet to spend $25 credits.
