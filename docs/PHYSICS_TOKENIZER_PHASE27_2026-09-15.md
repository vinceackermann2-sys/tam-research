# Physics Tokenizer Phase 27 — Unified Interaction Library

Status: **FAIL** under the preregistered 57/57 gate.

The Phase-26 62-feature interaction library was promoted to the primary identifier and evaluated without a law-family router across five synthetic PDE families: wave, Gray–Scott, nonlinear Schrödinger, advection, and viscous Burgers. The tokenizer remained the frozen Phase-10 18-bit companded representation trained only on wave + Gray–Scott. Context stayed at 8 frames and ridge at 1e-3.

Across three fresh registered replicates the primary model passed 53/57 cells. Replicate pass counts were 18/19, 17/19, and 18/19. Gray–Scott ID-h8 failed in all three replicates with ratios 1.7812, 1.6981, and 1.8749 versus raw persistence. Advection ID-h8 additionally failed in replicate 2 at 1.4165. Wave passed 9/9 cells, NLS 12/12, Burgers 12/12, and advection 11/12.

The nested 26-feature control remained much better on the simple long-horizon Gray–Scott ID task, while the 62-feature model was required for robust Burgers transfer. Therefore Phase 27 falsifies the claim that simply using the richer interaction class everywhere is a unified upgrade. The failure is a complexity-control / model-selection problem rather than a lack of representational capacity.

No registered seed was retried or replaced. All registered runs were local CPU only. This remains a benchmark on known synthetic dynamics and is not evidence of unrestricted law discovery or new physics.

Next scientific question: can a family-agnostic selector choose between the nested 26- and 62-feature models using only held-out transitions inside the observed context, then refit the selected model on the full context before rollout?
