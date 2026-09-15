# Physics Tokenizer Phase 28 — Context-Only Complexity Selection

Status: **PASS** under the preregistered 57/57 gate.

Phase 27 showed that always using the 62-feature interaction library was too flexible: Burgers needed it, but Gray–Scott ID-h8 failed in all three replicates and advection ID failed once. Phase 28 kept the same tokenizer, context length, ridge, 26-feature library, and 62-feature library, but added a family-agnostic selector.

For each trajectory, both candidate models were fitted on the first 5 observed transitions. The next 2 observed context transitions were held out for model selection. Selection used a BIC-like held-out score: `N*log(validation_MSE + 1e-12) + K*log(N)`, with K equal to the number of regression coefficients (52 for the 26-feature model and 124 for the 62-feature model). Ties went to the simpler model. After selection, the chosen model was refit on all 7 observed context transitions and recursively rolled out through the unchanged 18-bit companded token projection. No family label or future target outside the observed context was used.

All three fresh registered replicates passed all 19 five-family split gates: **57/57 cells below raw persistence**, with replicate pass counts 19/19, 19/19, and 19/19.

Family-level results across registered cells:
- Wave: 9/9 pass; median ratio 0.00230; mean 62-feature selection fraction 0.0069.
- Gray–Scott: 12/12 pass; median ratio 0.0497; worst ratio 0.9123; mean 62-feature selection fraction 0.1111.
- NLS: 12/12 pass; median ratio 0.2626; worst ratio 0.8532; mean 62-feature selection fraction 0.0052.
- Advection: 12/12 pass; median ratio 0.0560; worst ratio 0.6937; mean 62-feature selection fraction 0.1111.
- Burgers: 12/12 pass; median ratio 0.2052; worst ratio 0.8155; mean 62-feature selection fraction 0.8611.

The selector therefore learned the expected complexity pattern from context alone: almost always simple for wave/NLS, usually simple for Gray/advection, and usually interaction-rich for Burgers. This resolves the specific Phase-27 overflexibility failure without a law-family router.

No registered seed was retried or replaced. All runs were local CPU only. This remains a benchmark on known synthetic dynamics; it is not evidence of unrestricted scientific-law discovery or new physics.
