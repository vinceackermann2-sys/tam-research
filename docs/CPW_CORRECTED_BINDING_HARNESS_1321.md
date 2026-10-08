# CPW corrected-binding matched five-arm harness — #1321

**Status: implementation-only, zero-GPU preauthority.** Parent experimental protocol: [#1319](https://github.com/vinceackermann2-sys/tam-research/issues/1319).

## Why this version exists

The historical v4/v5 recall generator was invalid as a test of *key-conditioned binding* because a key-independent positional decoder scored 100% (audit #1301). The corrected #1307 generator fills eight value slots and chooses the queried key independently, so four different queries over the same source prefix must produce four distinct associated values. It is not legitimate to compare v4/v5 historical 100% Transformer data with this revised task as if it were the same evaluation.

## Five frozen implementations

| Arm | Parameters | Module |
|---|---:|---|
| Transformer | 24,940,288 | `tam_research.models.ResearchLM` |
| Sequence-only | 21,745,408 | `cpw_v3_sparse_world` with no WORLD block |
| AFM-last1 | 21,721,344 | Frozen `cpw_v5_afm` |
| AFM-first1 | 21,721,344 | Same frozen AFM swapped to block 0 by #1313 |
| R1-final | 21,745,408 | Fourteen local blocks + one 48-wide single-head causal attention block |

AFM-last1 versus AFM-first1 is a parameter-exact placement comparison; R1-final versus sequence-only is a parameter-exact retrieval-mechanism comparison. **The Transformer is larger**, so this is a capability/compute-controlled comparison, not a five-way exact parameter match.

## What the harness does

- Reuses `make_binding_batch` from corrected #1307 verbatim.
- Trains only on the final query token, with the same optimizer family and paired dataset seeds.
- Separately evaluates delays 32, 64, 128, and 256 on fixed held-out streams.
- Creates four queries for the identical 319-token source prefix, and reports both per-query accuracy and **all-four-correct source-group accuracy**.
- Verifies the symbolic key oracle is exact and the key-blind fixed-position control succeeds on **exactly one of four queries (25%)**, not necessarily the random-token baseline.
- Reports parameter counts, loss, training throughput, wall-clock compute, and VRAM (when CUDA is separately authorized).
- Has no Modal/H100 launcher, no preregistered scientific seeds, no result namespace, and no authority to run paid science.

## Required future separate authorization

Before paid training: freeze one executable merged SHA, fresh smoke and three replication seeds, an unused result root, one-shot retries=0, no resume, and an account2 credit/cost admission check. Apply #1319 validity and leakage gates. If R1 succeeds while AFM fails, it supports a **competitive retrieval control** rather than a new general architecture breakthrough. If early AFM succeeds and late fails, it supports a placement-specific mechanism under this test.

Do not treat CPU smoke behavior as model capability; no success claim is made here.
