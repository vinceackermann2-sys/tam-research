# CHM-v1 — Exact Indexable Episodic Memory (EIEM)

Status: **exploratory mechanism pass only**. This directory is a separate research track created from issue #849. It is not TAM, AERA, or CORTEX-S evidence and does not authorize GPU spend or a breakthrough claim.

## Thesis

Long context should not require every query to compare against every historical token or to compress history into a lossy recurrent/page summary. Keep a dense local working context, preserve episodic evidence exactly outside that window, and train the memory/query representation so an index can find relevant episodes with substantially sublinear reads.

CHM-v0 tested a lossy version of this thesis: page prototypes routed queries to a few pages. It saved reads but destroyed rare-fact addressability as memory grew. At 1,024 events CHM-v0 reached 0.5566 accuracy versus 0.9531 for flat retrieval, with route recall 0.5576. CHM-v1 directly attacks that failure mode.

## Mechanism

CHM-v1 / EIEM has two conceptually separate pieces:

1. **Learned address representation.** A memory encoder and query encoder learn a low-dimensional normalized address space from retrieval supervision. The exploratory screen uses compositional identities and holds out 25% of identity pairs from training.
2. **Lossless exact index.** Every episodic key/value remains intact. A zero-parameter branch-and-bound hierarchy stores only axis-aligned bounds and membership. A subtree is pruned only if its geometric lower bound cannot beat the current best exact candidate. For a fixed address representation, the indexed lookup must therefore return the same nearest neighbor as exhaustive flat lookup.

The index is deliberately not claimed as novel: exact nearest-neighbor trees and branch-and-bound are classical. The research hypothesis is narrower: **can an LM learn memory/query address representations that stay exactly retrievable and sufficiently indexable that sparse exact memory access improves the quality/compute frontier?**

## Exploratory CPU screen

The authoritative reproducible harness is `cpu_mechanism_screen.py`. It trains only the address representation on 64-event synthetic retrieval, then evaluates out of distribution at 64, 256, and 1,024 held-out compositional identities.

Exploratory harness history:

- seed `60429`: consumed invalid harness attempt. Overwrite expectations were captured before later writes and then scored after the final state. No scientific conclusion is allowed from it and the seed must not be reused.
- seed `60430`: corrected exploratory mechanism screen. It is permanently exploratory and must never become a scientific seed.

Corrected seed-60430 result at 1,024 memories:

| Test | Flat accuracy | CHM-v1 indexed accuracy | Exact indexed-vs-flat match | Stale error | Estimated vector reads / flat |
|---|---:|---:|---:|---:|---:|
| rare fact | 1.0000 | 1.0000 | 1.0000 | — | 0.037035 |
| overwrite/latest value | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.037809 |
| two-hop | 1.0000 | 1.0000 | 1.0000 | — | 0.037363 |

The untrained-address ablation reaches only `0.0009765625` flat retrieval accuracy at 1,024, showing that the learned representation is necessary for the synthetic task.

The frozen exploratory decision is:

`PASS_MECHANISM_ONLY_PREREGISTER_SMALL_LM_TEST`

This means only that CHM-v1 fixes the specific lossy-routing failure seen in CHM-v0 while using far fewer estimated address-vector reads. It is **not** an LM result or architecture breakthrough.

## Required next gate

After this exact reproducible mechanism record passes original CI and is merged, a separately preregistered small matched-LM experiment may be built. It must use fresh scientific seeds and a fresh result namespace and compare, at minimum:

- matched Transformer/local-attention baseline;
- flat episodic retrieval using the same learned address encoder;
- CHM-v1 exact indexed retrieval.

The test must keep tokenizer, corpus, token budget, optimizer, parameter accounting and evaluation fair; report language loss plus overwrite/state/rare-fact/multi-hop capability; report training and inference cost plus actual memory/read behavior; and fail closed if indexed retrieval requires an oracle unavailable at inference.

Before any broad claim, later gates must include strong modern sparse/recurrent/memory baselines, multiple seeds/scales, equal-compute comparisons, a second data mixture, mechanism ablations and real GPU efficiency.

## Non-claims

- CHM-v1 does not prove that hierarchical memory is generally superior.
- It does not prove sublinear wall-clock GPU retrieval; the CPU screen counts address-vector reads, not production kernels.
- It does not establish novelty over modern sparse-memory architectures.
- It does not authorize scaling, a scientific seed, or a breakthrough claim.
