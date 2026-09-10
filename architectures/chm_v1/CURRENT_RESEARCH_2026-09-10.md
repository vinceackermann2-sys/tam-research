# CHM-v1 current-research / novelty map — 2026-09-10

This note exists to prevent overclaiming. CHM-v1's broad thesis is **not novel on its own**: modern long-context work already combines local processing, persistent memory, learned retrieval, sparse addressing, recurrence, test-time learning, or hierarchical organization in many ways.

## Highly relevant overlap

### Memory Layers at Scale

Sparse learned key-value memory layers show that explicit memory can add factual capacity and that sparse lookup can be practical at large parameter counts. CHM-v1 must therefore beat or complement a serious sparse memory layer rather than claim sparse lookup itself as the contribution.

### Titans / MIRAS

Titans introduces learned long-term neural memory alongside attention and test-time memorization. MIRAS generalizes memory design choices. This overlaps strongly with the goal of persistent learned memory but not specifically with CHM-v1's exact branch-and-bound retrieval guarantee.

### Engram / Conditional Memory

Engram-style conditional memory uses deterministic sparse lookup of memorized patterns. It is especially relevant because it demonstrates that cheap conditional access can improve an LM. CHM-v1 cannot claim novelty merely from avoiding dense history scans.

### Fast-weight Product Key Memory (FwPKM)

FwPKM combines dynamic episodic memory with product-key sparse access and online updating. This directly overlaps with dynamic sparse episodic retrieval and long-context generalization. A CHM-v1 claim must distinguish exact evidence preservation/indexability from product-key retrieval and compare the resulting quality/compute frontier.

### Sparse Delta Memory

Sparse Delta Memory adds sparse addressing to a recurrent/delta-memory family. This overlaps with the hypothesis that recurrent memory benefits from selective access. CHM-v1 is not novel simply because it uses sparse addressing.

### SEEM / E-mem and hierarchical episodic systems

Recent episodic-memory work emphasizes structured events, provenance, reconstruction, and avoiding destructive summarization. This supports the importance of preserving exact evidence, but also means that "episodic memory plus hierarchy" is not a novelty claim.

### H2MT / hierarchical-memory routing

Top-down hierarchical memory routing overlaps with CHM-v0's original page-routing idea and reinforces the need to test hierarchy against flat retrieval and exact evidence-preserving controls.

## Classical overlap

CHM-v1's current exact search structure is a k-d-tree-style branch-and-bound index over learned address vectors. Exact nearest-neighbor trees, bounding boxes and branch-and-bound search are classical algorithms. They are engineering components, not a new AI architecture contribution by themselves.

## Narrow hypothesis worth testing

The potentially interesting claim is the **joint training objective/interface**:

> Learn the representation used for episodic memory addressing so that exact historical evidence remains retrievable while the geometry becomes efficiently indexable, then integrate that representation with a language model without requiring an oracle or losing end-to-end capability.

The CPU screen only establishes that this is mechanically possible on a synthetic compositional retrieval problem. It does not establish that natural-language representations become indexable, that GPU kernels benefit, or that the approach beats modern memory architectures.

## Falsification priorities

A small matched LM test should kill the idea early if any of the following occur:

1. address geometry collapses or becomes non-indexable on natural language;
2. exact indexed search no longer matches flat retrieval because the runtime needs approximate or stale index state;
3. index rebuild/update cost dominates recurrent or flat retrieval;
4. language NLL or downstream capability is worse at matched compute;
5. state overwrite / stale-belief replacement fails;
6. a simpler sparse/product-key memory baseline matches the same frontier;
7. the memory system needs labels, latent IDs, or oracle boundaries that are unavailable at inference.

## Claim boundary

Until those comparisons exist, describe CHM-v1 only as an **exploratory exact-indexable memory mechanism**. Do not describe it as novel, state of the art, or a breakthrough.
