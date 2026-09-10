# Current long-context / memory research map — 2026-09-10

Purpose: novelty check for CHM-v0 and successors. This is a planning map, not a comprehensive literature review.

## Thesis

The motivating thesis is that an effectively unbounded assistant should not repeatedly process its entire raw history. A scalable system should combine a small high-fidelity working context with conditional access to persistent memory, while retaining exact provenance when lossy semantic state is insufficient.

That thesis is directionally strong, but **not novel by itself**.

## Closest existing directions

| Work | Mechanism most relevant here | Why it matters to CHM |
|---|---|---|
| Infini-attention (2024) | local attention + compressive long-term linear memory | Shows bounded-memory streaming context inside Transformer blocks. |
| Titans (2024/2025) + MIRAS (2025) | neural long-term memory updated at test time; short attention + persistent learned memory | Strong overlap with learned persistent memory / fast-weight framing. |
| TTT-E2E (2025) | meta-learned test-time weight updates compress context into model state | Demonstrates long-context as continual learning rather than only attention. |
| Trellis (2025) | fixed-size KV memory compressed recursively with online updates and forgetting | Directly targets KV growth and bounded memory. |
| Lattice (2025) | low-rank KV-cache compression with orthogonal novelty-preserving updates | Directly addresses interference in compressed memory. |
| Gated Differentiable Working Memory (2026) | learned utility-gated consolidation / write budget | Overlaps conditional writes and compute-aware memory. |
| H2MT (2026) | semantic hierarchy with coarse-to-fine routing | Direct overlap with hierarchical query routing. |
| MemCoT (2026) | iterative stateful memory search; semantic + episodic short-term memories | Overlaps multi-stage query-dependent retrieval. |
| Structured Episodic Event Memory / SEEM (2026) | hierarchical event memory + precise provenance pointers | Overlaps structured episodic memory that keeps exact evidence. |
| E-mem (2026) | hierarchical episodic context reconstruction using uncompressed segments | Important warning against destructive compression. |

## Primary references

- Titans: https://arxiv.org/abs/2501.00663
- Google Titans + MIRAS overview: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/
- Infini-attention: https://arxiv.org/abs/2404.07143
- End-to-End Test-Time Training for Long Context: https://arxiv.org/abs/2512.23675
- H2MT: https://arxiv.org/abs/2605.24930
- Gated Differentiable Working Memory: https://arxiv.org/abs/2601.12906
- MemCoT: https://arxiv.org/abs/2604.08216
- Episodic-Semantic Memory Architecture: https://arxiv.org/abs/2605.17625
- E-mem: https://arxiv.org/abs/2601.21714
- Structured Episodic Event Memory: https://arxiv.org/abs/2601.06411
- Google Trellis publication page: https://research.google/pubs/trellis-learning-to-compress-key-value-memory-in-attention-models/
- Google Lattice publication page: https://research.google/pubs/lattice-learning-to-compress-the-cache-in-the-attention/

## Novelty boundary for a successor

A successor cannot claim novelty from any of the following alone:

- local attention plus long-term memory;
- recurrent state plus attention;
- test-time memory updates;
- compressing old KV state;
- hierarchical semantic retrieval;
- episodic + semantic memory tiers;
- a learned write gate;
- iterative retrieval;
- retaining raw provenance pointers.

A worthwhile new mechanism would need to establish a new quality/compute frontier under matched conditions. In particular, because AERA #838 was dominated by simple retrieval, a successor should treat retrieval as a first-class baseline rather than only comparing against dense attention.

## CHM-v0 lesson

CHM-v0 used a lossy learned page directory: 16 exact events were compressed to 4 learned prototypes, then the query selected only two pages. This reduced candidate comparisons but route recall collapsed with memory length. The result supports a concrete design constraint:

> Semantic compression may be useful for state and prioritization, but arbitrary episodic addresses should not be assumed compressible into a tiny fixed representation without a measurable recall cost.

The next useful research question is therefore not “can we add memory?” It is whether an architecture can preserve exact long-tail evidence while making the *compute spent finding and integrating that evidence* conditional and substantially cheaper than flat retrieval or full attention.
