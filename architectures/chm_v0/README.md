# CHM-v0 — Conditional Hierarchical Memory

Status: **exploratory CPU falsification — FAILED**.

Tracking issue: #846.

## Thesis under test

A useful long-context architecture should not repeatedly attend to every historical token. It should keep a small working context, route queries conditionally into long-term memory, and preserve exact episodic evidence when needed.

CHM-v0 isolates one mechanism-level question: can historical events be grouped into pages, summarized by a small learned semantic directory, and then searched by reading only the top routed pages without materially hurting exact long-distance recall?

## Mechanism

For the exploratory probe:

- historical events are grouped into pages of 16 events;
- each page is represented by 4 learned prototype vectors;
- a query scores the compact page directory;
- only the top 2 pages are treated as active for the exact episodic read;
- an exact event-level scorer operates inside those routed pages;
- a learned recency term must resolve repeated-key overwrites;
- a flat learned retrieval model is the mandatory baseline.

The test deliberately separates **routing/addressability** from full language modeling. A mechanism that cannot preserve a simple overwritten fact as memory grows is not worth integrating into a larger LM.

## Exploratory result

Seed `59317` is exploratory and consumed. Training horizon: 64 events. Each queried key occurs 2–4 times and the correct answer is the latest value.

| Events | CHM accuracy | Flat accuracy | CHM stale error | Flat stale error | CHM route recall | CHM / flat comparisons |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.7666 | 0.9688 | 0.1855 | 0.0488 | 0.8018 | 48 / 64 |
| 256 | 0.6338 | 0.9629 | 0.2383 | 0.0459 | 0.6553 | 96 / 256 |
| 1024 | 0.5566 | 0.9531 | 0.2529 | 0.0557 | 0.5576 | 288 / 1024 |

The directory saves candidate comparisons but loses the target page increasingly often as the memory grows. At 1024 events it uses 28.1% of the flat candidate-comparison count, but answer accuracy is only 55.7% versus 95.3% for flat retrieval.

## Conclusion

**FAIL. Do not scale CHM-v0 or spend Modal/GPU credits on this exact mechanism.**

The important failure mode is an information bottleneck: a few learned semantic prototypes cannot reliably preserve arbitrary rare episodic addresses. Increasing the context length makes the routing error dominate.

A successor must not obtain efficiency by silently erasing exact episodic addressability. It must also beat or complement simple retrieval rather than merely reimplement it.

## Relationship to existing research

The broad idea is not a novelty claim. Contemporary work already explores combinations of local attention, learned/test-time memory, compressed KV memory, gated writes, hierarchy, and episodic/semantic retrieval. Relevant families include Titans/MIRAS, Infini-attention, TTT-E2E, Trellis/Lattice, Gated Differentiable Working Memory, H2MT, MemCoT, SEEM, and E-mem.

Therefore a future CHM successor would need a specific, measurable mechanism advantage rather than the generic claim that models should have multiple memory tiers.

## Successor progression requirements

Before any paid scientific run, a genuinely new successor must pass a fresh CPU/mechanical gate against simple retrieval on all of:

1. repeated-key overwrite and stale-state replacement;
2. rare-fact recall far beyond the training horizon;
3. multi-hop queries requiring more than one memory item;
4. explicit memory-read/bandwidth accounting;
5. ablations showing that any gain comes from the new mechanism rather than extra parameters, an oracle index, or extra training signal.

Any successor gets a new architecture identity, new preregistration, new result/checkpoint namespace, and fresh scientific seeds. CHM-v0's exploratory seed must never be reused as scientific evidence.
