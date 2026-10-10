# CHM-v3 #1399 — static entity-identifier representation audit

**Purpose:** Diagnose the *representation* confound visible in frozen CHM-v3 #1377 tiny model without running or retraining it. This is not scored experimentation. All experiments in this proposal operate exclusively over #1391's **256 TRAIN-only episodes** and five invented synthetic identifier strings. Never read the used #1395 heldout split examples (dev entity 1, test entity 2), their predictions or model checkpoints.

## Known, not hypotheses
- Frozen tiny tokenizer has regex `V3-[A-Za-z0-9-]+`, so the complete identifier is one token, SHA256 mapped to an 8192-entry embedding table; new names generally have *untrained* rows, with possible hash collisions.
- That fact does **not** establish cause: a novel identifier repeated in the same prompt and completed memory may still be addressable by the network through within-sample matching.
- #1395's balanced CPU scored run 38066888678 (artifact 11675049602) demonstrated no robust test entity retrieval benefit. No new evaluation of those consumed panels may occur here.

## What this audit measures, without model execution
- Original `encode_sealed_view` parity on every TRAIN episode (original token IDs, code-token indexes, memory lengths and source anchors).
- Candidate representation: `ID_OPEN`, one tagged `IDCHAR:<character>` piece per character of the exact visible identifier, `ID_CLOSE`. The entire *identifier*, including the split namespace, remains injectively encoded; namespace is **not** removed, so cross-split entity aliases cannot be silently conflated. The hash stays identical for all ordinary tokens and candidate pieces. No input/model architecture is changed; this is a static proposal.
- Completed-memory source `[0000]` anchors and `CODE-18001` answer values remain indivisible and are checked at their new positions. Nothing crosses the memory/query boundary; shared identifier occurrences produce identical piece sequences.
- Audit reports training-only token budget changes, max lengths, count above frozen `MAX_TOKENS=384`, distinct TRAIN identifier hashes/collisions and synthetic *lexical* piece coverage of invented identifiers. If expanded tokenization exceeds budget, report overflow: **never truncate**, reinterpret, or claim an implementable model.
- The names `V3-train-E900`, `V3-train-D900`, `V3-hypothetical-E900`, `V3-hypothetical-D900`, `V3-hypothetical-X900-D0` are not benchmark development/test examples. Their purpose is only to probe unseen lexical surfaces and re-used characters.

## Reproducible control and ceilings
FROZEN reference model blob `ffe14b0701e18493a2bf9b45a1238b5acd5e3eab`; #1391 scheduler blob `6d40dafc7a487cfc8d5a39d5067ff00190e60181`; #1365 sealed model view blob `5bce89954d7a4fd788d1d1bdd972abf744399715`; #1373 benchmark blob `1b71eede754f7396ad2e7284693d6d5c1b8273b4`. Source-locked tests compare frozen encoder outputs at every TRAIN case.

This analysis is CPU/stdlib+frozen-source only, **0 GPU, 0 optimization, 0 learned-model inference, 0 scientific seeds**, 0 heldout scoring. Historical DAEC 100M Stage-C remains `CHM_V3_100M_DAEC_STAGE_C_STOP`, seed `2013161` forever consumed. Only after this structural report should a separate next-generation tokenizer+fresh-entity experiment be *proposed*, not automatically launched. Any future scored model needs new immutable seed/split, budget, parameter/FLOP/supervision fairness and controls before training.
