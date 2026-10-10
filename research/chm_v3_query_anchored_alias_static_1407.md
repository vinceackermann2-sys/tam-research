# CHM-v3 #1407 — query-anchored local identifier aliases (TRAIN-only)

**Stage A CPU structural experiment, NOT a learned model or scored comparison.**

## Motivation

The existing frozen model hashes each entire `V3-...` identifier to an embedding bucket. The first balanced 256-step one-shot run (PR #1398, Actions #38066888678; artifact 11675049602) did not demonstrate held-out retrieval; its test development entity index 1 and test entity index 2 are consumed and **must not be scored again**. A subsequent TRAIN-only static audit #1399 showed character-level decomposition increases input tokens from 19,264 to 71,040 (~3.69×) on 256 training cases and may alter FLOPs/latency substantially. Neither result proves an architectural cause.

## Alternative representation

`alias_sealed_input(view)` accepts **only** `SealedModelView(query,memory_text,answer_options)`. It finds the unique visible query identifier, assigns it `ID_ALIAS_Q`, then assigns other distinct model-visible identifiers numbered local aliases in first-memory-appearance order. Exact occurrences of a name in a query, relation or record get the same local alias. All aliases are derived entirely from **ordinary visible text equality**; it never reads evaluator `Fact` roles, oracle-selected pointers, gold answers, family, split, variant or original source seed.

Model-visible lexical tokens remain one-for-one with frozen whole-ID lexical tokens. Every ordinary word, completed-memory source position and atomic `CODE-...` token is unmodified; only the identifier token surface changes. Therefore memory/query boundaries, code target addresses, all copied values, source line indexes and total token counts are preserved. This introduces **a query-equality and local-binding inductive bias**, which a future trained model must get in **all arms**, with a separately tested non-anchored normalization ablation.

A no-match query is absent from memory and still appears as `ID_ALIAS_Q` in the query only. That creates a legitimate explicit absence signal, but it also changes task difficulty and must be accounted for in a fair comparison.

## Frozen zero-GPU validation

- Exactly 256 original BALANCED **TRAIN** cases, 32 groups of 8 counterfactual bindings (8 train entities × 4 families), source `tam_research/chm_v3_balanced_train_schedule_1391.py` and `tam_research/chm_v3_counterfactual_multiset_suite_1373.py`. Never generate original scored development/test data.
- For each positive eight-case group, canonical alias IDs, source positions, decoys, query and global code bag must be identical across counterfactuals; the two differences are precisely the swapped target and decoy code payloads. Negative group variants must be identical and have no query alias appearing in memory.
- Fail closed on missing/ambiguous query identifier, malformed memory source positions, duplicate addresses, alias cardinality overflow, changed code/candidate options or context overflow, SHA256 hash collisions among local alias symbols, and any accidental loss of relation↔record identity.
- Report static token counts, alias-cardinality bounds and paired-group invariants. No model training, optimizer step, inference, old consumed test sets, Modal/GPU/paid science or checkpoint replay.
- First exact-head original CI PASS → re-audited main → expected-head pinned merge → merged-main CI PASS. Failed/cancelled heads permanently frozen, no rerun/amend.

## Next prospective study (not authorized by this PR)

A separate, source-locked CPU comparison could feed the **same alias encoder** to the standard tiny Transformer, pointer+address-CE and pointer-no-CE arms, each trained under balanced schedule and *measured compute budget*. A non-query-anchored alias control should isolate the inductive bias and test that answer prediction changes with swapped facts. Use NEW never-scored entities, selected before any model scoring; original entity 0, development 1 and test 2 from previous experiments are consumed. The current proposal mentions a possible development/test index 3 (subject to source audit and preregistration) but does not construct those held-out cases.

Even a positive tiny synthetic outcome cannot prove general natural-language understanding, fair FLOP/supervision equality, 100M scaling, or a breakthrough. The old 100M Stage-C remains `CHM_V3_100M_DAEC_STAGE_C_STOP`; scientific seed `2013161` consumed forever; historical trained-checkpoint parity proof still missing.
