# CHM-v4 #1421 — independently grouped counterfactual memory benchmark

**Stage A: DATA ONLY, zero GPU, no model training or scoring.** Fresh branch `research/chm-v4-independent-entity-benchmark-1421-v1`; no modifications to original CPW/CHM-v3 checkpoints, old benchmark, other PGW/RLT/reduced-attention tracks or Modal workflow. Original 100M Stage-C `CHM_V3_100M_DAEC_STAGE_C_STOP` and consumed seed `2013161` remain immutable.

## Why a new benchmark?

The first valid alias-vs-hash 256-step CPU one-shot #1419 succeeded on original source run `38074130939` (artifact `11677472833`): all four tiny arms scored **3/24–4/24** on positive test memory tasks (roughly 1/8 chance). Aliased pointer found the first direct-retrieval source address 8/8 but answered only 1/8; on overwrite and two-hop its first source address matches were 0/8. Aliasing improved no-match detection; it did not demonstrate memory binding or fair architecture superiority. These tests used **one entity group per task family**, not an adequate independent sample. All v3 development E000/E001/E003 and test E000/E002/E003 cases scored in historical studies remain consumed and MUST NOT be tuned upon or rerun.

## NEW source and declarative heldout manifest

`tam_research/chm_v4_counterfactual_independent_1421.py` contains **fresh namespace `V4-`**, new deterministic layout RNG salt, new rendered language templates, and a NEW split declaration:

| Split | Independent entities per family | Eight-way rotations | Total across 4 families |
|---|---:|---:|---:|
| TRAIN | 64 | 8 each | 2,048 |
| DEVELOPMENT (reserved) | 16 | 8 each | 512 |
| TEST (reserved) | 32 | 8 each | 1,024 |

The **unit of independent analysis is the ENTITY within a FAMILY**, not eight correlated counterfactual rotations. We use `(family, entity)` as the cluster and report family-level entity aggregation in future studies. Candidate answer set remains eight codes 18001–18008 to permit familiar controls, but identities, layouts, templates, sampling and train/test reservations are newly generated; no model checkpoint replay.

`declarative_heldout_manifest()` returns *only the immutable declaration* and its SHA-256 over canonical JSON; it **does not generate development/test examples, and is NOT a checksum of heldout payloads**. A separately preregistered, source-bound first scoring workflow must later compute and publish an exact payload digest without feeding it to models.

### The four task families

**Direct:** one authoritative `(entity, CODE)` row, seven irrelevant candidate rows. **Overwrite:** older same-entity row stores obsolete non-candidate `CODE-19999`, newer row stores current candidate; seven candidate decoys ensure every answer code appears exactly once. **Two-hop with relation rebinding:** one older entity→prior-document relation, one newer entity→current-document relation, the current document's record stores the target; prior doc and six other records are decoys, all eight candidates present exactly once. **No-match:** query entity does not appear as an address in memory; eight unrelated records each contain one candidate code, so the mere presence of an answer choice is never sufficient.

For every positive `(split, family, entity)`, counterfactual variants 0–7 contain all eight correct answers **exactly once**. All layouts/positions/subjects/documents/query texts/render style are fixed within a group; target code exchanges with exactly ONE decoy so the global candidate-code *multiset is identical*. A model restricted to query-only, code-redacted layout, or bag-of-code-values therefore has maximum correct accuracy **1/8 on each full matched group**, regardless of its preference for code slots. Always-abstain is measured separately; its accuracy is zero for positive groups and 8/8 for no-match.

Model-visible data are **only** `SealedModelView(query,memory_text,answer_options)`. Structured `V4Fact`, source roles, document/reference metadata, family, gold answer, variant, authoritative physical positions and split ID are evaluator-only and may NOT be passed to model.forward(). The source oracle independently selects the latest entity row or latest entity relation followed by linked record, using actual completed-memory physical positions; it does not use the supplied gold label.

### CI safety and shortcut gates

CI constructs **only TRAIN** examples: all 256 eight-way `(family, entity)` groups = 2,048 cases. Unit tests check source uniqueness/position bounds, real overwrite timestamps, two-hop latest-relation rebinding, target/decoy value swaps, exact code multiset, query/redacted-layout equality, 3 renderer styles, answer balance (64 copies of each answer per positive family), at least 16 distinct target addresses per positive family, target/decoy identity preservation and no-match absent query. Deliberate malformed/duplicate/adversarial cases must fail closed.

Tests inspect split declarations for development and test **without constructing either heldout split**, and monkeypatch generation to assert `train_only_shortcut_audit()` does not access them. Neither a full heldout suite, learned model, checkpoint, optimizer, GPU nor Modal is invoked. No v4 TEST case is scored, read or used to choose a model.

## Known constraints and next gate

- Source templates are controlled synthetic English, not natural language understanding; `V4-` identifiers require a separately frozen model-visible encoder because prior v3 tokenizer/alias parser understands `V3-` names only. Do NOT silently reuse or tune old tokenizer without a new preregistered branch.
- Semantic kind labels and source positions exist only on evaluator side; a future hard address→value→answer implementation must derive them from text. A deterministic visible-text parser is a **rule/oracle reference**, not evidence of learned reasoning.
- Physical positions are randomized by entity and family, while counterfactual variants share layout. Overwrite uses a deliberately distinct stale non-candidate code; this means an explicit `CODE-19999` marker helps identify an obsolete record, and further adversarial controls may be needed before any scientific claim.
- The fixed candidate-code multiset is an analytic 12.5% blind ceiling **within each positive matched group**. It is not a statistical confidence interval or proof that every shortcut is eliminated.
- Future model comparisons require exact FLOPs, training labels, grouped CIs by independent entity, hard source-address and value extraction diagnostics, and preregistered no-match thresholds; a simple `8-way accuracy >1/8` without entity-level uncertainty is insufficient.
- A next **separate** data/CPU TRAIN-only hard-copy diagnostic PR may test extraction through visible text and classify source selection versus answer/abstention gating. A future first-run heldout evaluator must be uniquely preregistered on another branch, head CI PASS, expected-head pinned merge, merged CI pass and source-bound one-shot CPU workflow. No TEST peeking in engineering CI.
- Any zero-GPU toy improvement cannot reclassify the historical 100M scientific STOP, consume old 100M seeds, authorize 100M scaling or establish a CPW/Transformer breakthrough.
