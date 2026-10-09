# CHM-v3 / CPW: counterfactual memory benchmark — Stage A (#1358)

**Status:** zero-GPU benchmark-design artifact, not a trained model or new scientific result.

The immutable 100M DAEC Stage-C run #1323 remains `CHM_V3_100M_DAEC_STAGE_C_STOP`; scientific seed `2013161` is consumed. The archived v3/v4 cases and old checkpoint are never replayed or rewritten here.

## Why this successor is necessary

The original aligned-v4 benchmark had eight globally balanced answers, but within a family each repeated entity mapped to the same answer across all eight appearances. That permits a hypothetical entity-to-answer shortcut. In the archived #1345 forensic analysis, each of LOCAL, RAW and DAEC had the same correctness set as an always-slot-4 baseline. None of those facts proves the trained models used a shortcut.

## New data-only contract

The additive Python standard-library module `tam_research/chm_v3_counterfactual_memory_suite_1358.py` creates entirely new small symbolic episodes, without GPT-2, pretrained models, CUDA, Modal, or an optimizer. Split namespaces are disjoint: train 8 entities, development 4, test 4. Within **each** (split, entity, positive family), eight cases have **identical query text but different memory contents** and cycle through all eight candidate answer IDs exactly once. Every answer is accessible through completed prior-memory records, with positions < 128, 256, 512 or 1024.

Families are direct entity→answer, overwrite with newest authoritative answer, two-hop entity→document→answer, and no-match negative (query stays comparable to direct/overwrite). The two-hop case includes a wrong-document decoy. Oracle role/subject/document annotations are metadata for dataset validation; they are **not** learned natural-language representations and must not be supplied to a future model as ground-truth labels at inference.

The oracle sorts facts by memory position. A future model evaluator must receive only a separately preregistered encoding of query and memory, NOT `correct_answer_id`, oracle-selected positions, episode counterfactual index, or hidden labels.

## Anti-shortcut result (logical baseline, not model measurement)

For any deterministic function of query text **alone**, the prediction is identical over eight positive counterfactual variants of the same entity/family. Since eight different answers occur exactly once, that function can get **at most one of eight** correct in such a group; the included deterministic query-only control gets exactly 12.5% on direct/overwrite/two-hop separately. No-match is reported separately rather than blended into a pooled accuracy where abstention could mask retrieval failure. The answer-oracle proves addressability, not learnability.

Adversarial CPU tests require per-entity eight-way balancing, query identity and no answer label in query, cross-split identity disjointness, latest overwrite, two-hop link rebinding changing answer, memory record order invariance, negative no-match behavior, duplicate/out-of-memory position rejection, and exact deterministic regeneration.

## What remains unproven

No trained DAEC-A, Transformer, or LM is evaluated on these symbolic episodes. Synthetic records use externally supplied roles and discrete IDs; no claim of natural-language comprehension, 100M scaling, hard-flat trained-checkpoint parity, or CPW superiority is allowed.

A subsequent experiment may compare three **compute- and parameter-matched** controls only under a NEW preregistered model-facing encoding, negative-control calibration, and fresh scientific run/seed authority. Never tune against the frozen #1316 held-out data. No new Modal/GPU allocation, paid compute, new scientific seed, replica, or Stage D authority is conferred by #1358.
