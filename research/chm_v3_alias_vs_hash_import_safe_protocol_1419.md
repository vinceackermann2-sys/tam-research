# CHM-v3 #1419 — frozen import-safe, first-attempt CPU one-shot successor

**Original source is frozen.** Issue #1415 / PR #1418 merged at `1eddb1de3e7543534739aee5e13b512561dd6560`. First dedicated CPU workflow `38073483376`, job `114275520696`, failed at **ModuleNotFoundError: No module named 'scripts'** before any optimizer step or held-out case construction. The failed run, its source and workflow must NEVER be amended/retried. Neither dev3 nor test3 was scored or consumed. Original PR-head CI was PASS; the failure was Python packaging during real entrypoint execution.

## Exact repair

This is a NEW isolated additive `tam_research/chm_v3_alias_vs_hash_one_shot_import_safe_1419.py`. It copies the entire old #1415 scorer's training, scoring, controls, panel, budget and reporting code without changing any models or study settings. The **only execution change** is removal of the import from the nonpackaged `scripts` namespace and inclusion of the same evaluator-only gold/source token-index helper in `tam_research`, where it imports as an installed package.

The new CLI `scripts/chm_v3_alias_vs_hash_one_shot_import_safe_1419.py` explicitly supports `--preflight-only`, which checks package import and source schema **without constructing a training/example case, evaluating a model, or updating an optimizer**. Original repository CI must execute this exact command through a subprocess from repo root, reproducing the runtime that failed #1415, and check the expected import-only marker. Original evaluator-helper parity is tested against historical helper using **TRAIN examples only** (never scored split).

## Frozen experimental recipe (identical to #1415)

- Four arms: original whole-hash tiny decoder, aliased tiny decoder, original whole-hash soft two-hop pointer with train-only source CE, aliased equivalent pointer with identical source CE. Source model/adapter/aliases, sealed representation and counterfactual multiset benchmark **unchanged**, every blob SHA pinned before execution.
- Same deterministic 256 balanced TRAIN examples and optimizer updates/arm, original tiny initial tensors, AdamW LR0.005, wd0, single CPU thread, no CUDA/Modal or paid compute.
- Previously unscored and prospectively frozen development **E003** and test **E003** only, 4 families × 8 variants × 4 arms × 2 splits = **256 complete per-case scored predictions**. Old dev0/dev1 and test0/test2 are consumed and never evaluated again.
- Model forward receives only sealed visible text and options. Record actual chosen candidate/abstention/confidence, model diagnostic first/second visible token indices and evaluator-held gold source indices, per-family positive accuracy/answer-value following, negative false positives and all three 1/8 query-only/layout-only/bag-only analytic controls. Training tokens, parameter counts, four arm times and TRAIN-only preprocessing overhead.
- No optional scoring criteria, model/loss/hyperparameter changes, test-set peeking, trained-checkpoint replay, original evidence reuse or repeated scored attempts.

## CI and one-shot run

Add a NEW path-gated first-main-push GitHub Actions YAML, unique new workflow name, no schedule/manual dispatch, `github.run_attempt == 1`, CPU-only PyTorch, strict 35-minute cap. Fail-closed Git blob SHA checks for original source and new scorer/script/tests/protocol before any training. Full validated JSON and 256-row artifact retention 90 days. **The original PR-head CI must PASS on attempt 1**, re-audit live main, expected-head-pinned merge, verify first merged-main CI separately and first dedicated workflow one time only; freeze failed/cancelled attempts with no retry/amend.

Only a synthetic **input-representation ablation**, not an architecture or general language/100M breakthrough. Local aliasing provides model-visible equality as an explicit inductive bias. One held-out entity group per family/split means no credible confidence intervals; tiny pointer extra gold address labels/FLOPs, and soft-conditioned hops, preclude matched scientific claims. Original #1323 100M Stage-C `CHM_V3_100M_DAEC_STAGE_C_STOP` remains; seed `2013161` consumed forever, missing original trained-checkpoint parity unresolved. No GPU/Modal, scientific seed, Stage D or breakthrough authority.
