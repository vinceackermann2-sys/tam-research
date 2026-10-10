# CHM-v4 #1430 — Neural address logits, hard value copying and abstention (Stage-A CPU)

**Scope: one TRAIN-only optimizer step + structural/gradient tests.** No development/test generation, training sweep, model score, GPU/Modal, new scientific attempt, or breakthrough claim.

## Frozen research input and prior evidence

CHM-v4 dataset: `tam_research/chm_v4_independent_binding_benchmark_1421.py` from merged [PR #1425](https://github.com/vinceackermann2-sys/tam-research/pull/1425), exact first CI `38075449240` PASS (3224 tests), merged-main CI `38075802798` PASS. This source reserves **64 TRAIN entities / 16 DEVELOPMENT / 32 TEST** per family, with 8 counterfactuals per entity, train groups distinct from never-scored development/test groups.

Visible-only deterministic benchmark read: [PR #1428](https://github.com/vinceackermann2-sys/tam-research/pull/1428), exact source head `48ce132b080ac6ffbce541158a395062ce10cbd0`, CI `38076310597` PASS 3278/1 skipped, pinned merge `bc31d646e5cf153d17a1fb76ae9632f6142eb009`, merged-main CI `38080972393` PASS first attempt. The handcrafted reader exactly recovered TRAIN cases, confirming parseability of the benchmark, **not evidence of learned capability**.

Earlier v3 aliased-pointer TEST first-address hit 8/8 direct but answer hit only 1/8, and other positive families were around chance 1/8; its soft attention score was not a hard memory read. All old V3 dev/test holdouts are consumed and never reused.

## New neural front-end and model

`tam_research/chm_v4_neural_hard_copy_train_1430.py` is NEW; original v3 frozen `CausalBackbone` parameters/attention and V4 dataset are NOT modified. Its V4-specific tokenizer accepts **only** the dataclass `SealedInput(query,memory_text,answer_options)`: literal V4 entity/record tokens, atomic CODE-n, source `[NNNN]` anchors, memory/question boundaries. Exactly one of each eight candidate code must appear. The obsolete noncandidate marker is allowed but excluded from the eight output CODE candidates. No hidden Episode/role/source labels, family/split/variant/gold or structured oracle enters forward.

`TinyV4HardCopyPointer` contains a conventional tiny causal backbone, trainable first query/key line-address heads, a first-soft-attention-conditioned second CODE candidate-address head, and a distinct two-class copy/abstain head. **The first hop uses each line-END causal state, not blind beginning-of-line anchors**, so the state has actually processed that line's subject/code content. Line ids are visibly serialized source addresses, not evaluator labels.

At inference `hard_copy_prediction(output)` picks the model's **own argmax candidate CODE position** and copies that literal candidate ID from visible text if its learned gate predicts present; otherwise it abstains. No gold value is used to choose the output. First and final chosen physical source positions are logged separately for later correctness evaluation.

**Crucial limitation:** the second hop is conditioned on a **SOFT MIXTURE** of first-hop line embeddings, not a genuinely hard selected first address. Hard copying only occurs at the OUTPUT. This must never be called a real hard two-stage read until a separately designed hard-conditioned second hop is implemented and audited.

`train_only_supervised_loss` is evaluator-side and receives a TRAIN episode only **after** model.forward has seen the sealed visible text. It calculates CE on copy/abstain, plus 0.4 × first source CE and 0.4 × candidate-code source CE for positives. Missing entities supervise gate only. Extra source-address gold labels and FLOPs mean this approach is **NOT supervision/computation matched** to an ordinary Transformer baseline.

## Static and bounded first-attempt CI

Tests cover TRAIN split only:
- V4 sealed input, eight-code bag and physical address provenance; no stale marker selected as an output candidate.
- Hard CODE copy from the model-selected visible candidate (or abstention) independent of oracle, with synthetic logits to verify index→value path.
- Model rejects raw structured Episode; forward never reads hidden metadata. No dev/test generators, GPU, Modal or old 100M model checkpoint.
- Finite loss / gate+first+second address shapes and source labels on representative TRAIN direct/overwrite/two_hop/no_match episodes.
- **Exactly one** TRAIN-only optimizer step/one positive training example total; nonzero finite first query/key, second address projection, gate head and embedding gradients. Explicitly reject steps=0,2,256 and invalid types. This is a gradient smoke and must not be interpreted as trained performance.
- AST executable-call prohibition instead of raw string/docstring match, avoiding the separate frozen #1426 false-positive CI failure.

Only original first-attempt exact-head full CI PASS qualifies this source for expected-head pinned merge after fresh live main audit; independently verify merged-main first CI. Failed/cancelled heads are frozen and never amended/rerun.

## Future science gate (NOT authorized here)

A separately preregistered, source-locked first-attempt CPU experiment could train matched Transformer and hard-copy/soft-copy models using the same NEW V4 tokenization and train-only data, explicitly match compute and address supervision (or stratify by budgets), then score genuinely unseen 16-development/32-test independent entity clusters per positive task. Separate gating FPR uncertainty must be grouped by entity; with 32 no-match test entities, even 0 observed FP has one-sided 95% upper bound ~8.94%, so cannot meet upper-bound <5%. Report first physical address, final candidate position, copied-value correctness, abstention independently and all case predictions. Success does not establish general-language/100M CPW or Transformer superiority without replication.

Historical CHM-v3 Stage-C `CHM_V3_100M_DAEC_STAGE_C_STOP`, scientific seed `2013161` consumed permanently, trained checkpoint scorer parity missing. No GPU/Modal credits, 100M scaling, new science seed or breakthrough claim.
