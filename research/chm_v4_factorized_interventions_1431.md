# CHM-v4 #1431 — factorized physical source, hard copy and answer gating

**TRAIN-only mechanical intervention analysis.** This stage does not train a model, score v4 reserved development/test entities, or claim an AI architecture improves.

## Frozen lineage

- #1421 independent V4 benchmark: merged PR #1425, source `tam_research/chm_v4_independent_binding_benchmark_1421.py`, blob `41fd46670a11f9db9615de020c85a80cc4a0db00`. Future DEV16/TEST32 manifests **unopened**; old v3 panels consumed forever.
- #1427 visible-text source address→CODE hand-rule: PR #1428 merge `bc31d646e5cf153d17a1fb76ae9632f6142eb009`, source `tam_research/chm_v4_visible_hard_copy_train_1427.py` blob `32439754d923c7d162da148db811ba9dad6e06aa`. Source original exact-head CI PASS; merged-main CI `38080972393` PASS first attempt.
- Original CHM-v3 100M `CHM_V3_100M_DAEC_STAGE_C_STOP`; seed `2013161` consumed permanently.

## Three separate stages

1. **Proposed first physical memory source**: the only input to the new hard-reader beyond `SealedInput` is a proposed `[dddd]` address. The proposed address is not a hidden target label, and can be intentionally wrong. The visible-only deterministic `read_visible_text(view)` upper bound supplies the *natural* proposal for the engineering audit; future learned architectures would supply their own predicted source.
2. **Hard address→payload read**: `copy_from_proposed_source(view,address)` retrieves the exact visible source line, follows a visible document relation if that line represents one, and selects its latest visible document payload. `copy_value_at_explicit_payload` independently copies exactly a nominated visible candidate CODE physical position. No hidden role, gold label, oracle pointer or answer head participates.
3. **Answer/abstention gate**: `gate_copied_value` emits a retrieved candidate if present, else abstains. Explicit **forced-abstain** and **forced-default** policies illustrate what happens when gating fails, not recommended inference methods.

### Deliberate counterfactual interventions

All interventions are within the **2,048 TRAIN-only cases**, consisting of 64 independent TRAIN entities × 4 task families × 8 code swaps. They use only completed-memory text and the *non-learned visible baseline*. The evaluator oracle is consulted only after the composed prediction.

- For each of 1,536 positive TRAIN examples, send the reader to a different valid visible candidate-code source: by code-multiset uniqueness the answer must change. Also deliberately nominate a different payload source (wrong code) or force abstention despite correct copy. These separate address error, payload error and answer-gate error.
- For each of 512 two-hop examples, use the **older** literal relation source instead of the current relation: it should lead to a different decoy document/code, demonstrating the necessity of latest-link routing.
- For each of 512 overwrite examples, route to the **stale noncandidate CODE-19001** source: it must reject rather than silently output an invalid code.
- For each of 512 no-match examples, natural source absence must produce abstention; forcing CODE-18001 without any retrieved source produces a false positive.

**Expected perfect natural performance is engineered by the template-specific parser** and should NOT be presented as trained generalization. The intervention success counts measure logical dependency and detect shortcut/implementation errors, not the ability of a neural network to learn the function. Counterfactual rotations within one independent entity are correlated. No statistical confidence interval is appropriate.

## Strict execution

No GPU/Modal, no 100M or paid science, no PR rewrite to fix failed CI. Additive branch→original exact-head CI first PASS→re-audited main→expected-head pinned merge→first merged-main CI PASS. Future model experiment must separately preregister and measure a trainable address selector, a hard-readable value extractor, and abstention gate against a budget/source-supervision-matched Transformer across independent V4 entities before TEST is opened.
