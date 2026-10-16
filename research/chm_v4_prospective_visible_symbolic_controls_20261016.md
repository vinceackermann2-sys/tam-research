# CHM-v4 prospective visible-text symbolic control and independence audit

**Classification: PROTOCOL ONLY. No heldout generation, scoring, training, GPU, or scientific attempt.** Separate from already-frozen PR #1461 and scorer #1450.

## Source-locked evidence (read-only)
- Benchmark: `tam_research/chm_v4_independent_binding_benchmark_1421.py`, blob `41fd46670a11f9db9615de020c85a80cc4a0db00`.
- Neural pointer: `tam_research/chm_v4_neural_hard_copy_train_1430.py`, blob `78f9de5df6580cb85f557272af5d7f7cffccec55`.
- Frozen grouped scorer: `tam_research/chm_v4_grouped_go_no_go_1450.py`, blob `b7aaa96e01bcd9e684523960a667ce336cf2483e`; exactly 32 positive entities per family, 32 negatives, seven Bonferroni claims. NO RETROFIT.
- Draft #1461: separate NAME-ONLY 176-negative reservation; original first-head CI successful, no independent episodes or scoring.

## Corrected local synthetic contract
The previous 64 handcrafted toy examples passed the visible-text parser but violated an important generator invariant: compared with base rotation, positive cases changed up to 8 code positions, not exactly 2. They must not be described as strictly source-faithful. A new standalone, LOCAL, model-free fixture suite corrected this:
- 256 invented sealed-input examples in 32 **correlated mock groups** (four families x two text templates x four group labels x eight rotations).
- All 256 visible-text symbolic lookups matched fixture labels; 168 positive rotation-vs-base comparisons each changed exactly two CODE positions.
- Query/answer options and eight-code candidate multiset remain invariant per counterfactual group. Negative rotations are one repeated entity, NOT eight independent samples.
- Eight direct/no-match fixture pairs share the same query and options but require opposite presence outcomes. A deliberately novel paraphrase was rejected by the parser.
- The fixtures are NOT actual TRAIN-generator scores, independent holdout evidence, real-language generalization or an architecture breakthrough.

## Prospective falsification controls before future source-locked scores
1. **Visible-only symbolic control:** freeze a parser receiving only exactly CHM-visible `query`, `memory_text`, `answer_options`; no oracle, role, gold address, family or split metadata. Fixed synthetic grammar can be solved by rules, so CHM beating a weak learned baseline alone is insufficient for novel reasoning.
2. **Fair neural Transformer:** identical input, data order, optimizer/steps, parameter and measured train/inference compute budgets, comparable tuning and *equal supervision*. Current neural pointer uses gold first-address and payload-address cross-entropy in TRAIN and a SOFT first hop; answer-only Transformer is not label matched.
3. **Leakage/ablation:** eight-code bags/query invariant under positive rotations, exactly two CODE swaps vs base, negative rotations identical; count by independent entity. Add query-only and bag-only controls. Preserve absent/present matching text and memory structure.
4. **OOD transfer:** separately source-lock novel query paraphrases, aliases, document formats, source layouts, record distractions and authentic text distributions, avoiding a fixed-rule grammar as the only benchmark. Independently preregister unseen examples; don't inspect reserved DEV/TEST early.
5. **Sampling and inference:** freeze entity/document/template provenance, RNG domain/seed, domain overlap, independent sampling units, two replications and joint-power/cost budget prospectively. Existing 32-negative scorer cannot score 176 unchanged. A NEW scorer and multiplicity plan must be preregistered. Single global intersection-union conjunction is not synonymous with seven simultaneous individual claims.
6. **Governance:** isolated branch -> PR -> ORIGINAL first-head CI PASS -> live main re-audit -> expected-head pinned merge -> FIRST merged-main CI PASS; failed/cancelled heads frozen. No paid/GPU/Colab, no heldout, no reused scientific seed `2013161`, no V3 100M Stage-C STOP change.

**No-go/go interpretation:** A TRAIN-only 256-step fit (#1436) proves in-sample capacity at best; failure alone is not terminal. Breakthrough requires independent replication and meaningful matched neural + symbolic control, compute, utility and real-world transfer superiority. Futility requires a preapproved bounded independent experiment sequence and opportunity-cost review. No such verdict is authorized here.
