# Original SI v22 — Policy-relative second-probe value calibration audit (2026-10-16)

**LOCKED VERDICT:** Source/audit integrity gate A PASS. **Conditional cheap-first Q1 underestimation diagnostic gate B PASS** for both full-size original SI hybrid and matched Transformer, all three fresh seeds. **Optimistic public-bin potential correction gate C PASS** via matched Transformer only (hybrid fails its >=.01 mean bar). This is an audit, **NOT a learned better policy or original-hybrid advantage**. Gate D prohibits any claim of architecture superiority/paid scaling. The smaller GRU exhibits the same underestimation, so this fault is NOT original-SI-specific.

## Scope and strict evidence provenance
- Isolated from **exact-head CI-successful original SI v21 PR #1494**, ultimately the original screenshot-inspired full hybrid PR #1459. No changes to `main`, other research tracks (CPW/PGW/CHM/RLT/EHW), paid GPU/Modal/Colab, automatic task or merge.
- Full local frozen protocol `PROTOCOL_V22.md` SHA256 **`5df5aece91888849fafea87f508da6bea3722ef72f92e11271389b7288e4f98e`**; GitHub preregistration `docs/SI_ORIGINAL_SECOND_VALUE_AUDIT_V22_PROTOCOL_2026-10-16.md` committed **BEFORE** confirmations, with unique seeds, settings, bins and hard diagnostic gates.
- Canonical full original 298-line SI hybrid `tam_research/si_original_hybrid_v0.py` Git blob **`575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`** attested at runtime. No changes to original architecture, v19 training, v21 training or task code. New pure EVALUATOR `scripts/si_original_second_value_audit_v22.py` Git blob **`e98d3d6ff914259344d7d373d59bffa1f458967b`**, SHA256 **`12a6bf16c654af96077f5e104155a6f5f69cb2c8ff6b6617dbfd7a4748a1dcfd`**; tests `tests/test_si_original_second_value_audit_v22.py` blob **`8f9e0fceaf3e755b4e454ba0ff0268760395fbef`**, SHA256 **`24686a798fb09c14ea40782b3003d69c6384b94d8200b2cfe0193ebb0e2d173b`**. Both repo/local exact blob-verified, tests **6/6 PASS** pre/post confirmation. Initial test upload had one extra blank line; corrected to exactly match local tested blob, no behavior change, no scientific seed reuse.
- **Excluded development:** 114000 two-step smoke, 114001 smoke, full-training 114010 diagnostic 96 pairs/family×6 MC replicates, exit0, excluded. Full-dev negative/pattern results disclosed in protocol BEFORE confirmations; no threshold/code tuning since.
- **Fresh previously unused engineering CPU seeds 114211,114212,114213** each reserved as a unique directory BEFORE CPU, launched ONCE, **process exit0**, **stderr empty**, full JSON **64782,65283,65022 bytes** respectively, **NO retries/replacements**. Each trains original v21 hybrid (~64.6k backbone), matched Transformer and smaller GRU by exact unchanged original actor160+v19 critic240+v21 Q1 head160+Q0 head160 AdamW steps, batch96, CPU two threads. Audit evaluates **256 twin pairs = 512 opposite-label heldout episodes/family/seed** × 4 families × 2 FORCE-FIRST branches (cheap/strong) × **12 independent possible second measurements per context**. Same heldout worlds for all backbones and policies; independent model-training seed n=3. Only small synthetic CPU engineering evidence, NOT powered replication.

## What is measured and what is NOT
Freeze each model completely. For each heldout observed first-probe context, compare `Q1[buy]-Q1[skip]` with Monte Carlo *expected net reward difference for the very SAME FROZEN final-action classifier* when a second independently sampled sensor is actually purchased, versus skipped, including the real listed second price. This avoids falsely assuming that the original SI/Transformer/GRU terminal decision itself is Bayes-optimal. Evaluator uses latent answer and simulated hypothetical second probes ONLY FOR HELDOUT SCORING; policy/training never accesses unpurchased measurements, hidden labels or these counterfactual targets; weights and world tensors unmodified. All 6 causal tests pass.

**Crucial impossibility/optimism distinction:** a per-world clairvoyant selection using heldout true hidden label and MC samples is UNACHIEVABLE and its potential gain is **ONLY an optimistic upper bound**, never a deployable 'regret'. Binwise evaluation aggregates empirical marginal net gains in **preregistered public predicted-delta bins** [-inf,-.2,-.05,0,.05,.2,inf], then retrospectively chooses whether to buy within each bin. This is STILL an optimistic test-set-derived diagnostic, not an actual calibrated policy or measured prospective intervention. Only a separate training/validation/test experiment can substantiate a reward improvement.

## Frozen three-seed mean second-purchase net value (higher better)

| Family / forced FIRST | SI predicted | SI empirical | Transformer predicted | Transformer empirical | GRU predicted | GRU empirical |
|---|---:|---:|---:|---:|---:|---:|
| **Base / cheap first** | **-0.2186** | **-0.0370** | **-0.2024** | **-0.0298** | -0.2182 | -0.0304 |
| Base / strong first | -0.0720 | -0.0699 | -0.0588 | -0.0705 | -0.0793 | -0.0955 |
| High probe price / cheap first | -0.3615 | -0.1720 | -0.3452 | -0.1647 | -0.3613 | -0.1685 |
| Weak sensors / cheap first | -0.2197 | -0.0505 | -0.2029 | -0.0523 | -0.2221 | -0.0462 |
| **Strong discounted / cheap first** | **-0.0829** | **+0.0848** | **-0.0665** | **+0.0836** | -0.0816 | +0.0955 |
| Strong discounted / strong first | -0.1105 | -0.1008 | -0.0977 | -0.1095 | -0.1181 | -0.1290 |

**Key causal observation:** all three networks consistently underestimate *the net value of buying the remaining strong second probe after a cheap first probe* by 0.15–0.19 across task families, but their estimates after a strong first probe are substantially closer to measured value. This includes GRU, so the effect is not unique to the original SI architecture. On discounted-strong OOD, cheap-first second purchase has a **POSITIVE actual mean net benefit ~+0.085** for hybrid/Transformer while Q1 predicts negative. This is a real evaluation finding for the frozen final classifier, not just a discrepancy with Bayes optimality.

## Preregistered diagnostic gate decisions

**A validity PASS:** original source hash, no hidden/unpurchased Q1 input, 6/6 tests, frozen parameters/world tensors under audit, exit/status/integrity.

**B forced cheap-second undervaluation PASS BOTH full backbones:**
- Hybrid base cheap empirical−predicted **+0.18159**, positive per seed **[+0.207,+0.152,+0.185]**, exceeds frozen +.10, and larger than ABS strong-first bias **0.00213** by >=.05.
- Transformer base cheap **+0.17264**, per seed **[+0.203,+0.137,+0.177]**, exceeds +.10 and ABS strong-first bias **0.01175** by >=.05.
- GRU base cheap ~**+0.1878** corroborates a GENERAL learner/target feature problem, not original SI excellence.
- Mean cheap-first base true net value is still **negative** (hybrid −.0370, Transformer −.0298). Buying second **universally** under base conditions would be incorrect, regardless of pessimism. Need conditionally calibrated decisions.

**C optimistic public-bin correction potential PASS via Transformer only:**
- Hybrid base cheap-first frozen-bin optimistic gap **+0.00893** (below prereg +0.01) and positive **2/3** seeds: **FAIL** individual architecture subgate.
- Transformer base cheap-first optimistic gap **+0.01607** (>=+.01) and positive **2/3** seeds: **PASS** prereg conditional criterion.
- Smaller GRU base cheap-first ~+0.0186 average (control only).
- In **one fresh seed** the gap is zero for both full backbones; test-set-derived bin corrections may NOT generalize or generate real reward. Never claim demonstrated improvement before prospective intervention.

**D original architecture superiority NOT ESTABLISHED:** This is intentionally only a diagnostic with exact unchanged v21 training. Executed BASE normal v21 rewards across three seeds: hybrid [0.6934,0.7427,0.6898] mean **0.7086**; Transformer [0.6674,0.7268,0.7037] mean **0.6993**; GRU [0.7191,0.7264,0.6328] mean ~0.6928. These are seed dependent, and mean hybrid-vs-Transformer differences are not the criterion for this audit. No hybrid relative gate is invoked, and the simpler GRU exhibits identical calibration failure. No claim of general-purpose SI, unique architecture advantage, or efficient GPU inference.

## Next CPU-only hypothesis
Because the v21 trained final decision model is held fixed, the next **prospective** test could use **logged randomized buy/skip second outcomes** to calibrate the *incremental value* conditional on public first-probe quality and confidence (with validation split and no test-set labels), then run heldout independent-seed evaluation against unchanged v21, a uniform-buy policy and a simple linear uplift head. Better option: diagnose calibration of Q1 success heads separately on low-first and high-first trajectories with explicitly balanced logging coverage and cost-normalized labels. A new experiment MUST demonstrate actual net-reward gain rather than retrospective bin potential. No paid GPU scaling, merge, automatic research, or breakthrough authority. Keep PR draft, exact-head CI/review required.
