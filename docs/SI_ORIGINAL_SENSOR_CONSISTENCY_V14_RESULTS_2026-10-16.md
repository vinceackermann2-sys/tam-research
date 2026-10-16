# SI Original v14 — independent sensor consistency and reference repair (CPU results, 2026-10-16)

**VERDICT: FAIL preregistered robustness gate, FAIL original hybrid superiority gate.** A third physically observed calibration channel helps when the original trusted reference sensor fails late in calibration. But simply comparing two reference-like channels **does not identify which one is wrong**. The same diagnostic misrepairs a failing third channel, does not handle two flips of the target, and adds an information fee/compute expense. This is not general epistemic uncertainty, autonomous world discovery, SI or a breakthrough.

## Frozen provenance and compute audit
- Isolated branch `experiment/si-original-sensor-consistency-v14-20261016` stacked on CI-successful v13 original-SI PR #1486, ultimately original screenshot-inspired SI PR #1459. No main merge, other research changes, automation, or GPU/Modal/Colab expenditure.
- Preregistration `docs/SI_ORIGINAL_SENSOR_CONSISTENCY_V14_PROTOCOL_2026-10-16.md` **committed BEFORE new confirmation seeds**. Detailed local frozen protocol SHA256 `25ad3b824eda0528620351f007ed6d9ab9a2a200dac79b73756caa47cb528bae`. The initial GitHub source copy differed from the frozen local file only in positions of empty lines. It was byte-aligned to **exact Git blob `cbb42495f203d4ab5269ead686dde9241a872488`** prior to finishing any reported science, with **no semantic code change**; SHA256 `6d3076c0875864d8cd8219e8f386d839660f10144ffa110631bbfb9ee18a2bea`. Test Git blob `1a981b1b55e86354efc9ee013f20bb6c2ebf8e88`, both GitHub/local verified identical; **7/7 targeted CPU tests passed**.
- Canonical full original 298-line module `tam_research/si_original_hybrid_v0.py` blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`; unchanged v12 trainer blob `fc160944420fe35ec9cf2f626def07c75a86656a`, v13 reference blob `51e76032747bcdf90f755164216a923f310dbed3`.
- Development-only 106000 smoke, timed-out **106010 and 106011** before complete JSON, short dev 106012: all excluded, never reattempted. **Fresh preregistered confirmatory CPU seeds `106211,106212,106213`** each launched once, no retries or replacements; each produced complete parseable 48KB JSON and empty stderr. OS exit status explicitly captured as **0** for 106212/106213; detached 106211 completed JSON/empty stderr and terminated, but separate OS exit status was not captured (do not represent it as 0). Three-seed **engineering** not high-powered independent scientific replication.
- Each seed trains original hybrid (64,614 parameters), nominally matched Transformer (64,637), smaller GRU (10,283, **not** parameter parity), each in separate raw and belief form using **unchanged** v12 logged-feedback training, 180 actor +200 critic-only updates × batch96 × 2 CPU threads. Paired evaluation uses **512 independent worlds/family/seed**, 8 families = 1,536 pooled worlds per family with only **3 independent training seeds**.
- Synthetic third sensor is independently observed `anchor_cal=theta_cal*1.8+N(0,1)`; hidden theta only generates measurement inside simulator. Neural weights never train on anchor. Causal flag `mean(y_cal*anchor_cal/(mu_ref*1.8))<0.5` permits substitution `y_repaired=anchor_cal*(mu_ref/1.8)`, not a learned fault model. **Every anchor-assisted action pays synthetic cost 0.025 per episode**, even if not substituted, raw/bel/v13 policies pay no extra fee. Controls include **always substitute**, **same-frequency shuffled substitute** (offline evaluator-only grouping by family), unchanged v13 fallback, both detectors combined. Some assisted controllers load **two full 64k models**; added independent measurement and Bayesian selection FLOPs are not parity-matched to single model.

## Exact three-seed mean net reward, higher better

| Family | Hybrid raw | Hybrid belief | Hybrid consistency repair | Hybrid shuffled | Transformer raw | Transformer belief | Transformer consistency repair | Transformer shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stationary base | 0.7707 | **0.7847** | 0.7641 | 0.7603 | 0.7694 | **0.7830** | 0.7606 | 0.7591 |
| One known target flip | 0.7601 | **0.7930** | 0.7687 | 0.7678 | 0.7593 | **0.7965** | 0.7728 | 0.7717 |
| Two target flips, OOD | **0.7165** | 0.6329 | 0.6116 | 0.6106 | **0.7361** | 0.6202 | 0.5976 | 0.5959 |
| Reference sign reversal late | 0.6915 | 0.5754 | **0.7524** | 0.7469 | 0.7189 | 0.5574 | **0.7527** | 0.7442 |
| Reference sign reversal early | 0.7770 | **0.7978** | 0.7791 | 0.7799 | 0.7751 | **0.7946** | 0.7755 | 0.7783 |
| Reference high-noise dropout | 0.7438 | 0.6963 | **0.7587** | 0.7322 | 0.7419 | 0.6833 | **0.7521** | 0.7236 |
| **Independent anchor fails**, OOD | 0.7622 | **0.7925** | 0.5602 | 0.5583 | 0.7732 | **0.7796** | 0.5490 | 0.5448 |
| Weak calibration | 0.7497 | 0.7507 | 0.7264 | 0.7253 | 0.7610 | 0.7498 | 0.7250 | 0.7249 |

Additional strong always-substitute comparator on late reference reversal: **hybrid 0.7615**, Transformer **0.7624**, both **BETTER** than selective repair ~0.752 after the same 0.025 sensor fee. This prevents falsely attributing the gain to sophisticated detection when merely using a new sensor suffices.

**Detector flag frequencies** (same across architectures): stationary **0.074**, one-target-flip 0.078, double target flip 0.059, late reference reversal **0.926**, early reference reversal **0.919**, high-noise dropout **0.519**, independent anchor reversal **0.918**, weak calibration 0.074. It detects two-sensor disagreement but *cannot say which side is failing*. Conditional Q and true action labels are never used to select repair at inference.

## Prespecified gates: FAIL, not post-hoc selection

**Gate A robust reference repair FAIL** for both full architectures despite strong improvement on one family:
- Late reference selective-minus-belief: hybrid **+0.1770**, Transformer **+0.1953** (required +0.05 PASS), but detector minus shuffled only hybrid **+0.0055**, Transformer **+0.0085** (required +0.02 FAIL).
- Early reference repair minus belief: hybrid **−0.0187**, Transformer **−0.0191** (required >=+0.05 FAIL), and minus shuffled ~0 or negative (FAIL).
- Familiar stationary repair minus belief: hybrid **−0.0206**, Transformer **−0.0224** (max allowed harm 0.015 FAIL).
- Two target flips: repair minus standalone raw hybrid **−0.1049**, Transformer **−0.1385** (max allowed harm 0.02 FAIL).
- New anchor reverses: repair minus standalone raw hybrid **−0.2020**, Transformer **−0.2242** (max allowed harm 0.02 **FAIL decisively**).
- Reference reversal detection >=0.80 and stationary false flags <0.15 PASSED; this is an insufficient *detector* endpoint compared to realized reward and safe fault identity.

**Gate B original hybrid superiority FAIL**: normal selective hybrid 0.7641 vs Transformer 0.7606, difference **+0.0035**, below required +0.015; late reference selective hybrid **0.7524** vs Transformer **0.7527** (hybrid loses). Hybrid two-network training ~**19.99 CPU seconds** vs Transformer ~**12.62 seconds**, ratio **1.58x** exceeding prereg maximum 1.5x. GRU remains a strong smaller no-novelty check.

**Gate C generalized model-class recognition FAIL**: anchor-reversal seen only at test time is observationally mistaken for nominal reference failure. Even with third-sensor redundancy, channel disagreement alone cannot determine faulty channel identity. This test is **not** proof that no possible three-sensor algorithm could isolate faults; it demonstrates this particular fixed rule's assumptions and failure. With only two originally observed channels, which one failed can be intrinsically ambiguous without independent information or stronger priors.

## Next bounded CPU research option

Focus on **per-sensor reliability estimation** treating all three observed channels as potentially faulty, with a transparent classical redundancy/robust majority filter as control, proper paired heldout multi-fault and simultaneous failure cases, and physical observation/inference cost accounting. Only pursue if its added complexity can beat simple raw and always-use-alternate policies **and** comparable Transformer/GRU under true net reward. No paid GPU, main merge, original SI advantage, broad breakthrough, continual/multimodal ability or efficiency demonstrated. Preserve all negative evidence and require exact-head GitHub CI.
