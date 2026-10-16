# SI Original v8 — source-attested episodic belief learning: CPU results (2026-10-16)

**Decision: NO-GO for unchanged original SI hybrid as a paid-compute / breakthrough candidate.** The hybrid demonstrably uses causal calibration history, but it fails the preregistered quality gate versus a parameter-matched Transformer, a smaller GRU, and a known-model Bayesian reference. This is useful targeted negative evidence; it is **not** proof that the broader original SI architecture cannot work.

## Provenance, branch and scientific limits

- Branch `experiment/si-original-belief-transfer-v8-20261016` is isolated, stacked on v7 benchmark PR #1479 -> original hybrid PR #1459; no main or other projects modified, no merge, automation, Modal, Colab or paid GPU. Original 298-line hybrid model module locally **byte-identical** to canonical GitHub blob **`575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`**. Full original 3-layer attention + recurrence + sparse MoE + refinement is used unchanged. Original fast adapter and external episodic memory store are instantiated but not activated; v4 delta-rule memory is not included here. This test isolates the original backbone's in-context inference.
- Full executed `scripts/si_original_belief_transfer_v8.py` byte-identical to GitHub blob **`233dff4d660da142fbb57a91cd06b0c4ec17a0d0`**, SHA256 `b3d0350bfbde917c8ce61060a8e22b4d9b680dbe1196ba97594859faf9ee413d`. `tests/test_si_original_belief_transfer_v8.py` Git blob `a72624c17f6dc9eb09b9b3695498c1a7cf740551`, **5 local tests passed**. Protocol in `docs/SI_ORIGINAL_BELIEF_TRANSFER_V8_PROTOCOL_2026-10-16.md` committed before confirmation.
- **CPU attempt accounting:** Frozen confirmation seeds were 100211/100212/100213. Seed 100211 started but command timed out at outer 120s limit, leaving **0-byte output**, no surviving worker. It is **consumed, not rerun, not evidence**. Fresh replacement 100214 registered in `docs/SI_ORIGINAL_BELIEF_TRANSFER_V8_TIMEOUT_ADDENDUM_2026-10-16.md` **before execution**. Completed seeds **100212,100213,100214**, each finished once with exit 0 and complete JSON (7,614–7,670 bytes), no errors, no retries. This is engineering evidence with a documented seed substitution, **not unamended scientific replication**.
- Same training samples/seeds, 240 AdamW steps × batch64 per model/seed, LR .001, weight decay .01, clip-grad norm1, 2 CPU threads. Train only Gaussian `base` sensor simulator. Probe-mask sampling uses a seed derived independently of architecture initialization so all models receive the same optional information during training.
- Each model receives six calibration trials with noisy target and trusted-reference sensor observations and per-trial known sensor strength, followed by one new hidden-bit query. The controller may pay for a trusted additional measurement before acting. Reward = correct final decision minus probe fee. Hidden sensor polarity z is **learned from calibration history**; oracle z and true query outcome are NEVER model inputs. However, synthetic training **supervises** z auxiliary label, query outcome label, and optimal Bayes-probe decision; not unsupervised latent discovery, RL or multi-step planning. This is a small known-family task, not a full SI benchmark.
- **Matched neural architectures:** OriginalSIHybrid **64,614 params** vs TransformerControl **64,637 params**, common tokenization, head usage, optimizer and supervision. An additional GRU **10,283 params** is a smaller, non-parameter-matched reference. Bayesian controller is explicitly given the correct likelihood model; it is a known-model reference, not an equal-knowledge learned architecture.

## Three-seed mean metrics

**Realized reward, higher better:**

| Model | Normal | OOD inverted prior p=0.8 | OOD noisier reference sensor | OOD weak calibration |
|---|---:|---:|---:|---:|
| Original SI hybrid | **0.795203** | 0.791985 | 0.754255 | 0.726525 |
| Matched Transformer | **0.815619** | 0.816706 | 0.776703 | 0.747946 |
| Smaller GRU | **0.829782** | 0.826097 | 0.778017 | 0.775892 |
| Exact Bayes (known model) | **0.850919** | 0.851607 | 0.829793 | 0.801756 |
| Bayes without calibration | 0.759288 | 0.757613 | 0.761747 | 0.758759 |
| Always probe (with calibration) | 0.788869 | 0.786991 | 0.780994 | 0.771861 |
| Never probe (with calibration) | 0.772095 | 0.775513 | 0.732340 | 0.672770 |

**Base-world per-seed paired reward:**

| Seed | Hybrid | Transformer | GRU | Hybrid - Transformer |
|---:|---:|---:|---:|---:|
| 100212 | 0.808030 | 0.805200 | 0.828520 | +0.002830 |
| 100213 | 0.798580 | 0.825300 | 0.826850 | -0.026720 |
| 100214 | 0.779000 | 0.816360 | 0.833980 | -0.037360 |

**Calibration and acquisition metrics (3-seed mean):**

| Model | Normal hidden-polarity Brier (lower better) | Polarity accuracy | Probe rate | OOD weak-calibration Brier | Mean CPU train seconds |
|---|---:|---:|---:|---:|---:|
| Hybrid | **0.087355** | 0.879883 | 0.478434 | 0.203256 | 11.815 |
| Transformer | **0.078533** | 0.895874 | 0.495972 | 0.196144 | 10.535 |
| GRU | **0.055364** | 0.926799 | 0.494344 | 0.181018 | 2.701 |
| Bayes | **0.039589** | Not applicable here | 0.483439 | 0.156548 | Solver, no training |

OOD inverted-prior Brier: hybrid 0.080432, Transformer 0.072957, GRU 0.061974, Bayes 0.039431. The benchmark uses calibrated posterior z; while the Bayes reference assumes 50/50 training prior even in shifted-prior evaluation, its six calibration pairs mostly resolve z.

## Causal calibration-history ablation

At test only, zero target and reference calibration values without retraining (an out-of-distribution input ablation, **not** a perfectly matched trial):
- Hybrid base reward **0.795203 -> 0.637090**
- Transformer **0.815619 -> 0.650590**
- GRU **0.829782 -> 0.674390**

All model types rely on calibration experience. The observed dependency cannot be uniquely credited to original SI recurrence, MoE, refinement or its dormant episodic memory.

## Gate decision (preregistered, no post-hoc threshold edits)

- **Task valid:** known-model Bayes normal reward 0.850919 beats no-calibration-Bayes 0.759288, always 0.788869 and never 0.772095 by >=0.03; the known-model controller purchases information in ~48% of trials.
- **Original SI hybrid FAIL:** hybrid mean normal reward 0.795203 is **0.020416 below** Transformer; inverted-prior reward 0.791985 is **0.024721 below** Transformer; only **1/3** paired seed wins normal and same poor aggregate elsewhere. Hybrid base Brier 0.087355 is worse than Transformer 0.078533. Hybrid normal reward is 0.055716 below exact Bayes (required <=0.03). CPU hybrid/Transformer training ratio **1.122×**, within the 1.5× budget but insufficient to overturn quality failures.
- OOD noisy reference and weak calibration cases worsen the hybrid relative to Transformer, and the smaller GRU is best among the trained models in all families. This gives **no competitive proof** for adding the original architectural complexity, despite its history use.
- **No paid compute and no SI breakthrough.** Keep all three completed results and consumed interrupted seed; require exact-head CI and independent replication before even considering main merge. The next CPU hypothesis should focus on **learning observation-model uncertainty without supervised hidden z labels**, then use true multi-step active calibration, while preserving explicit Bayesian and Transformer/GRU controls, matched compute and proper leakage tests. Do not reinterpret this as general intelligence.

