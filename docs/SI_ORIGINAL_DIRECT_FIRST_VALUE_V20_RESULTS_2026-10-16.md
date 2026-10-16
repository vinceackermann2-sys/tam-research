# Original SI v20 — First-stage logged complete returns versus fitted double-Q bootstrap (three-seed CPU results, 2026-10-16)

**Frozen decision: task/control validity PASS; original-hybrid-over-Transformer relative gate PASS numerically; full learned direct-return mechanism advancement FAIL.** No breakthrough, no paid GPU/Modal/Colab scaling, no main merge or automatic research task.

## Protocol, provenance and scientific integrity

- `docs/SI_ORIGINAL_DIRECT_FIRST_VALUE_V20_PROTOCOL_2026-10-16.md` was committed **BEFORE any new confirmation execution**, explicitly documenting an unfavorable/mixed full-budget development run 112010 and all four locked A/B/C/D gates. Isolated v20 branch stacked directly on CI-successful v19 PR #1492, ultimately the original screenshot-inspired complete SI hybrid PR #1459. Main/other branches untouched.
- Full original 298-line model `tam_research/si_original_hybrid_v0.py` **Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`** unchanged; v19 trainer `scripts/si_original_crossfit_value_v19.py` blob `933512894318a00afd6ee33136cc477116a129cb` unchanged. New **locally executed and GitHub-byte-identical** executable `scripts/si_original_direct_first_value_v20.py` Git blob **`dd95ac8ae978431a6806cc0ea423276cd8234020`**, SHA256 `84f58d0ecae8dca0e82d3d190d33d8747233969c6b5ce4e02b462328b1d907af`; tests `tests/test_si_original_direct_first_value_v20.py` blob **`1e106c04a0cb3a4c6b01f2214bb8aab2b14c822c`**, SHA256 `e4388d5c4a7a6b793478ffb6a406354b34acb1b02a057897052a6430842fd1ed`. **7/7 targeted tests pass** locally before and after confirmation: source/size parity, no future unpurchased probe leakage, selected first-arm-only gradients, acceptance filter independence of outcomes, original actor+Q1 byte-identity, no Bayesian purchase labels, finite heldout factual first-arm calibration.
- **Development ONLY:** 112000 two-step smoke, 112001–112005 test worlds, full-budget 112010 (160 original actor +240 critic +160 Q0-only steps, 256 twin pairs/family). Development showed hybrid direct reward 0.6399 versus frozen 0.6410, Transformer direct 0.6781 versus frozen 0.5000, GRU direct 0.7050 versus frozen 0.6699; no optimization changes or threshold tuning after observing these values.
- **Fresh preregistered seeds 112211,112212,112213** each reserved before CPU and completed **exactly once**, process exit **0**, empty stderr, complete ~33KB parseable individual JSON per seed, no retries, replacements, GPU or scientific-seed reuse. CPU PyTorch2.10, two threads. Per seed train each hybrid, Transformer and small GRU full v19 actor+double-Q ONCE (160+240 steps), clone identical frozen model into two 160-step Q0-only updates (`extra_bootstrap` vs `direct_logged_return`) and unchanged `frozen_v19`. Batch96; four identical heldout worlds per policy: base, higher-cost OOD, weaker-sensor OOD, discounted-strong OOD, **512 exactly coupled opposite-label twin pairs per family per seed** (1024 episodes/family). Only **3 independent training seeds**, small synthetic CPU engineering evidence.
- Mechanism: randomly log first stop/cheap/strong; if purchased log second buy/skip at independent fair coin. For `direct_logged_return`, retain only purchased-first logs whose second randomized action matches the **frozen Q1-deployed** decision, then execute frozen final classifier and fit selected first Q0 output to **observed full net return**. Stop uses randomly executed observed correctness. Original learned actor and Q1 weights remain identical between conditions; ONLY Q0 head trained. Acceptance for first-bought logs averaged **0.501 hybrid / 0.500 Transformer / 0.506 GRU**, as predicted from .5 second logging propensity; heldout Q0 MSE uses the same policy-matched factual return. `extra_bootstrap` gets the same 160 Q0-only optimization updates on identical first-choice logged worlds but Q0 labels are previous Q1-A choose, Q1-B estimate minus first cost, with more eligible first-bought labels. **Data effective sample sizes differ**; effects cannot be isolated as pure change of target label independent of coverage.
- Full original hybrid attention/gated recurrence/sparse MoE/refinement are present; true long-lived external episodic memory, continual updates and real multimodal skills are NOT tested. Original hybrid 64,614 backbone vs nominally matched Transformer 64,637, small unmatched GRU 10,283; the v19 +32 task-head params are unchanged in v20. Similar parameter counts do not establish FLOP/energy equality.

## Three-seed means: REAL net reward, higher better

| Controller/method | Normal base | High probe cost | Weak sensor | Discounted STRONG probe |
|---|---:|---:|---:|---:|
| Known-model two-stage Bayes | **0.748437** | **0.64168** | **0.62766** | **0.849802** |
| Fixed strong-only reference | 0.714900 | 0.57527 | 0.59621 | **0.842050** |
| Fixed cheap-only | 0.672593 | **0.61877** | 0.58342 | 0.645746 |
| Original hybrid frozen v19 | 0.64040 | — | — | 0.6430 |
| Original hybrid 160 extra Q0 bootstrap steps | **0.66638** | — | — | 0.6510 |
| Original hybrid **direct actual return Q0** | **0.68461** | 0.54549 | 0.58339 | **0.77963** |
| Transformer frozen v19 | 0.63141 | — | — | 0.6780 |
| Transformer 160 extra Q0 bootstrap steps | **0.64436** | — | — | 0.7000 |
| Transformer **direct actual return Q0** | **0.64666** | 0.46737 | 0.54886 | 0.73446 |
| GRU frozen v19 | 0.65399 | — | — | 0.6840 |
| GRU extra bootstrap | 0.65270 | — | — | 0.6800 |
| GRU direct actual return | 0.66377 | 0.54652 | 0.56140 | 0.72206 |

`—` omits non-primary OOD reference metrics in this table, not missing evaluation; per-seed JSON and the archived machine-readable summary contain all four families per method. Strong-discount Bayes chooses strong first in **97.6%** of episodes; fixed strong-only beats fixed cheap-only by **+0.19630** and normal Bayes exceeds fixed strong-only by **+0.03354**.

### Primary Q0 calibration, selection and seed-paired changes

**Base paired direct Q0 reward minus unchanged/further bootstrap:**
- Hybrid direct−frozen v19 **[+0.006791,+0.077205,+0.048640]**, mean **+0.044212**, wins 3/3. Hybrid direct−**equal-step bootstrap** **[+0.003123,+0.002923,+0.048640]**, mean **+0.018229**, wins 3/3.
- Transformer direct−frozen **[+0.000516,+0.017702,+0.027537]**, mean **+0.015252**, wins 3/3. Transformer direct−equal-step bootstrap **[+0.000516,+0.008162,−0.001797]**, mean **+0.002294**, wins 2/3.
- GRU direct−frozen mean **+0.009783**, direct−extra bootstrap **+0.011074**; each 2/3 wins.

**Base Q0 factual logged-arm MSE (lower better):** hybrid frozen **0.2063**, extra bootstrap **0.1919**, direct **0.1855**. Improvement direct vs frozen **+0.02075 MSE**, vs equal-step bootstrap **only +0.00635** (frozen threshold >=0.015 vs BOTH **FAIL**). Transformer frozen **0.1753**, bootstrap **0.1688**, direct **0.1634**; improvements **+0.01186** vs frozen and **+0.00534** vs bootstrap (**FAIL**). GRU 0.2007→0.1856→0.1817, modest additional gain.

**Base policy first strong / second buy mean:** direct hybrid **76.2% / 27.6%**, Transformer **42.1% / 77.2%**, GRU **45.1% / 35.6%**. Per-seed second buy: hybrid **[67.5%,15.2%,0.0%]**; Transformer **[97.4%,50.3%,83.9%]**. The frozen interval **5–65% in EVERY seed** fails for BOTH. Strong-discount direct first high: hybrid **78.1%** (>=60% threshold), Transformer **37.0%** (FAIL), GRU **46.3%**. Even hybrid discounted reward **0.7796** is **0.0624 below** fixed strong-only0.8421 (allowed gap<=.035 FAIL).

Mean measured CPU training wall per seed shared v19 actor+Q1 plus one Q0 refit: hybrid **11.42s**, Transformer **9.25s** (~1.23x), GRU **3.04s**. Q0 direct training seconds are not evidence of total inference/energy efficiency; second-stage double-Q still frozen and suboptimal.

## Preregistered A/B/C/D gate outcomes (do NOT revise)

**A task validity PASS:** known Bayesian normal > best fixed single by +0.03354 (required +0.025); strong-discount Bayes strong-first 97.6% (>80%), fixed strong outperforms fixed cheap +0.1963 (>0.10); exact twin observational equality and private label restrictions remain.

**B useful direct-return Q0 sequential decision mechanism FAIL:** Hybrid direct base gains pass >=.015 against both old and equal-step bootstrap with 3/3 seed wins, and its factual Q0 MSE improves >=.015 versus untouched baseline. But **Q0 MSE versus equal-step bootstrap improves only .00635 (<.015), FAIL**; Transformer reward vs matched extra bootstrap only +.00229 (<.015) and Q0 MSE gains <.015 against both. Both fail per-seed second purchase interval. Transformer fails discounted strong-first >=.60; both fail discounted reward within .035 of fixed strong-only and FAIL base reward >= best fixed single+.015 / within Bayes .045. Improved Q0 regression and higher high-first fractions do **not** establish competitive acquisition.

**C relative original-hybrid advantage PASS as a narrow numerical gate ONLY:** direct hybrid base **0.68461** vs matched Transformer **0.64666**, difference **+0.03795**, hybrid wins all **3/3** paired seeds. Hybrid direct > Transformer direct on high costs (.54549 vs .46737), weak sensors (.58339 vs .54886), discounted strong (.77963 vs .73446). Training ~1.23x, <=1.5. This relative success **does not override B**: hybrid remains below the best single-probe strategy and Bayesian reference, and its per-seed second-buy policy is unstable. Not standalone evidence of original SI architecture advantage, and not authority for GPU scaling.

**D causal, source/attempt integrity PASS:** only observed first logged-arm rewards used as direct targets; second policy frozen; Q0-only gradient on selected arm; unpurchased future probe and latent truth excluded from decisions, original source and Git blobs identical, seven tests and all confirmation attempts complete. Exact PR-head repository CI pending separately.

## Final research verdict and next bounded question
The specific **Q0 bootstrap's target design contributes to poor first-probe selection**: actual realized trajectory targets can improve rewards and strong-probe usage, even with an unchanged weak second-stage policy. However, data support is halved by unbiased agreement filtering and the same linear Q0 head is often inadequate; improvements are seed-sensitive, the stronger probe is still undervalued by Transformer under cost shifts, and first-stage improvements cannot repair pathological Q1 second spending. A stronger next CPU test would independently calibrate *both* policy stages using logged full trajectories or fitted policy evaluation with known logging propensities, and compare a **simple linear/contextual regression baseline** before adding neural complexity. Require more seeds, active compute accounting and independently heldout fault families. Keep original hybrid out of paid GPU advancement; no SI breakthrough, no main merge, no background automation.
