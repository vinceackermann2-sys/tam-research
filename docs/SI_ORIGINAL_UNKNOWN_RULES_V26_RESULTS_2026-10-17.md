# Original SI v26 — Unknown-rule in-context transfer, three-seed CPU results (2026-10-17)

**PRESPECIFIED SCIENTIFIC DISPOSITION: STOP promoting the unchanged original 64k-parameter hybrid on tiny synthetic tasks.** New-task integrity gate A PASS; generalization B FAIL, original architecture superiority C FAIL, value beyond simple classical lookup D FAIL. **No superintelligence breakthrough, GPU scale-up, production promise, background automation or main merge.** This is a negative finding for THIS small model/training budget/benchmark, **NOT** a theorem that future SI architectures cannot work.

## Preregistration, provenance, attempt controls
- Isolated branch `experiment/si-original-unknown-rules-v26-20261017` stacked on CI-passed v25 draft PR **#1498**, descended from original full screenshot-inspired hybrid PR #1459. `main` untouched and unrelated CPW/PGW/CHM/RLT/EHW projects unchanged.
- The protocol `docs/SI_ORIGINAL_UNKNOWN_RULES_V26_PROTOCOL_2026-10-17.md` was **committed BEFORE any new v26 confirmation seeds** in commit **1d6d2d46ed356fc866ed95b5c59371228e0a577b**. Full original local frozen protocol SHA256 **1526f46789d1909c959d998967e0065ce5a070948ab452ab199869fe84908e0e**.
- Original unchanged canonical 298-line `tam_research/si_original_hybrid_v0.py` Git blob **575c17dc2f39a29d930d82179aa0c96dcbdc7d0f** is attested at runtime. Includes causal local attention, gated recurrence, top-1 experts and optional refinement; inactive/untrained wider memory, online weights, multimodality and OS tools are NOT tested. Full hybrid **64,614 parameters**, nominally matched TransformerControl **64,637**, smaller unmatched GRU **10,283**.
- Exact source `scripts/si_original_rule_in_context_v26.py` **Git blob e3a4d1e92f6ca27c2908a97d3b36f62e2a5fa1ec**, SHA256 **e644d2e1e40b356642a8d36ae1386d2a7835f06a1f98655fa37ea80435587ad0**. Tests `tests/test_si_original_rule_in_context_v26.py` **Git blob 52fe05fe35d5fb45e6fe9287ed15493310d96bc7**, SHA256 **39cbc029daf76f92a16dceaf548392ad2cb4789fd3fbad851e6825196d568faf**. Both source/test blobs locally and GitHub byte-identical; **7/7 local pytest PASS** both before freeze and after experiments. No hidden query label/latent rule in tokens; seen vs unseen function ID separation and baseline correctness tested.
- Excluded DEVELOPMENT-only seed **118010** (100 steps, 256 eval worlds/family) was recorded BEFORE freeze, including unfavorable hybrid results. No changes after that result to code, learning settings, thresholds, test task IDs, output parsing or gate definitions.
- **NEW predeclared fresh engineering seeds 118211,118212,118213**, each DURABLY reserved in unique directory BEFORE CPU and launched **exactly ONCE**, subprocess **exit0**, **empty stderr**, complete parseable original JSON **3884/3883/3884 bytes**; elapsed wall approximately **9.65,9.04,8.69 seconds** respectively. No retries, reruns, replacements, GPU or paid Modal/Colab. CPU Torch2.10, two threads. Each trained all 3 architectures with 180 AdamW steps × batch96, learning rate.001, weight decay.01, clipping1. Evaluate **1024 independently generated queries per family** × 4 families per seed, identical episodes across the 3 neural models and 3 classical references. Only **3 independent model-training seeds**; a bounded synthetic CPU engineering screen, not broad statistical replication. Complete original JSON/exit/stderr/reservations and source verification SHA256 manifest in `si_original_v26_cpu_evidence.zip` (47 verified archive files; zip SHA256 **d396cf8d121827678e0695aad0b20145bb0f6837817f9831f15f25713a276212**).

## First different-task falsification after many purchase benchmarks
Each independent episode draws a hidden Boolean function. Eight noisy labeled calibration examples (A,B,label) are followed by query (A,B) **without its answer**. Train rule IDs 0–3 ONLY XOR, XNOR, AND, NAND; the four withheld rules OR, NOR, A→B, B→A are completely absent from training. Training label flips 8%; evaluate four preregistered families: seen, unseen heldout functions, noisy support labels 27%, longer sequence with 12 demonstrations. All receive same seven-dimensional causal tokens, only query outcome supervision during training; no privileged task IDs or known-rule teacher/purchase labels. This is outcome-supervised in-context fitting, not model-discovered physical laws.

**Classical controls:**
1. **Uninformed example lookup:** count observed labels ONLY on matching input (A,B), Beta(1,1) smoothing, NO special function-family knowledge. It is cheap and directly tests whether a complex sequence model is necessary.
2. **Known-four-rule Bayes:** uses explicit train task-family truth tables and observation noise (privileged).
3. **Known-eight-rule Bayes:** ALSO knows unseen rule structures and correct noise in advance (EVEN MORE privileged). A comparison to the known-eight rule Bayes measures distance to a strong synthetic reference, not an equal-information model contest.

## Three-seed mean classification accuracy (higher better)

| Method | Seen four rules | Four completely unseen rules | 27% support noise | 12 support examples |
|---|---:|---:|---:|---:|
| **Original complete SI hybrid** | **0.90299** | **0.67936** | **0.75163** | **0.53548** |
| **Nominal matched Transformer** | **0.94889** | **0.73275** | **0.79590** | **0.60905** |
| Small GRU | 0.81185 | 0.47591 | 0.73340 | 0.83659 |
| **Uninformed exact input-pair lookup** | **0.90332** | **0.91927** | 0.74609 | **0.94629** |
| Privileged known-eight-rule Bayes | 0.95833 | 0.94173 | 0.80436 | 0.97754 |
| Privileged known-four-rule Bayes | 0.97461 | 0.67025 | 0.85677 | 0.98633 |

In heldout unseen rules, hybrid per-seed accuracy **[.72266,.66309,.65234]**, Transformer **[.68652,.71680,.79492]**, GRU **[.49219,.46582,.46973]**. Thus hybrid is below the unseen minimum .70 overall and misses simple lookup by **.23991**. In 12-example long-context, original hybrid per-seed **[.48926,.49512,.62207]**, Transformer **[.43652,.63477,.75586]**, GRU **[.85938,.83301,.81738]**; original hybrid mean **.53548** despite more supporting evidence. This likely reflects brittleness to token-position distribution/sequence length, not proof that recurrence cannot work in principle.

Actual mean CPU training wall on reference CPU: hybrid **3.054 sec**, Transformer **1.942 sec**, GRU **0.655 sec**. Hybrid/Transformer ratio **1.573×**, above locked limit1.5. Nominal model parameter parity does NOT prove equal active FLOPs or computational efficiency.

## Exact locked scientific advancement gates

**A validity PASS:** verified canonical original source, 7 integrity tests, unseen-rule separation, unique clean no-retry attempts, seen known-eight Bayes **.95833>=.85** and uninformed lookup **.90332>=.75**. No leakage or incorrect task-boundary evidence found by targeted checks.

**B in-context transfer competency FAIL for BOTH full architectures:** Hybrid unseen mean **.67936 < .70**, long-context mean **.53548 < .70**, unseen simple-lookup shortfall ~.240 exceeds .10 allowed. Transformer unseen **.73275>=.70** but far below lookup **.91927** by .18652 (> .10), long-context **.60905 < .70**; its noisy-label accuracy drop from seen .153 exceeds allowed .12. Multiple mandatory conditions fail.

**C original SI architecture superiority FAIL:** On seen rules hybrid minus Transformer **−.04590** (requires +.025); unseen **−.05339** (requires +.025); hybrid also worse across noisy and long families and CPU ratio1.573 exceeds1.5. Training-seed variance does not rescue the failed means.

**D value beyond simple family-uninformed classical lookup FAIL:** Original hybrid does NOT exceed lookup by >=+.02 on ANY OOD family; on unseen and long conditions it loses by **.23991** and **.41081** respectively. Matched Transformer exceeds lookup on noisy support by ~+.04981, but original hybrid does not; joint ALL-condition requirement fails.

## Decision / honest scope of termination
**STOP this unchanged tiny synthetic original-SI architecture promotion research track** as *not worth additional repetitive benchmark tuning or paid scaling under present evidence*. It now fails strict predeclared criteria on both the old acquisition tasks (v16–v25) and a different in-context hidden-rule transfer task (v26), often against much cheaper simple classical baselines. This is a valid scientific **NO-GO for current implementation and tested training regime**, not a disproof of the user's broad screenshot-inspired SI concept, brain-inspired designs, or an impossibility result about more capable architectures.

If a future project is proposed, it needs a **different independently falsifiable mechanism**, real persistent online memory/learning, real multi-step interactions and strong controls from the beginning—NOT an unbounded sequence of ad hoc tiny-head patches. A new paid GPU/scaling attempt requires its own explicit user approval and prospective scientific protocol. Keep all current PRs draft/unmerged; require GitHub CI exact-head pass and independent peer review.