# Original SI v17 — Preregistered sequential information acquisition (2026-10-16)

Frozen BEFORE fresh confirmation CPU attempts. Isolated branch stacked on CI-passed original SI v16 PR #1489. Main and other RLT/CPW/CHM/PGW tracks unchanged. No GPU, Modal, Colab paid execution, merge or automation.

Full frozen local protocol: /mnt/data/si_v17/PROTOCOL_V17.md, SHA256 bb2d2d67e5f0c8a99c699f6557f257fac6b0a923bf9c44524486728d15c039d5. The rules below match it; preserve snapshot and don't tune after viewing confirmation.

## Hypothesis and world
Build balanced twin sensor-fault worlds from v16: identical preprobe calibration/query and costs but opposite true binary query labels. The policy may stop, buy cheaper noisy probe or buy expensive accurate probe; after actually observing purchased first signal, may stop or buy the other probe, then choose final answer. Each purchase costs actual listed fee, charged in score. No hidden answer, world-kind tag, private fault labels or unpurchased probe signal enter policy input. Low probe mu Uniform[.45,1.0] and price Uniform[.025,.140]. High probe mu Uniform[1.10,2.15], price Uniform[.13,.33]. Each independently gives Normal(theta*mu, 1) if bought. OOD high-cost raises low/high price by .065/.14; OOD weak-sensors sets low mu Uniform[.25,.60], high mu Uniform[.65,1.25]. Training base only, test all.

## Analytic reference and model controls
Known-family Bayesian planner solves two-stage value of information: after first observation y from quality mu, posterior log odds l=2*mu*y; compare stopping success max(sigmoid(l),1-sigmoid(l)) to buying remaining second, whose expected success is p*Phi(mu2+l/(2*mu2))+(1-p)*Phi(mu2-l/(2*mu2)) minus cost2. Choose second if valuable. Stage0 compares stop=.5 to first low/high integrated future optimal value using fixed 24-node Gauss-Hermite quadrature (tested versus 48 nodes). Strong controls: never, always cheap, always expensive, always both, one-step greedy first choice with optimal second stage, and full two-stage Bayes. Report greedy including unfavorable comparisons, no false claim of exact optimality beyond quadrature approximation.

Full exact original 298-line SI model unchanged, Git blob 575c17dc2f39a29d930d82179aa0c96dcbdc7d0f; 64,614 parameters hybrid with attention/recurrence/MoE/refiner versus 64,637-parameter matched Transformer and smaller unmatched GRU ~10,283 parameters. All receive identical 7-dimensional phase-coded causal tokens for purchase actions and observations. Stage0 3-way action, stage1 buy remaining or stop, then final binary answer. Neural networks explicitly SUPERVISED BY THE ANALYTIC BAYES TEACHER for first and second purchase labels, and terminal query labels reconstructed from random executed action plus correctness feedback. This is behavioral imitation, NOT independent world-rule discovery, general intelligence, or a complete SI.

## Frozen provenance / attempts
Executable scripts/si_original_active_experiment_v17.py git blob aa60524483117237d9929a0e44da87a3d10dd8c5 SHA256 f70db56c096fd9ee7cb68f67eabdd01d6bd675173319b797260ac14a013ca4c1. Tests tests/test_si_original_active_experiment_v17.py git blob e0cc960fb7298c126711aa7473d034dcbd6f156d SHA256 2d69d0a260fc68b77deb81a351c212a5a642db16c6841ef46cb8284d0d493b33; 7 local tests passed before freeze. Development ONLY: 109000 (4 steps), 109001-109006 correctness, 109010 (100 steps), 109011 (240 steps), 109019 Bayes diagnostics. Development seed 109011 reward hybrid ~.731, Transformer ~.719, GRU ~.726, Bayes ~.734, greedy ~.735; no tuning after this observation.

FRESH ONE-TIME engineering seeds 109211,109212,109213, each attempt exactly once, retries=0, no reuse. Each 3 models trained 240 AdamW steps × batch96, LR .001, wd .01, clip norm1, CPU torch two threads. Evaluate 1024 counterfactual twin PAIRS (2048 episodes) per base/high-cost/weak-sensor family per seed; paired worlds/costs across all controls. Log attempt markers, exit, stderr, all per-seed JSON, model CPU times, source hashes; 3 independent model seeds = engineering screen, not high-powered independent science.

## Predeclared gates (ALL mandatory within each category)
A. Sequential task valid if full Bayesian base mean reward > never by >=.15, each simple single-probe fixed baseline by >=.025 and always-both by >=.04; second purchase fraction (0.05,0.60); both low/high initial choices >.15; 24- vs48-node Bayes first-choice disagreement below .03 in independent development integrity test. Report greedy result regardless of rank.

B. Learned two-stage acquisition usefulness if BOTH full neural backbones select low AND high initial at >=.10 rate and second purchase fraction within (.05,.65) in EVERY seed, base reward within .045 of known-law Bayes and >=.015 above best fixed single-probe controls on identical heldout worlds. No access to teacher at inference.

C. Original hybrid architectural advantage only if hybrid >matched Transformer base mean reward by >=.015, wins >=2/3 paired seeds, loses no OOD family by >.015, and hybrid mean CPU training <=1.5x Transformer. Small GRU is additional control. Parameter similarity not evidence of equal active FLOPs.

D. Twin preprobe observations must be bitwise identical and truth opposite; no-probe classifiers exactly 50%; stage0 independent of both unpurchased probes; stage1 independent of remaining unpurchased probe. Any leakage invalidates experiment.

No after-result tuning, claimed breakthrough, GPU scaling, merges to main, or automatic tasks even on a pass. Review GitHub exact-head CI and preserve failures.
