# SI Research Track — Executable Hypothesis Workspace (EHW)

Status: **hypothesis + completed local CPU pilot, no breakthrough, no GPU authorization.**
Date: 2026-10-12
Isolated branch: `experiment/si-ehw-causal-hypothesis-pilot-20261012`
Source base: main `4a51f83a70cb58564cc1252a9f57ce7dae811f57`

## Research question

Can an adaptive model learn to **construct, execute, compare, and revise multiple causal hypotheses about a world**, instead of storing all environmental regularities only in shared network weights? Can this yield better transfer, intervention planning, and non-catastrophic online adaptation at matched compute?

EHW is a *candidate mechanism*, not an invented or demonstrated SI. Parts resemble classical system identification, program synthesis, active learning, Dreamer-style world models, object-centric causal representation learning, and test-time memory. Novelty requires rigorous comparison with these predecessors.

## Two competing candidate designs

### A. Hybrid predictive workspace (conservative)

Learned multimodal state encoder -> sparse attention/recurrent backbone -> per-context memory -> action-conditioned latent world predictor -> planner -> verifier. Sparse MoE/recursive thinking optional, only if ablations show a benefit. This has a shorter engineering path but risks stacking already-known components.

### B. Executable hypothesis workspace (riskier; preferred test hypothesis)

- **Observation abstraction:** encode observations into entities, events, relations, uncertainty, and provisional latent state. Variable binding and representation invariance matter more than architecture labels.
- **World compiler:** from experiences `D_t={(state, action, outcome)}`, construct a *small population of executable transition hypotheses*, such as dynamically assembled causal graphs or learned functional programs. The compiler must be learned on diverse worlds, not given the correct functional form.
- **Belief selection:** score each hypothesis against observations with predictive error, uncertainty calibration, and a complexity penalty; retain competing explanations instead of committing too soon.
- **Counterfactual imagination:** execute candidate hypotheses under alternative action sequences, estimate outcomes and disagreement between models.
- **Active experimentation:** plan low-risk interventions that both accomplish goals and distinguish between conflicting explanations.
- **Versioned memory:** retain experiences and skills along with scope, reliability, provenance, validity period, and change points; write/retrieve quickly, consolidate slowly with replay.
- **Compute controller:** use few samples/hypotheses for simple queries, expand search/depth when uncertainty or stakes rise; include verifiers and stop conditions.
- **Continual learning:** fast per-context belief updates first, occasional gradient-based skill consolidation behind replay and regression tests. Store durable external memory separately from transient model state.
- **Language/tool head:** render beliefs into text, tool calls, or motor commands; the text generator is an interface, not the sole world state.

Mathematically (a target, not yet implemented):

```text
z_t = Encode(observation_history)
H_t = Compile_theta(experiences, beliefs)         # hypotheses as executable modules
w_k ~ exp(-predictive_error(H_k) - lambda * description_length(H_k))
future_k = Simulate(H_k, z_t, candidate_actions)
a_t = argmax_a ExpectedUtility(a) + beta*InformationGain(a)
                            - rho*Risk(a) - gamma*ComputeCost(a)
D_(t+1) = D_t union (observation, action, verified_outcome)
H_(t+1) = Update(H_t, D_(t+1))
```

Potentially *stranger still*: the world-model representation could be self-constructing compositional programs, not a monolithic dense vector. This may trade raw throughput for sparse transfer and interpretability. It could also fail completely on messy high-dimensional sensory data. Both outcomes are scientifically useful.

## Phase-0 executable component

Implemented: `experiments/si_ehw_pilot.py`. A four-dimensional simulated world gets a fresh unknown action-conditioned transition matrix **every episode**. Models see 12 context transitions and predict one held-out state transition. The true matrix is never supplied to them.

- **EHW component:** differentiable episode-local ridge system identification from contexts, followed by a learned nonlinear residual correction.
- **Transformer control:** a two-layer Transformer receives the same observed state/action/outcome triples and query; no hand-coded solver.
- **Ridge-only ablation:** a non-neural training-free direct solver in `experiments/si_ehw_ridge_control.py`.
- All predictive models receive identical training batches, observation budgets, learning-rate settings, 700 training steps, 32 episodes/step, and paired seeds. This is **not** a parameter- or compute-matched comparison.
- CPU torch/numpy; no Modal, cloud, GitHub Action GPU, checkpoints, secrets, or paid inference.

Reproduction from repo root (Python 3.11+ with torch and numpy):
```bash
for s in 8117 8118 8119; do
  python experiments/si_ehw_pilot.py --steps 700 --seed "$s"
  python experiments/si_ehw_ridge_control.py --seed "$s"
done
```

### Results — 3 seeds, mean squared error (lower better)

| Held-out evaluation | EHW (5,573 params) | Transformer (68,484 params) | Ridge-only (no trained params) |
|---|---:|---:|---:|
| Unseen world, same family | **0.004087** | 0.025751 | 0.005268 |
| Stronger transition weights | **0.010297** | 0.040154 | 0.011796 |
| Different nonlinear quadratic effects | 0.042291 | 0.042355 | **0.039705** |
| Surprise environment switch | 0.080647 | **0.061986** | 0.074552 |

Per-seed EHW same-family: 8117 = 0.003867, 8118 = 0.004323, 8119 = 0.004070.
Per-seed Transformer same-family: 8117 = 0.027819, 8118 = 0.024115, 8119 = 0.025320.

Mean local CPU training times (machine-dependent, not benchmark compute parity): EHW ~1.10 s, Transformer ~4.40 s.

**Interpretation:** The *solver prior* is doing almost all the work. EHW does improve over ridge-only on the in-family world task, but the gain is modest and the system is worse under abrupt dynamics changes. This is a useful feasibility result and clear failure case, **not evidence of general intelligence, SI, or research novelty**.

## Scientific weaknesses & required fixes

1. **Structural-prior advantage:** current environment is deliberately close to linear system identification, enabling the hand-coded solver. Next must use unknown nonlinear/compositional laws. Include linear-system-ID classical baselines; compare with equally structured neural alternatives.
2. **No parameter parity:** current Transformer is >12x larger, but architecture-comparison claims still require matched **training FLOPs, inference work, wall time, and data**, not just parameter counts. Sweep comparable capacities.
3. **Change-point failure:** the environment can switch halfway through demonstrations; EHW assumes one law for all samples. Add versioned hypotheses, calibrated surprise detection, evidence windows, and explicit reset/retention logic. Ablate them.
4. **Query independence:** current contexts are independent state-action probes, not temporally linked trajectories. Next require partial observability, long-horizon rolls, accumulated error and state inference.
5. **No planner, language, continual training, or multimodal learning yet:** only one isolated predictive component exists. Do not label it an end-to-end SI model.
6. **Not prescient:** evaluate held-out worlds, unseen intervention distributions, composition, non-stationarity, and safety-related uncertainty on predetermined splits. Report failures and costs.

## Preregistered next experiments (CPU first)

**Gate 1: causal revisions.** Mix unseen families (linear, nonlinear, switching, delayed, noisy, distractors). Compare Transformer, recurrence, ridge, EHW, change-point-aware EHW, and an equally structured learned dynamics solver on at least 3 locked seeds. Include no-oracle and ablation controls. Primary: normalized per-environment held-out multi-step action-conditioned prediction loss. Secondary: adaptation lag, uncertainty calibration, compute/experience.

**Gate 2: planning.** Given budgets and uncertain dynamics, measure actual reward/success from choosing interventions and evaluating consequences; include passive predictors and random exploration. Separate predictive accuracy from action effectiveness.

**Gate 3: transfer & long memory.** New object counts, causal graphs, hidden state, rewired rules, delayed reward, and learned skill regression. Test catastrophic forgetting with stable historical tasks.

**Gate 4: language and real tools.** Only after prior gates, implement an equally trained, parameter/compute-matched sequence-model alternative and compare with robust Transformer, recurrent/hybrid, and world-model/agent baselines. Report held-out loss, task completion, sample efficiency, total latency/cost, ablations, and at least three independent seeds.

**No paid GPU job is approved by this pilot.** Before spending, preregister frozen source/tree/harness digests, exact scientific seed/job IDs, attempt reservations before launch, retry=0, budget, success thresholds, and controls. Do not rerun consumed attempts or claim a breakthrough from an engineering smoke test. Keep main untouched; branch -> draft PR -> standard CI/review.

## Prior art relevant to differentiating the idea

- DreamerV3, Hafner et al. (Nature 2025): learned dynamics and imagination-based planning.
- V-JEPA 2, Assran et al. (2025): latent prediction and action-conditioned physical planning.
- Titans, Behrouz et al. (NeurIPS 2025): test-time learned memory.
- Causal-JEPA, Nam et al. (ICML 2026): object-level latent masking and relational world models.
- Object-Centric World Models for Causality-Aware RL, Nishimoto & Matsubara (AAAI 2026).

Important: these works make the general ingredients known. Executable multi-hypothesis composition and controlled adaptation are *hypotheses to test*, not novelty claims.
