# PGW-v3 Stage-2B6: prospective CPU study runner design

Prepared 2026-10-13. **Design only: no source implementation, optimizer step, scientific seed reservation, training job, paid allocation or publication is authorized by this document.** Candidate numerical settings below are recommendations for a new frozen contract, not approved settings. No scientific attempt IDs or seeds have been assigned.

## 1. Source and completed prerequisites

Canonical continuation is [issue #1437](https://github.com/vinceackermann2-sys/tam-research/issues/1437), `PGW_V3_STAGE2B6_CPU_STUDY_RUN_AUTHORITY_GATE_DESIGN_ONLY`. Live readback has no comments or labels, updated 2026-10-10T19:55:56Z. No newer PGW scientific issue appeared in the live issue search. This is a successor design to [#1429](https://github.com/vinceackermann2-sys/tam-research/issues/1429), not authority to revive historical experiments.

Stage-2B5 PR #1432 merged at `ceb2008b24441eb650af7ebb2f4ca96b7fc456ec`. Original exact-head CI #38081189575 passed; original merged-main CI [#38081489818](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38081489818) independently reads `completed/success`, `run_attempt=1`, exact merge SHA. The relevant Stage-0, Stage-1B and Stage-2A/2B0/2B2/2B3/2B4/2B5 source paths are unchanged from that merge to inspected checkout `78dcf2a1c40dadc3b69a5aa542d7d50629afa33d`. The merged public source previously checked was `107342cb15fa53e81948869eac450f2c69d1419a`. The eventual runner must reference its own reviewed commit merged into main; this document does not select an unmerged launch commit.

The exact merged-module `candidate_corpus_manifest()` readback already reproduced all literals frozen in #1429 comment `6101556127`:

| Population | Samples | SHA256 |
|---|---:|---|
| train | 2,880 | `0174103c2ef1cf14e5dec0df23498d994062dabd0e4151d02f7d2ba4f055d9a3` |
| validation | 1,440 | `c6b4fb7bf1943755510bb24fb98053a0ccf45cc83ecb4b51c821193845668611` |
| test | 1,440 | `3b3ab8e8b5d68337ad1608053541838f02369033b6c95e538a7266b330d8c7d3` |
| all | 5,760 | `36a40585fe3deb2c4fc268c32eb93092ebc46f7809b772cdbc223d0185d940ef` |

Existing Stage-2B4/2B5 engineering tests passed locally: **42 passed in 8.03s**. Direct corpus/control generation took 2.63s. These are completed zero-training checks, not scientific attempts; this design does not request repeating frozen CI heads.

Keep negative PGW-v2 #1181 and PGW-core #1201/#1213 classifications immutable. No Modal launch, consumed namespace, historical seed, checkpoint resume or automatic fallback is part of this design.

## 2. Candidate estimand and arm classification

Default next study: a **frozen-arm synthetic retrieval comparison with explicit mechanism, depth and auxiliary-objective confounds**. It can establish whether these particular implementations learn this particular task at the declared budgets. It cannot establish a capacity-matched architecture advantage or a breakthrough.

Primary comparison candidates:

- Frozen `PGWV3Stage2A(PGWV3Stage2AConfig(route_mode="hybrid"))`.
- Frozen `GlobalPositionCausalReference()`; the no-global Transformer is not the primary reference.

Do not automatically add the five additional PGW routing arms, full-no-global or chunk-only. Their source-level structural audit is useful, but trained secondary arms would consume new budgets/seeds and need explicit inclusion. If later included, share each family's initial tensors as defined prospectively, retain every result, and classify unequal active capacity as an ablation.

The existing instantiated counts differ by only seven parameters. Exact static accounting is:

| Group | PGW hybrid | full-global reference |
|---|---:|---:|
| token + local-position embeddings | 8,448 | 8,448 |
| causal attention projections/biases | 4,224 | 4,224 |
| FFN + two encoder norms | 4,320 | 6,010 |
| answer head + output norm | 1,153 | 1,153 |
| predictor | 512 | 512 |
| utility | 32 | 0 |
| remaining workspace tensors | 1,665 | 0 |
| total | **20,354** | **20,347** |

Reference FFN width is 90 versus PGW width 64. Nominal mismatch is `7/20347 = 0.0344031%`. Reference FFN capacity replaces PGW workspace parameters; this is a different allocation of capacity, not equal mechanistic depth.

### Active-capacity derivation and measurement contract

Only 140 token IDs can occur in this fixed grammar: four command/marker tokens, eight keys, 32 values, 96 fillers. `NOT_FOUND=64` is an external label. Consequently **at least 3,712 embedding coordinates** (`(256-140)*32`) have no input lookup and cannot receive an answer/auxiliary gradient. A small batch activates still fewer rows. AdamW's decoupled weight decay can nevertheless change unused embedding rows when their tensor participates in an update; these are objective-gradient eligibility bounds, not counts of all coordinates touched by the optimizer. Report total instantiated coordinates, structurally eligible coordinates, parameter tensors with gradient edges, exact nonzero derivative coordinates, and gradient magnitudes separately.

Predictor surprise is detached inside PGW routing, so answer CE cannot teach its 512 predictor weights. Reference answer CE also excludes its predictor. The existing auxiliary enables these weights but uses different information in the two arms.

The following are structural **upper bounds**, not measured active counts:

| Arm | CE+aux eligible ceiling after grammar exclusion |
|---|---:|
| PGW hybrid / utility-only | 16,642 |
| PGW surprise-only / fixed-random / recency | 16,610 |
| PGW no-workspace | 14,945 |
| reference full-global / full-no-global / chunk-only | 16,635 |

The 32-coordinate utility is disconnected in surprise-only/fixed-random/recency; no-workspace disconnects the 1,697 nonpredictor workspace coordinates. These bounds follow graph usage, not a claim that every remaining coordinate has a nonzero derivative. The nominal seven-coordinate gap divided by these primary upper bounds is 0.04208%; it still does not establish actual parity.

Before science, an additive reviewed preflight should examine **TRAIN-only** samples at delays 1/2/4, using B=2 and proposed B=16, separate answer and auxiliary losses, no optimizer. Retain per-tensor masks, exact union counts, gradient norm distributions and structural eligibility. Use deterministic engineering fixture initializations explicitly distinct from later science seeds. Do not redefine active capacity by an arbitrary gradient threshold; provide exact nonzero counts plus sensitivity summaries. The same derivative information for real scientific initializations can be recorded only after their approved reservations. A union across samples is an observed lower bound, not proof of equality over training trajectories.

For a strong primary architecture attribution require the frozen `<=0.1%` active-capacity criterion in the new contract and a documented effective-depth/objective policy. If it fails, classify the comparison as an unmatched mechanistic study or stop before optimizer, according to the prospectively chosen interpretation. Never add inert parameters to force the count.

### Effective depth and state

T is 80/88/104; there are 10/11/13 chunks. Before the final READ query, PGW has executed **18/20/24 sequential workspace writes**, two for every preceding chunk. The longest causal memory path includes chunk-local attention/FFN, event projections, recurrent gated key/value overwrites, and a final soft read/output projection. This is not a one-layer Transformer path. The final READ chunk's writes occur after its output and cannot teach that answer, although the frozen implementation still computes them.

Workspace state is four slots times (eight key + sixteen value coordinates), **96 dynamic scalars per example**, reset to trainable initial tensors every example. The reference has one full-prefix attention block. It receives deterministic global positions; PGW uses repeated local positions plus sequential state updates. No persistent cross-episode or natural-language capability claim follows. Matching reference depth or changing the state mechanism would require new architecture choices and a separate source-bound preregistration; this design preserves both frozen arms.

## 3. Exact objective and its remaining confound

Default candidate loss is `L = mean_answer_CE + (1/32)*existing_auxiliary_loss`.

Both frozen helpers sum squared normalized error over width 32, then average over B x eight WRITE chunks x seven within-chunk transitions. **They do not average over the 32 coordinates.** The recommended lambda `1/32` explicitly converts this sum to coordinate-mean MSE. Each epsilon-regularized layer-normalized vector has norm at most sqrt(32); therefore the raw auxiliary is in [0,128] and its recommended weighted contribution is in [0,4]. This is a normalization argument, not a measured optimal lambda. No coefficient tuning is allowed after validation/test outcomes.

Use the existing PGW `predictor_auxiliary_loss()` and reference `forward_with_aux()` unchanged in the default candidate. Targets are detached before normalization, position 0 is excluded from prediction, position 7 is target-only, and only the eight WRITE-prefix chunks enter auxiliary loss. Preserve all mask/anchor and label leakage tests.

PGW hidden states are independently chunk-local; reference hidden states use earlier chunks and absolute positions. Thus target, prediction input and learning signal differ despite the same algebraic loss. Label this confound explicitly. Setting lambda=0 would strand the PGW predictor and change the tested hypothesis. A new shared-target projection/token-prediction objective, or forcing both predictors to the same local views, would be a separate architecture/objective implementation; do not introduce it silently.

## 4. Static compute derivations: useful accounting, not measurements

For B examples, width32 and sequence T, dense QK/AV MACs are `64*B*T^2` for full attention and `512*B*T` for chunk-local attention. Their ratio is T/8, but this is not total forward, derivative, optimizer or wallclock cost.

Counting linear/dense-matmul MACs only, excluding elementwise nonlinearities, norms, softmax, routing/top-k, biases, memory movement, backward and optimizer:

- PGW answer forward: `B*(10320*T + 1056)`; includes local QKV/output/FFN, dense local QK/AV, per-token predictor/utility/read, and two gated writes per chunk. It includes the computed first suppressed read and final unused writes.
- Current PGW auxiliary helper independently re-encodes the first 64 tokens: **585,728*B** additional MACs (including 56 predictor transitions).
- Reference shared-hidden CE+aux forward: `B*(9856*T + 64*T^2 + 1056 + 28672)`.

| T | full QK/AV MACs per example | local QK/AV MACs | current PGW CE+aux partial MACs | reference CE+aux partial MACs | ratio |
|---:|---:|---:|---:|---:|---:|
| 80 | 409,600 | 40,960 | 1,412,384 | 1,227,808 | 1.15033 |
| 88 | 495,616 | 45,056 | 1,494,944 | 1,392,672 | 1.07344 |
| 104 | 692,224 | 53,248 | 1,660,064 | 1,746,976 | 0.95025 |

These are **source-level dense arithmetic estimates**, not empirical FLOPs or latency. They already show why an attention-pair ratio alone misstates total compute. A later additive shared-hidden PGW adapter could avoid encoder recomputation while preserving mathematics, but needs a separate equivalence-reviewed implementation; do not silently discount operations that the executed helper actually performs.

### Required real CPU resource preflight

Record CPU model/instruction support, OS/kernel, Python, exact torch wheel, BLAS/backend versions, core quota, affinity, intra/inter-op threads, dtype and implementation commit. Current local engineering environment is Python 3.12.14 / torch 2.10.0+cpu / x86_64; it is not evidence of the Colab CPU configuration. Set one intra-op and one inter-op thread in a fresh child process; FP32, no autocast/compile, dropout already zero. Preserve validation and runtime costs.

Use the exact proposed CE+aux pathway. For both primary arms and each length, measure inference forward and forward+CE+aux+backward separately with 3 warmups and 10 recorded repeats on TRAIN-only B=2/B=16 batches. Alternate arm order by a fixed list, restore state, verify byte-identical parameters, no optimizer. Record raw times, median/p95, setup, data-generation and serialization costs separately. Profiling and timing are separate passes; profiler overhead is not training time.

Collect actual executed CPU forward/backward operator traces with shapes and a declared arithmetic counting rule (one scalar multiply or add = one FLOP, one multiply-accumulate = two FLOPs; report transcendental/compare counts separately). The torch profiler's `with_flops=True` covers selected matrix/convolution operators and does **not** certify whole-step FLOPs. Report counted FLOPs, uncounted operator classes and coverage. If fused attention backward, elementwise or optimizer operations remain unaccounted, label the measurement partial and block an equal-FLOP or strong compute-efficiency claim. Never fill omissions with the attention-only table. An approved science runner must subsequently include the actual optimizer phase in its measured ledger. This proposed profiling protocol is not executable authority for either derivative probes or optimizer probes; it needs its own reviewed engineering scope, with optimizer phases prohibited until the science job is explicitly approved.

Use isolated subprocess resident memory, baseline and sampled peak RSS, explicitly a process-memory metric. Tensor-allocation profiling is a separately named metric only if the implementation tracks allocations/deallocations correctly. Do not substitute Python tracemalloc or cumulative global RSS for native peak tensor memory.

Primary comparison is **equal-token**, with full measured CPU time/FLOP ledgers. It does not become an equal-compute claim merely because both arms have the same wallclock cap. A genuine equal-wallclock/equal-FLOP training panel is separately scoped before training, with its own fresh reservations if it requires new trajectories. Preserve checkpoints and loss/accuracy-versus-consumed-resource data for prospectively selected fixed-budget analysis; no extrapolation or outcome-driven checkpoint selection.

## 5. Candidate single-seed-pair schedule and limits

This recommendation is a deliberately bounded first study; it may be underpowered and a negative result at this horizon must be retained. Do not add epochs automatically to reach the acceptance gate.

| Setting | Proposed value, pending new contract review |
|---|---|
| arms | frozen hybrid and frozen full-global only |
| precision/device | FP32 CPU; hard reject CUDA/TPU execution |
| optimizer | AdamW, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01 |
| LR schedule | constant; no warmup, no scheduler/tuning |
| gradient clipping | global norm 1.0 after combined loss backward |
| auxiliary lambda | exactly 1/32 |
| batch / accumulation | 16 complete equal-length examples, accumulation=1; no padding |
| horizon per arm | **one complete train epoch: 180 updates, 2,880 examples, 261,120 input tokens** |
| buckets | 960 train examples at each T=80/88/104; 60 batches per bucket |
| paired data order | identical row-ID order for both arms; deterministic train-only shuffles bound to approved seed and explicit namespace |
| bucket cadence | round-robin delay1,delay2,delay4; repeat 60 times; no dropping/replacing rows |
| validation | full frozen validation once at terminal update180; no early stopping or configuration selection |
| test | full frozen test once after all updates and locked terminal checkpoint; no checkpoint selection |
| checkpoint cadence | initial and updates30/60/90/120/150/180; terminal only for primary scoring |
| proposed training cap | **45 seconds per arm**, including zero-grad, CE+aux forwards, backward, clipping and optimizer; a reviewed child-process watchdog must enforce it before a job can claim a hard cap |
| optimizer attempts | each arm exactly once; no resume or retry |
| science seeds / IDs | **UNASSIGNED** for paired seeds1,2,3 and six arm subattempts |

The exact train token total derives from `960*(80+88+104)=261120`. Each complete validation or test pass processes `480*(80+88+104)=130560` input tokens per arm. Training examples are not counted as inference tokens; the helper's duplicated prefix and predictor work belongs in compute, not a falsely enlarged data-token budget.

Freeze initial tensor generation prospectively: same approved science seed maps to separate documented PGW/reference initialization namespaces; copy exactly matching shared tensors (token/local-position embeddings, attention, compatible norms, answer head, predictor) where the contract specifies, while nonmatching FFN/workspace shapes have independent namespace-defined draws. Save initialization hashes. Do not choose seeds based on observed gradients or results. The final contract must decide this mapping before a seed is reserved; family/shared copying is a recommended policy, not an already implemented initializer. Arm order candidate across paired jobs is PGW/reference, reference/PGW, PGW/reference, fixed by pair index before launch.

One paired job is one issue-worker attempt, with two immutable arm subattempt records. At least three separately approved paired jobs are needed for the frozen >=3-seed threshold. Each worker invocation stays `--max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4`, with CPU specified. Do not interpret this design as authorizing three launches, increasing limits or reusing #1443.

Both arms propose 90 seconds maximum training allocation in total. In the currently inspected worker, the shared 180-second allocation/execution/retrieval deadline reserves 45 seconds for collection, so `colab run` has approximately **135 seconds**, less pre-execution API overhead, for provisioning, bootstrap and script execution. The proposed full job must fit startup/dependencies, both training phases, **four complete held-out passes** (validation and test for each arm), and checkpoint packaging into that smaller run allowance, and verified retrieval into the remaining deadline. This fit is **not established** by local tiny tests. Stop/post-usage checks in the worker's `finally` path use separate bounded calls and can extend total elapsed time beyond 180 seconds; do not claim the existing setting caps the entire lifecycle or compute-unit spending. Teardown and the final balance/session verification remain mandatory.

Reserve explicit measured headroom in the final specification; if the representative no-training preflight predicts the full job cannot fit, do not reserve a science seed or begin optimization. A longer job needs reviewed implementation and separately approved limits; the bridge maximum accepted configured timeout remains 300 seconds. Never silently shorten the epoch or skip an evaluation to fit. The candidate 45-second arm cap is not implemented by this design or by the existing generic worker; a future reviewed runner must enforce it while retaining durable partial checkpoints.

If an arm reaches its 45-second cap before update180, fail the pair as **INCOMPLETE_BUDGET**, retain all partial artifacts, and do not claim the equal-token primary endpoint. The other arm is not an automatic fallback/retry; the predeclared job handles cleanup and partial records. An incomplete pair cannot satisfy the three-complete-pair acceptance gate. A watchdog is not a compute-unit spending cap.

## 6. Prospective acceptance and reporting

Preserve #1429/#1437's candidate thresholds; before execution the new contract must freeze their aggregation explicitly. Recommended interpretation is a stringent primary conjunction:

- All three or more approved paired attempts finish the declared epoch/evaluation; no failed attempts are excluded or replaced.
- Across seeds, mean PGW test macro-cell accuracy >=0.80, mean present-only >=0.80, mean missing-only >=0.90, and PGW present-only >=0.75 in every seed.
- Mean paired PGW minus full-global test macro >=+0.05 absolute; strictly positive deltas in >=2/3 seeds (for more than three, the exact win-count rule must be frozen prospectively).
- Finite loss/gradients/outputs, source/manifest integrity, zero illicit label access, and artifact retrieval/cleanup evidence complete.

Per split preserve all prediction IDs/classes, 180-cell correct/total, present/missing confusion, delay/overwrite/last-position strata and immutable initialization/configuration/checkpoint hashes. Report all per-seed endpoints, paired deltas, mean/sample SD, and a preregistered paired interval with its small-n limitation; three seeds do not support confident broad claims. Synthetic examples share a grammar and key/value domain, so exact semantic trace uniqueness is not distribution-shift independence.

Always-NOT_FOUND must remain 50% macro/overall, 0% present, 100% missing. Report oracle and naive algorithm controls as integrity checks, not trained baselines. No best-seed or best-checkpoint reporting, test-based stopping, post-hoc lambda/routing selection, silent replication, or rewriting a failure classification.

If thresholds pass but active/depth/objective fairness remains unresolved, classification is **task-specific signal in an explicitly confounded frozen-arm study**, not Transformer superiority. If thresholds fail, retain the negative study. No synthetic outcome permits a breakthrough, language NLL claim, H100 substitution or scaling. A later language/compute panel needs a separate protocol and authorization.

## 7. Attempt, artifact and cleanup contract

Before allocation: verify a real local worker exists with the unchanged settings, current usage balance, no residual runtime, reviewed main-merged script/source and owner/approval-label/GPU/lock checks. Local OAuth stays local. Before optimizer: durable reservation binds issue/job ID, paired scientific seed, source/config/data roots and both arm identities. IDs/seeds/storage root remain unassigned until the owner reviews the final study scope; elapsed time is not approval.

Persist initial/final/intermediate checkpoints, optimizer/config/RNG/data-order state, exact row-ID schedules, predictions/metrics, loss/resource curves, complete logs, source/dependency/runtime identity and failure receipts. Checkpoints are evidence, not permission to resume. Default failure/timeout path packages what exists with explicit incomplete status.

Use the reviewed bridge's agreed durable storage location, with an archive manifest, SHA256, expected-member and source/job/seed provenance verification. Verify successful receipt outside temporary Colab storage **before teardown**; a printed success message alone is insufficient. Fail collection if integrity does not match, retain the attempt consumed, and surface the cleanup outcome. Public telemetry is small permitted nonsecret metadata; raw private logs/credentials never enter issues. Check current balance and confirm actual runtime teardown after every attempt. No automatic retry, second-GPU fallback or Modal transfer is involved in this synthetic study.

## 8. Decisions that actually require owner scientific review

1. **Interpretation or new architecture:** accept the recommended immutable-arm mechanistic study with explicit depth/auxiliary confounds, or commission a new comparable-target/depth/capacity implementation. The latter is a new scientific design, not a backend migration. Strong architecture claims are blocked until those choices are resolved.
2. **Concrete scope and resources:** approve or revise the proposed one-epoch paired schedule, objective coefficient/optimizer and three-pair scope only after no-training Colab resource evidence shows it fits. Assign fresh seed IDs, attempt identities, storage and final merged source in that approved contract. None exists now.
3. **Threshold aggregation:** owner/reviewer must freeze whether the existing >=80/80/90 criteria mean across-seed means as recommended here or an even stricter every-seed requirement; #1437 does not settle this wording. Resolve before outcomes, retain the already frozen +5pp / >=2-of-3 / each-seed-present>=75% constraints.

Routine runner choices (hash verification, length bucketing, CPU rejection, nonsecret logging, durable receipt validation, source checkout) can be implemented under separately authorized engineering scope without asking the owner to choose arbitrary internals. This prospective design is published for review with PR #1446; it is not an executable job contract.

`cpu_training_authorized=false`
`gpu_authorized=false`
`scientific_seed_consumption_authorized=false`
`scientific_seeds=UNASSIGNED`
`attempt_ids=UNASSIGNED`
`scale_up_authorized=false`
`breakthrough_claim_allowed=false`
