# Prospective RLT continuation design after BB

Status: DESIGN FOR REVIEW ONLY. No training, heldout scoring, paid execution,
attempt reservation, job identity or scientific seed assignment is authorized
or performed by this design. Publishing or merging this document does not
authorize either proposed experiment.
The numerical panel choices below are proposals to freeze in a reviewed protocol
before any execution; they are not approved jobs or revised historical gates.

## Source and immutable evidence

Read-only source: `7cff4302caed515b19f683ba1a0f9be10a065659` on
`exp/rlt-ax-bounded-transformer`. PR #1416 is draft, targets `exp/rlt-colab`,
and is not merged into main. The Colab issue-worker accepts reviewed scripts
from commits merged into main; branch-local execution must not bypass that rule.
Review/importing a continuation requires an explicit scope that preserves the
isolated RLT lane and does not trigger old experiment workflows.

Latest source decision:
`experiments/rlt/decisions/2026-10-10-bb-trained-carry-causal-ablation-stage-gate.md`.

Freeze AW, AY, BA and BB jobs, claims, source commits, metrics and interpretation.
Never rerun/resume their entrypoints, replace their results, or reuse their IDs.
Protected scientific replication seeds `58232` and `58233` remain untouched.
Do not reuse seed roles from consumed historical attempts. No prospective seed
integers or attempt IDs are assigned by this document.

- AW: intermediate memory positive; 127-token retention near chance.
- AY: new training outcome collapsed at long gaps; checkpoints were lost.
  Preserve the negative result and missing-weight limitation, with no replay.
- BA: new trained carry model succeeded on the narrow binary task; checkpoint
  actually exists in Git, 120,530 bytes, SHA-256
  `74fe2bb1cfb74242aaba3d0eb8ac5df13c1a421a7d50054a0d7dbf503fb4149b`.
- BB: causal erasure of the same BA weights removes long-gap capability on new
  evaluation inputs. This establishes a task-specific mechanism for one trained
  model, not training robustness, language competitiveness or breakthrough.

## Question and independent units

The next training question is whether the unchanged bounded carry model learns
the registered SET/QUERY task reliably under one frozen training recipe across
independent fresh initialization/data draws. It is not a replication of AW, AY
or BA, which used different seed vectors and partly different comparisons.

Proposed fixed panel: eight matched training triplets, chosen before outcomes.
One independent unit is a triplet's fresh initialization/data seed vector, not
a token, heldout example, gap, counterfactual member, checkpoint or trained arm.
All three arms within a triplet share its deterministic data stream. Carry and
reset start from exactly identical weights; the separate Transformer receives
the registered initialization rule but has different parameter shapes.

Use disjoint domains for initialization, short-stage data, long-stage data,
heldout data and diagnostics. Freeze the resulting integer seed table only in
the separately approved protocol; check collision/reuse against historical
reservations, protected seeds and all per-step/per-gap derivations. Do not use
implicit `base + step` namespaces without bounding and collision validation.
This joint-seed design estimates recipe stability; it cannot isolate the causal
effect of initialization versus data randomness. Such a factorial study would
require its own protocol and additional authorization.

## Comparisons and model invariants

1. Bounded carry RLT: frozen tiny configuration, 27,712 parameters, vocabulary
   16, width 32, four heads, two stages, eight-token fixed chunks, chunk-local
   causal encoder/self/cross attention, one carried affine recurrent state.
2. Same-weight-architecture reset control: identical configuration and parameter
   count, only the nontrainable carry flag differs. It resets to the learned
   start state at each eight-token boundary. No hidden KV retention.
3. Independent AX sliding-window Transformer: frozen 27,680-parameter, two-layer,
   width-32, four-head model, eight-token per-layer KV window, maximum full
   receptive distance 14. Its 32-parameter difference is 0.11547% relative to
   RLT: this is a disclosed close-count engineering control, exceeding the
   repository's 0.1% scientific matching target. Do not silently add unused
   parameters or describe it as exact matching. A redesigned matched baseline
   needs separate reviewed source and cannot inherit these comparisons.

The reset arm has no cross-chunk memory; the AX arm can see older than the final
eight tokens through its stacked layers. Register information-access proofs for
each task, including the full 14-token field. A deliberately inaccessible SET
cannot support a general Transformer-superiority claim. Both models' full-BPTT
activation storage grows with sequence length; fixed inference carry size is
not constant-memory training.

No event-specific gate logic, optimizer tuning, curriculum changes, additional
model capacity, precision substitution or early stopping can enter this panel.

## Objective, data and budgets

Keep the original synthetic token semantics: SET0=1, SET1=2, QUERY=3,
distractors=4..15; QUERY is the last input token and its binary next-answer
target is not included in the input. Independently implement/verify the
last-write oracle. Freeze a generator manifest and exact train/heldout fixture
identities, include label balance by distance and forbid target information in
the permitted suffix for registered inaccessible cases.

Proposed primary comparison uses the historical common **equal CPU wall-time**
recipe: batch eight, short stage sequence eight with gap one for 15 training
seconds per arm, then sequence 128 with gaps [1,4,16,64,127] for 90 training
seconds per arm. Per-step batches are paired across arms while available; faster
arms see a longer prefix of the same deterministic stream. No optimizer-step
cap. This preserves time policy but intentionally does not equalize tokens,
steps, FLOPs or the effective information horizon. Report those differences.

Binary cross-entropy at the final QUERY only. AdamW: short LR 0.003, long LR
0.001, weight decay 0.01, gradient norm clipping 1.0, betas (0.9,0.999),
epsilon 1e-8, amsgrad/maximize/capturable/differentiable false. Freeze the actual
PyTorch implementation choices for foreach/fused and set/log defaults explicitly
rather than letting a later release redefine them. Keep optimizer moments across
the LR transition; no scheduler beyond the declared stage switch. CPU FP32,
no autocast, no compile, two intraop threads, pinned Python/PyTorch/package
versions and platform recorded. Confirm these explicit settings match the
historical source defaults under its reported runtime; document any unavoidable
runtime change instead of claiming bit-identical replication.

Historical AW records Torch `2.14.1+cu130` while this validation environment uses
Torch `2.10.0+cpu`; the structural tests passing on the latter do not establish
identical training outcomes or throughput. The future protocol must explicitly
choose/pin a reviewed CPU runtime and classify the runtime change. Do not copy
the historical version string into new provenance or claim equal old/new CPU
time comparisons merely because both execute without a GPU.

Start each stage's clock before batch generation and stop at the first complete
optimizer update after its time target. Report target time, actual elapsed time,
last-update overrun, generator time, forward/backward/update time, steps and
input tokens separately. Evaluation, checkpointing, setup and retrieval are
outside the training clocks and inside separately bounded job lifecycle limits.
Rotate/balance arm execution order using a predeclared schedule and record CPU
host/load/thermal conditions. Never pool dissimilar hardware into an equal-time
efficiency estimate. An implementation failure or short watchdog makes an arm
incomplete; it does not authorize a retry or truncation of the scientific budget.

Do not call the final endpoint equal-token. A future equal-token study would
freeze short/long optimizer counts and tokens for every arm plus a separately
enforced execution-time ceiling before separate approval; no posthoc budget
chosen from these results, early successful checkpoints or changing CPU throughput.

Eight proposed triplets imply **at least 2,520 CPU training seconds (42 minutes)**
before evaluation, environment bootstrap and artifact verification. One triplet
already requires 315 training seconds. No suitability for the unchanged Colab
180-second worker or 300-second implementation maximum is asserted. The bridge
limits remain 180 seconds / one attempt / CPU,T4. Longer execution, decomposition
with checkpoint transfer, or larger limits need reviewed implementation and
explicit job scopes; division into jobs must not alter any arm's optimizer/data
state or become unauthorized continuation after a timeout.

## Sealed evaluation and grouped reporting

Before any panel execution, seal an independent balanced corpus shared across
all arms and training units: proposed 256 examples per class per registered gap.
Seal a separate counterfactual corpus; pair members have opposite oracle answers
and identical accessible suffixes where the registered information proof applies.
The answer oracle, training/evaluation split labels, gap metadata and generation
seeds are available only to the harness, not as additional model inputs.

For every unit/arm/gap report class-conditional and balanced accuracy, binary
NLL, prediction class proportions, both-members-correct counterfactual fraction,
and logit differences. Include zero-shot initial and fixed final checkpoints;
no best-checkpoint selection, retry on unfavorable outcomes or extra training.
Report gate/state dynamics as diagnostics, not proof that direct gate-product
decay alone caused success/failure: indirect paths and readout behavior exist.

Primary unit-level endpoint proposed for review: carry training succeeds only
if final class-conditional accuracy is at least 95% for BOTH classes at EACH
off-chunk gap 16,64,127, and the near-window positive controls pass the proposed
95% criterion, to be frozen with this new protocol. Publish every unit's full
scores and success indicator. The numerical definition must be approved/frozen
prospectively, never retrofitted.

Report the proportion of successful training units with an exact binomial
interval, per-unit minimum off-chunk class accuracy, median/range across units,
and paired within-unit arm deltas. For intervals on mean deltas or pooled task
summaries, resample entire training units and keep all arms/gaps within each
unit together. Do not claim hundreds of independent replications from heldout
examples or five gap bins. Eight units provide limited uncertainty resolution;
even 8/8 observed successes do not establish universal recipe reliability.

List attempted/completed/failed units and denominators separately. Scientific
collapse, infrastructure failure and failed artifact preservation remain
distinct visible outcomes. An incomplete triplet stays incomplete; its favorable
arms do not silently enter a complete-case superiority aggregate. Preserve all
attempts and report missingness. No optional stopping or selective seed removal.

The decision supported by this panel is whether to investigate tiny synthetic
training stability further. No synthetic percentage is a language-model,
hardware-efficiency or breakthrough acceptance criterion.

## Separate richer inference-only design

Use the verified BA checkpoint unchanged, with no optimizer, retraining,
selection of weights or change of vocabulary/readout. Register a NEW prospective
evaluation manifest and authorization; do not execute the consumed BB runner
or reuse BB's heldout seed identities. Model-only BA weights do not justify
claiming an optimizer-state training resume.

Proposed inference factors, all within existing binary SET/QUERY semantics:

- Latest-write versus stale-write interference: two or more opposite-valued SETs,
  balanced last answer, controlled spacing and earlier-write density. Score the
  latest answer and stale-answer error separately.
- Distractor shift: uniformly sampled, repeated and structured existing distractor
  tokens, balanced across target labels with suffix-identical counterfactuals.
- Length/phase transfer: anchor total lengths 64,128,256,512 place the final
  query at phase eight. To test phases one through eight, separately register
  the total lengths from anchor-minus-seven through anchor, keeping the same
  framing origin. Register only valid gap/length combinations with gap less
  than total length. Treat length 128 as an independent-manifest anchor rather
  than replaying BB. Full fixed chunks plus a final partial terminating chunk
  only; do not claim arbitrary segmentation invariance.
- Boundary contrasts: last writes just inside/outside the RLT final chunk and
  AX's true 14-token field. These distinguish reachable history from architectural
  information exclusion. Specify phase-dependent accessible positions explicitly.

For each task compare unchanged carry, same-weight reset and the same-weight
late erasure. Define the erasure boundary as the actual start of the final chunk,
not hardcoded token 120 for every length. Use additional state-swap interventions
only when source/protocol are independently reviewed; do not relabel state norms
as causal evidence. A trained, competent full-context baseline would require a
separate training protocol; an untrained AX model is not a fair capability
competitor to the trained BA checkpoint.

Within this inference-only study the independent learned unit remains ONE BA
checkpoint. Example families and counterfactual pairs assess task sensitivity
conditional on that model; they are not independent training replications.
Group reports by task, length, gap, query phase and class before any optional
macro-average. Mark unseen distributions explicitly. New task/vocabulary semantics
(multi-key recall, arithmetic, natural language) are separate studies, not a
reinterpretation of this binary-trained model's logits.

## Retention, review and launch prerequisites

Before execution, reviewed source merged into main must freeze script/source
hashes, full model and optimizer config, split/generator manifest, limits and
the authorized attempt scope. Owner validation, approval label, GPU allowlist,
exclusive worker lock, durable attempt reservation and no automatic retries
remain enforced. Check actual local worker, current balance and sessions; never
infer availability from consumed #1443 or historical 108.89 units. No Modal
execution or fallback.

For every new training unit preserve final model weights, optimizer state,
phase/step/tokens seen, RNG/sampler state, manifest/config, metrics, logs and
failure receipt. Namespace immutable files by the later authorized identity;
create once, never overwrite another unit or historical evidence. Record each
file's size/SHA-256, source commit, runtime metadata and scientific classification.
BA model-only historical checkpoint stays untouched and clearly model-only.

Retrieve the complete bundle to the separately agreed durable destination while
the runtime is still alive; independently verify hashes, member manifest,
expected checkpoints and immutable receipt after download/fsync. Confirm
weights/config are readable and expected state keys exist without starting
training. Git tracking alone is insufficient when files are ignored; archive
guard and remote-tree or trusted-storage readback must establish retention.
Return success only after retrieval succeeds. Preserve partial/failure evidence
if capture fails; the attempt stays consumed. Teardown must still run, then
verify live sessions and current balance after every attempt; unresolved cleanup
blocks subsequent allocation. A timeout is not a spending cap.

## Separate structural successor

The 112 selected scan/window/stream/archive-guard tests passed in the read-only
git-archive export of the frozen source using Python 3.12.14 / Torch 2.10.0+cpu.
They cover causal fields, scan/gradient equivalence, fixed carried-state shape,
partition equivalence within each API's documented framing and preservation guards.
No historical experiment entrypoint, optimizer update or scored panel ran.

Additional construction-only checks identified missing type guards:

- `BoundedStreamingConfig(..., chunk_size=1.5)` is accepted although its stated
  contract requires an integer; subsequent `range(..., chunk_size)` cannot use it.
- `SlidingWindowConfig(window=True)` and `window=1.5` are accepted although window
  is an integer architecture dimension. Other positive dimension fields need
  independent review; the successor deliberately fixes only chunk/window
  dimensions.

These are malformed-input/API faults in the frozen source, not evidence that
historical valid integer configurations changed or that scientific results are
invalid. Draft PR #1449 at `3d3d73a5d167cad2321c99e73686300a7484c2c1`
contains a separate constructor-only patch rejecting boolean/non-integer
chunk/window dimensions before model initialization. It targets `exp/rlt-colab`,
not main, and provides no research execution authority. Local validation passed
25 new constructor cases plus 31 existing bounded/window structural cases
(56 total); valid historical architecture and parameter counts stayed unchanged.
This patch does not change phase/stream-state behavior or validate the proposed
new tasks. No architecture mutation or historical replay follows.

Other preparation gaps: old recipes rely on implicit optimizer defaults;
heldout helper is hardcoded to length 128; late erasure is hardcoded to token 120;
and close parameter counts exceed the scientific fairness tolerance. Freeze
explicit metadata and add generalized pure generator/information-access tests
before proposing richer inference. Do not reuse the consumed entrypoints.
