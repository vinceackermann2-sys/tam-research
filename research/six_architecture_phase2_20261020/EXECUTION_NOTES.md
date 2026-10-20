# Execution notes and interpretation limits

Phase one remains in ../research_20261020 and draft PR1528. Phase-two sources
are separate and frozen when launched. Git remote heads were rechecked during
this phase and still matched all four pinned source commits. No research source
branch was rewritten or merged.

The previous A100 assignment had ended before this turn; its completed evidence
was already backed up. One fresh A100 was allocated through the user's existing
Colab entitlement under their explicit compute authorization. This session is
NVIDIA A100-SXM4-40GB, torch2.11.0+cu130/Python3.13.15. No upgrade or credits were
purchased. The models fit well below40GB. A private local exporter retrieves
immutable results and full optimizer checkpoints in64MiB SHA256-verified chunks.
Session credentials are confined to work/ and must never enter reports or Git.

Hardware preparation retained the original unaligned model/source and its
measurements. Nearest16 FFN rounding was subsequently attempted; the Conv7
count100,009,488 triggered the mandatory parameter guard BEFORE an optimizer
update. The final selected source uses floor16 and actual99,616,272 Conv7
parameters. These were hardware-only synthetic runs, not language evidence.
Unaligned Conv7's speed penalty substantially shrank with aligned FFN widths;
the final language-training throughput remains the efficacy comparison.

SI's recorded gradient_coverage includes stale gradients left by the last policy
update when freezing parameters. It must not be interpreted as evidence that
those frozen parameters were trained by the head-repair objective. Only the
head parameters are in the repair optimizer, and WEIGHT_INTERVENTION_AUDIT.json
independently verifies that all non-head weights are bitwise identical between
policy and repaired checkpoints in every seed. The stale gradients also enter
global norm clipping; this is the actual executed recipe, retained without a
silent source edit. Direct logits are bitwise identical before/after repair.
The three models' native persistent states are exported; memory/replay have no
ingested episodes in this isolated experiment. Full-system memory/fast learning
is not thereby validated. The finite cue task has only1024 distinct discrete
cue/query combinations, with stochastic sampling and repeated combinations;
perfect held-out sample accuracy is a small engineering result, not broad
multimodal generalization or useful planning superiority over the direct policy.

CHM diagnostic gold interventions use already consumed development labels and
are never reported as deployable accuracy. Native TEST stays closed. PGW traces
use new examples in the same development generator and the earlier tiny trained
checkpoints; slot collapse is a measured failure mode of those checkpoints,
not a family-wide impossibility proof. RLT's public gate prior adds explicit
instruction-type supervision; any robustness gain needs broader tasks before
language scaling. Neither memory diagnostic consumes optimizer token budget.

The large language panel is one seed per lineage, staged on the fixed2B cosine
schedule. Its gates can stop particular recipes below2B. Model size and token
count are ceilings; continuation is conditional on measured research value.
No claim of a completed2B run may be made until that terminal result is saved.

Receptive-field wording clarification: CPW's15 one-step mixers can access the
current token plus15 preceding tokens, a16-position field. The frozen protocol's
"15-token temporal receptive field" refers to the15-token lookback, not the
total number of positions including the current token. The inference check
perturbs only positions more than15 steps back. The model equations are unchanged.

PGW's traced coefficients decompose the final value read for the observed
assignment/gate trajectory. Keys and routing/read weights also depend on prior
content, so a small direct target-value coefficient is not a complete causal
information bound. Slot collapse is observed; its causal responsibility for the
retrieval failure remains a repair hypothesis until an intervention validates it.

Language training TPS times synchronized forward, loss, backward, clipping and
optimizer step. Batch preparation/transfer and zero_grad are outside that timer;
evaluation and checkpoint/export work are also excluded. segment_wall_seconds
includes the segment's training/evaluation/checkpoint work after model setup.
Corpus preparation is a separately recorded shared cost. These are implementation
measurements, not FLOP-matched training budgets or end-to-end service costs.

SI post-hoc component restoration further limits the positive interpretation:
the repaired verifier chooses zero searches on this easy task. Forced depth1/2
search is also100% in all3 seeds. Restoring the original untrained transition
reduces forced depth2 accuracy to24.46%/5.57%/50.63%; restoring either the old
reward head or old value head individually leaves100%. The latter heads can
remain useful fixed projections after the transition coadapts. Thus the small
task supports a transition/planning-coordinate repair and appropriate search
skipping, not an isolated causal benefit from training the value head or useful
multistep planning. These are post-hoc consumed-domain diagnostics, not new
held-out confirmations.
