# AERA final requirements checklist

Status: live acceptance matrix for the pre-100M architecture candidate. A requirement is not marked complete merely because an interface exists; it must have a mechanism, a test, and where relevant a measured systems result.

## User requirements -> AERA contract

| # | Requirement | Final AERA mechanism | Evidence required before 100M |
|---|---|---|---|
| 1 | Activate only the useful fraction of a much larger model | hierarchical stage/chunk controller + true top-1/top-2 expert execution + stored/active/FLOP accounting | selected experts only execute; route/load stats; grouped/fused GPU path; stored vs active params reported |
| 2 | Spend compute proportional to difficulty | event/chunk controller chooses expert count and latent depth; variable event patching is an orthogonal input policy | learned difficulty->budget relation; hard runtime steps; easy-vs-hard compute gap; event-patch distribution |
| 3 | Do not make attention carry all memory | Flash causal working attention + predictive recurrent stream state + fast neural memory + external exact-memory interface | reset-state ablation, fast-memory stress/isolation, exact-retrieval controller test |
| 4 | Do not store mutable facts/user state in base weights | stable weights for skills; stream state for temporary state; per-session fast memory for mutable neural memory; external DB/retrieval for exact durable facts | overwrite tests, export/import/reset, session isolation, no base-param mutation |
| 5 | Learn/adapt without full-base backprop for every experience | local delta fast-memory update + optional small-module consolidation path; base pretraining may still use backprop | verified local update changes future behavior with zero base-weight mutation; replay/consolidation comparison |
| 6 | Reuse post-training/agent experience instead of discarding it | verifier-backed replay record, prioritized reuse, corrected-lesson replacement, consolidation interface | second-attempt/future-task improvement; stale lesson replacement; provenance retained |
| 7 | Reason in latent space instead of generating long English scratchpads | bounded latent recurrent reasoner with learned depth/halt budget | latent-depth utility at matched compute; hard maximum; adaptive-vs-fixed-depth ablation |
| 8 | Autoregression is not sacred | autoregressive baseline plus parallel block draft/verify interface; masked/diffusion objective remains optional later path | >1 accepted output units/core call at high precision; rejection/latency accounting; AR control |
| 9 | Multimodality should share a world model, not blindly tokenize everything | modality-specific adapters -> shared event/world latent -> action-conditioned state prediction/planning interface | controlled cross-modal retrieval, action dynamics, isolation/fusion proof |

## Additional non-negotiable systems/science requirements

- Causality: chunk/event decisions that affect earlier positions may not inspect later positions.
- State privacy: no session/user state leakage.
- No hidden dense compute: sparse claims must correspond to skipped kernels/FLOPs.
- Controller overhead counts toward active compute.
- Stored parameters, active parameters/event, active FLOPs/event, GPU seconds, throughput, VRAM and compile/setup time are separate metrics.
- Hard inference behavior must be measured; soft differentiable training proxies are not counted as runtime savings.
- Precision-budget output is instrumentation only until real conditional precision kernels exist.
- External retrieval/tool latency and cost count in system evaluation when invoked.
- Every claimed gain requires an ablation that removes the mechanism.
- A lower NLL alone is never sufficient for a breakthrough claim.

## Current evidence snapshot

Mechanism/integration evidence already obtained before this checklist:

- learned sparse routing and expert-count control passed controlled CPU gates;
- learned adaptive latent depth passed controlled CPU gates;
- fast-memory overwrite stress reached zero stale resurrection in the recorded 1,000-operation gate, with no cross-session leakage/base-weight mutation;
- predictive recurrent state achieved 96.61% carried boundary accuracy versus 4.10% with state reset on the frozen controlled gate;
- block drafting reached 99.49% controlled accuracy with ~99.92% precision on accepted drafts;
- controlled fresh-noise multimodal alignment/action-dynamics gate passed;
- BF16 hard-depth execution was fixed and covered by autocast regression tests;
- conditional hard compute is real (lean top-1/depth-2 is faster than heavy top-2/depth-4), but the v6 selected-expert implementation still failed the dense-GPU systems threshold.

## Remaining mandatory gates before a 100M launch can be recommended

1. **Native grouped/fused sparse GPU gate**: hard sparse execution must have a credible production-style grouped kernel and demonstrate useful end-to-end savings, not merely fewer theoretical FLOPs.
2. **Small real-language S2 experiment**: 10-25M-class matched Transformer/AERA comparison on identical real corpus bytes, tokenizer, seed family, optimizer, context and GPU, across >=3 seeds if the first seed is viable.
3. **Real-language mechanism ablations**: dense-routing AERA, fixed-depth AERA, no-fast-memory/state-reset controls, plus a recurrent baseline/TAM where fair.
4. **Final requirement audit**: verify every row above has concrete code + tests + measured evidence or is explicitly labeled a later optional module rather than silently advertised.
5. **Architecture freeze**: after the pre-100M evidence is reviewed, freeze an exact commit before seeing any 100M outcome.

## 100M readiness signal

Recommend a 100M active-equivalent AERA vs matched Transformer/TAM experiment only if the small real-language evidence shows all of:

- no catastrophic language-quality loss;
- stable, non-collapsed routing;
- genuinely variable hard compute;
- reproducible state/continual-memory advantage;
- a credible grouped/fused sparse GPU path;
- and either >=1.25x useful throughput at matched quality / >=20% lower measured compute at matched quality, **or** a unique state/adaptation advantage strong enough to justify the extra systems complexity.

A strong efficiency claim still targets >=1.5x useful throughput or >=30% lower measured cost at matched quality. A breakthrough claim additionally requires replication, fair baselines/ablations and downstream/state-specific capability gains.
