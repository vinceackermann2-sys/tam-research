# Research allocation decisions

These are practical budget decisions about tested implementations, not proofs
that an entire architecture family can never work. New measurements here are
development evidence. See ASSESSMENT.md for source-specific findings and limits.

| Architecture | Current allocation | What is stopping a stronger result? | Next bounded decision |
|---|---|---|---|
| Conv7 | Highest priority to continue | The25M FineWeb screen passed quality/speed gates, but only one initialization at this horizon; no modern-baseline frontier proof | Replicate the0.036 NLL gain and 1.1008× training speed, then increase horizon/size within the actual100M/2B caps. Same-checkpoint erasure confirms convolution reliance; inference cost remains unmeasured. |
| CPW | Continue local mixer; hold AFM scaling | Local receptive field cannot handle distant binding; the51.2M-exposure corrected-binding Transformer control still failed | Repair controlled learnability before more AFM training. Keep local-mixer efficiency separate from unproved long-memory claims. |
| PGW | Stop scaling the old mechanism; hold V3 | Historical carry gain disappeared against reset/local controls; new positive control failed and auxiliary information is unmatched | Repair a sufficiently expressive learnable full-attention control and objective parity. Do not spend another large language budget until query-specific retrieval is demonstrated. |
| CHM | Stop large runs of current prototypes; allow cheap redesign | Broad64-entity training still failed all three DEVELOPMENT gates; addressing/copy and missing-fact rejection remain weak | Repair optimization and unseen-identity/second-hop/gate generalization. Require fresh independent evaluation and exact index/flat parity before scaling or claiming acceleration. |
| RLT | Continue state robustness cheaply; hold language scaling | Carry passed2/3 seeds at trained lengths; only 1/3 extrapolated through 1023; bias4 passed0/3. Published recurrence differs. | Fix seed/length reliability and separately reproduce the actual published feedback merge/layout. One causally verified retention model is not a general language win. |
| SI Original V2 | Pause full-system scaling; repair training first | Search value is untrained, replay/rollout states mismatch, and search reduced action accuracy; persistent pilot state was incompletely exported | Train value/dynamics/reward in consistent logged planning coordinates, export full persistent state, and test search against the same trained direct-policy system on factual action feedback. |

No architecture has demonstrated a breakthrough in this work. Conv7 and CPW's
local mixer have the strongest reasons for the next experiment. Repeating an
invalid control comparison or increasing size without fixing the diagnosed
failure would have low research value. The user ceilings are maxima, not a
requirement to spend2B tokens on every implementation regardless of validity.

For a later size increase, choose actual counted models <=100M, freeze each
training horizon and its learning-rate schedule, preserve unsuccessful attempts,
and report quality, total training cost, inference latency/memory and ablations.
Model size and token count alone do not establish frontier novelty or utility.
