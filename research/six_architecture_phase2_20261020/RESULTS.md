# Phase-two measured results

L4 training complete: False. See [L4_RESULTS.md](L4_RESULTS.md) for the separate paired trial.

The following language table belongs to the interrupted A100 trial. Candidate outcomes are unknown.

| Model | Parameters | Training exposures | Development NLL | Training tokens/s |
|---|---:|---:|---:|---:|
| transformer | 99,837,440 | 99,999,744 | 4.513550 | 100,177 |
| attention8 | 99,558,912 | 99,999,744 | 4.571129 | 113,396 |

Comparisons at the same exposure count are saved in LANGUAGE_COMPARISONS.json.
Intervals resample token blocks and are descriptive; a one-seed study is not a robust superiority result.

| SI seed | Direct policy | Native gated search before | Native gated after | Forced depth2 after | Reward MSE before/after |
|---|---:|---:|---:|---:|---:|
| 202610301 | 100.00% | 50.63% | 100.00% | 100.00% | 1.026459/0.001416 |
| 202610302 | 100.00% | 64.99% | 100.00% | 100.00% | 0.842179/0.002416 |
| 202610303 | 100.00% | 100.00% | 100.00% | 100.00% | 0.892922/0.001312 |

All SI non-head weights and direct logits are unchanged by the repair. This small finite cue task does not demonstrate an advantage over the direct policy.

SI post-hoc head restoration: the calibrated gate selects no search in all three repaired systems. Forced search still scores100%, so this is not merely hiding a bad branch.
Restoring the untrained transition reduces forced-search accuracy, while restoring the untrained reward or value head individually preserves100% in this task. The transition is the critical reliance found by these interventions; value-head training is not separately demonstrated to help this one-step problem. Coadaptation limits attribution.

| SI seed | Restored head | Forced depth2 accuracy | Fraction searched by calibrated gate |
|---|---|---:|---:|
| 202610301 | restore_transition | 24.46% | 0.00% |
| 202610301 | restore_reward_head | 100.00% | 0.00% |
| 202610301 | restore_value_head | 100.00% | 0.00% |
| 202610301 | restore_verifier | 100.00% | 49.37% |
| 202610302 | restore_transition | 5.57% | 0.00% |
| 202610302 | restore_reward_head | 100.00% | 0.00% |
| 202610302 | restore_value_head | 100.00% | 0.00% |
| 202610302 | restore_verifier | 100.00% | 48.54% |
| 202610303 | restore_transition | 50.63% | 0.00% |
| 202610303 | restore_reward_head | 100.00% | 0.00% |
| 202610303 | restore_value_head | 100.00% | 0.00% |
| 202610303 | restore_verifier | 100.00% | 26.66% |

| RLT seed | Training recipe | Accuracy at gaps127/255/511/1023 | All-length gate |
|---|---|---|
| 202610311 | native | 100.0%/100.0%/100.0%/50.0% | False |
| 202610312 | native | 100.0%/100.0%/100.0%/78.5% | False |
| 202610313 | native | 100.0%/100.0%/100.0%/100.0% | True |
| 202610311 | public_gate_prior | 100.0%/100.0%/96.5%/50.0% | False |
| 202610312 | public_gate_prior | 100.0%/100.0%/100.0%/100.0% | True |
| 202610313 | public_gate_prior | 99.2%/96.9%/73.4%/51.2% | False |

The RLT gate prior adds public instruction-type supervision; it is not an architecture-only gain. Carry erasure must score50% on every paired prefix panel.

PGW: the following traces quantify slot collapse in the original tiny hybrid checkpoints.

| Seed | Delay | Most-used slot write share | Target contribution to final query read |
|---|---:|---:|---:|
| 202610201 | 1 | 93.12% | 0.00879222 |
| 202610201 | 2 | 93.47% | 0.0068051 |
| 202610201 | 4 | 94.68% | 0.00514572 |
| 202610202 | 1 | 97.65% | 0.0120256 |
| 202610202 | 2 | 97.50% | 0.00996856 |
| 202610202 | 4 | 97.01% | 0.00775916 |
| 202610203 | 1 | 99.41% | 0.0037043 |
| 202610203 | 2 | 99.47% | 0.000672601 |
| 202610203 | 4 | 98.89% | 2.23336e-05 |

CHM: evaluator-gold source context is a failure-localization intervention, not model accuracy.

| Seed | Family | Native accuracy | Gold gate only | Gold gate and first-hop context |
|---|---|---:|---:|---:|
| 202610271 | direct | 19.53% | 20.31% | 20.31% |
| 202610271 | overwrite | 17.97% | 17.97% | 17.97% |
| 202610271 | two_hop | 10.16% | 10.16% | 10.16% |
| 202610272 | direct | 3.91% | 12.50% | 12.50% |
| 202610272 | overwrite | 21.88% | 21.88% | 21.88% |
| 202610272 | two_hop | 14.06% | 14.06% | 14.06% |
| 202610273 | direct | 3.12% | 3.91% | 3.91% |
| 202610273 | overwrite | 16.41% | 16.41% | 17.19% |
| 202610273 | two_hop | 5.47% | 5.47% | 5.47% |

Frozen RLT state readout: additional supervised long-context linear-head training, not unmodified-model extrapolation. Composite27,778parameters;3,031,040 additional positions plus1,363,200 backbone pretraining positions.

| Seed | Base recipe | Probe accuracy at128/256/512/1024 | New length/overwrite gate |
|---|---|---|---|
| 202610311 | native | 100.00%/100.00%/100.00%/100.00% | False |
| 202610312 | native | 100.00%/100.00%/100.00%/100.00% | False |
| 202610313 | native | 100.00%/100.00%/100.00%/100.00% | False |
| 202610311 | public_gate_prior | 100.00%/100.00%/100.00%/100.00% | False |
| 202610312 | public_gate_prior | 100.00%/100.00%/100.00%/100.00% | False |
| 202610313 | public_gate_prior | 100.00%/99.41%/99.02%/93.55% | False |

All three native-state probes decode the constrained first-SET task through1024 perfectly, with erased carry50%. All OOD follow-up gates fail: extrapolation to4096 is near chance and many last-write gaps fail. This identifies recoverable information without establishing a general readout fix.

SI native memory revision and persistence: frozen trained representations, no optimizer updates; association correctness is not action utility. Each checkpoint receives56 actual fixture feedback events/616 raw positions, on top of736,000 prior positions.

| SI seed | A after learning B/C | A after8 revised outcomes | Native replay restored | Explicit full replay restored |
|---|---:|---:|---:|---:|
| 202610301 | 100% | 50% | 0/56 | 56 |
| 202610302 | 100% | 50% | 0/56 | 56 |
| 202610303 | 100% | 50% | 0/56 | 56 |

All eight revised A associations return old deltas in every seed; the eight unchanged associations remain correct. Native append-only memory retains duplicate keys, and cosine argmax chooses the first tied key. The same-feedback latest-record dictionary returns all current deltas by construction. No capacity overflow occurs. World A logits are bitwise unchanged after B/C feedback, and native weight+fast/memory import preserves outputs exactly, but native export omits replay. The separate diagnostic checkpoint saves/restores all56 replay records. These identify changing-fact and lifetime-checkpoint repair targets, without establishing continual decision improvement or family-wide failure. See SI_MEMORY_REVISION_PROTOCOL.md and CONTINUAL_LEARNING_STATUS.md.
