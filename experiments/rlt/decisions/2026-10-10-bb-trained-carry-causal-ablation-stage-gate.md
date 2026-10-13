# BB stage gate — trained BA recurrence is causally necessary on the bounded last-write task

Date: October 10, 2026. Branch: `exp/rlt-ax-bounded-transformer`. **Zero-GPU, inference only.** Do not touch `main` or the frozen `exp/rlt-colab` head.

## Immutable evidence

- BA was completed once: [workflow 38081374598](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38081374598), commit `742fba9f9852e517efaba96743d4d3e2bb3a6f94`, model checkpoint SHA-256 `74fe2bb1cfb74242aaba3d0eb8ac5df13c1a421a7d50054a0d7dbf503fb4149b`. Its `.pt` file was actually persisted and verified from the remote Git tree, unlike AY's missing weights.
- BB independently passed [checkpoint/ablation preflight 38081782859](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38081782859).
- BB terminal inference-only result: [workflow 38081891782](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38081891782), GitHub commit `2c69fcad388ba2f79c91353df19683820ce420b2`, JSON `experiments/rlt/cpu/results/rlt-ba-weight-ablation-20261010-bb.json`. The run trained **no** new model and used **no** GPU. All three variants loaded precisely BA's 27,712 original weights.

## BB preregistered intervention and independent data

Three exact-weight variants: (1) carry-RLT, (2) all-chunk recurrent state reset, (3) erase state only at time 120, before the final 8-token chunk. All retain the same chunk-local causal attention. Two fresh balanced evaluation seeds `20631081` and `20641081` yielded 512 heldout examples (256/class) **per model per gap**, over gaps 1, 4, 16, 64, 127. Counterfactual evaluation `20651081` tested 16 opposite-label pairs per off-chunk gap; paired inputs had identical final 8 tokens and their write-flip switched the oracle.

| SET gap (tokens) | Carry RLT | All-chunk reset | Erase only at t=120 |
| --- | ---: | ---: | ---: |
| 1 | 100% | 100% | 100% |
| 4 | 100% | 100% | 100% |
| 16 | **100%** | 50% | 50% |
| 64 | **100%** | 50% | 50% |
| 127 | **100%** | 50% | 50% |

The carry model's heldout binary NLL remained near 0.00027–0.00043. The erasure model's out-of-window NLL worsened to around 2.20–2.21 and always predicted class 0 (class0 100%, class1 0%) at 16, 64, 127.

For all three off-chunk gaps, the BA carry model classified **both members correctly in all 16 counterfactual pairs**. Its mean absolute difference in final binary logits between opposite-write pairs was 7.82 at gap16, 7.97 at gap64, and 7.89 at gap127. For either erasure or full reset the final binary logits were exactly equal for the opposite pairs (mean absolute difference zero), and both-correct fraction was zero. This is an explicit causal recurrent-state necessity test under the constrained attention architecture.

## What this actually establishes

The model can learn and use a differentiable fixed-width carried state as the **only permitted pathway** from older than one chunk to the current chunk, in this narrow binary SET/QUERY task. The late state intervention removes the capability without changing the trainable weights or locally accessible input. Inference carry state is fixed-size; full-BPTT training is **not** constant-memory.

These results do **not** establish a Transformer breakthrough. The 2-layer AX Transformer with an 8-token per-layer sliding window has a full receptive distance of 14 tokens, so it is mathematically excluded from using the off-window SET under the chosen task. A full-context, deeper, memory-augmented, or recurrent Transformer is not ruled out. BB tested only one **trained BA checkpoint**, not multiple independent training seeds or language generalization.

## Combined AW–BB research decision

- AW: Intermediate 16–64 memory positive; 127 near chance.
- AY: Fresh-seed carry model collapsed to chance for 16/64/127; trained checkpoint not persisted because `*.pt` was gitignored.
- BA: Independent CPU engineering seed learned 100% at 1/4/16/64/127 and now has a durably preserved checkpoint.
- BB: New heldout samples and targeted intervention show that BA's long-gap success really **depends on its carried recurrent state**, not leakage through chunk-local attention. It does **not** demonstrate a reproducible training success rate, as earlier experiments gave conflicting outcomes.
- Historical 15M FineWeb-Edu experiments AK–AQ did not beat Transformer on language NLL.

**Stage gate:** Mechanism verified for a trained tiny model; generality, optimization reliability and real-language competitiveness remain open. Do not scale to 250M/5B, consume protected replication seeds 58232/58233, spend Modal GPU, or make a breakthrough declaration. The next admissible work is preregistration and correctness preparation for an independent CPU *training-stability* panel (multiple fresh unique seeds and negative-result reporting) and richer inference-only tasks, with strict checkpoint persistence. Do not treat `continue` as permission for GPU scientific replication.
