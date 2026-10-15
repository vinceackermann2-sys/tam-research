# RLT BC — frozen fixed-update three-seed postmortem

Date: 2026-10-15. Completed one-shot CPU engineering run: https://github.com/vinceackermann2-sys/tam-research/actions/runs/38799067479 (first attempt, success). Evidence commit `6dd09e019a65ea307f9027f6aabd4f31b11a5b3c` on isolated RLT topic branch `exp/rlt-ax-bounded-transformer`. The original `main` and `exp/rlt-colab` branches were not changed by this panel. **Never rerun, resume, or amend BC.**

## Frozen resource and training conditions

Three fresh initialization seeds [20671081,20671082,20671083], the same examples at the same optimizer update per seed (data seed 20681081 easy, 20691081 long), 27,712-parameter carry RLT, chunk 8, AdamW, 2100 short easy updates followed by exactly 1200 full 128-token long updates, batch 8. Checkpoints at terminal step; heldout accuracy and binary NLL at milestone long update 400,800,1200. Heldout at every gap had 128 examples for class0 and 128 for class1, same data across seeds. All three passed short positive control (100% accuracy).

## Full terminal long-gap outcomes (accuracy at 1200 updates)

| Initialization seed | Gap16 overall / class0 / class1 | Gap64 overall / class0 / class1 | Gap127 overall / class0 / class1 | Preregistered pass |
| --- | --- | --- | --- | --- |
| 20671081 | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% / 100% | YES |
| 20671082 | 89.84% / 82.03% / 97.66% | 90.63% / 82.81% / 98.44% | 100% / 100% / 100% | NO |
| 20671083 | 52.34% / 7.81% / 96.88% | 49.22% / 0.78% / 97.66% | 51.56% / 3.91% / 99.22% | NO |

Pass gate: each class independently >=95% correct at **each** gap 16,64,127 after 1200 long updates; seed 20671082 cannot pass on overall or class0 at gaps16/64; seed 20671083 nearly collapses to class1 at all long gaps.

Intermediate outcomes matter, not merely terminal scores:

- Seed 20671081: at only 400 long updates, already 100% overall at gaps16,64,127 and stayed 100% at steps800,1200.
- Seed 20671082: gap127 improved from 50% (step400) to 100% (800) and stayed 100%, but gaps16/64 plateaued below the preregistered 95% per-class threshold. This demonstrates nonmonotonic horizon-wise learning: passing a longer gap does not imply passing every shorter one.
- Seed 20671083: gaps16/64/127 remained near chance throughout all three milestones; despite consistently strong short-gap scores. At step400 it favored class0; at step800 and 1200 it instead favored class1 on the long gaps. The near-chance overall averages conceal this severe class bias.

All three per-case JSON files and **all three 120,530-byte .pt checkpoints** were committed on the evidence branch; SHA-256 hashes are recorded per case in the immutable terminal JSON. The Actions run also retained an independent 90-day backup artifact. Remote-checkpoint validation step succeeded. Do not confuse a successful Actions workflow with all model seeds passing: the workflow succeeded but the scientific engineering gate passed **1 of 3**.

## Research judgment and next experiment

This is **direct evidence of initialization/optimization sensitivity** of this tiny recurrent model under identical optimization updates, task examples and evaluation conditions. It reduces confidence that BA's positive synthetic result reflects reliably trainable long-horizon retention. The outcome is still **INCONCLUSIVE for general language breakthrough**: three seeds do not establish a population success rate, and this is not a strong comparable language-model Transformer benchmark. Historical 15M held-out language NLL favored a Transformer.

Next permitted zero-GPU step: load each of the three *already saved BC weights* and measure class-balanced counterfactual state divergences, gate/direct retention and readout-logit differences at gaps16/64/127 on **fresh, independently seeded, label-balanced inputs**, without new training. This distinguishes memory write/decay, retrieval/readout and pure class-bias failure. Report all three failures/successes, no selecting only seed20671081. Before changing any parameters or launching new training, predeclare a separate controlled intervention and stronger fair full-context or recurrent-Transformer language baseline.

Research must stay in reviewable topic branches and avoid `main`, frozen consumed attempts, GPU/Colab/Modal credits, scientific replication seeds58232/58233, and 250M/5B scaling. No breakthrough claim.
