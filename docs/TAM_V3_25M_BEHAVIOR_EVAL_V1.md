# TAM v3 25M Full — Behavioral Evaluation v1

Date: 2026-08-21

This document records the first direct behavioral test of the completed `TAM-v3-25M-Full` lineage. The purpose is to distinguish language-loss / post-training metrics from actual user-visible capability.

## Models compared

Same TAM v3 architecture and seed 7400 at three stages:

1. **Base** — 24.94M parameter TAM v3 after 300M FineWeb-Edu pretraining tokens.
2. **SFT** — Base checkpoint after instruction SFT.
3. **DPO** — SFT checkpoint after preference optimization; final checkpoint `final_dpo.pt`.

All checkpoints are stored on Modal Volume `tam-research-data` under `/vol/full-model-runs/...`.

## Evaluation design

Evaluation was CPU-only; no GPU was allocated.

Two complementary probes were used:

- **12 four-way forced-choice tasks**, scored using normalized continuation log probability. Categories included arithmetic, basic knowledge, simple transitive/logical reasoning, language, pattern completion, and basic Python knowledge.
- **Five greedy free-form prompts** testing strict instruction following, arithmetic, elementary coding, short explanation, and structured list generation.

Full machine-readable result:

`/vol/full-model-runs/TAM-v3-25M-Full-seed7400/behavior_eval_v1.json`

GitHub issue callback log: issue #95. Saved-output report: issue #97.

## Forced-choice result

| Stage | Correct | Accuracy |
|---|---:|---:|
| Base | 4 / 12 | 33.3% |
| SFT | 4 / 12 | 33.3% |
| DPO | 3 / 12 | 25.0% |

Random-choice expectation for four-way questions is 25%. With only 12 items this is a tiny diagnostic, not a statistically strong benchmark. The result provides no evidence that this 25M model has robust arithmetic, reasoning, coding, or general-knowledge capability.

## Representative free-form outputs

### Strict instruction: “Reply with exactly the word OK and nothing else.”

- Base: `The word right is the word right.`
- SFT: `The word right is "The word right`
- DPO: `The word right, "The word right`

None followed the instruction.

### Arithmetic: “What is 17 * 6? Answer with only the number.”

- Base: `What is the number of numbers?\nThe number of numbers`
- SFT: `The number of words in the English language is the number of`
- DPO: `The number of words in the English language is 18.`

None produced the correct answer, 102.

### Coding: “Write a short Python function add(a, b) that returns their sum.”

- Base: repetitive prose about creating code.
- SFT: repetitive prose about a “Python function,” without a valid function.
- DPO: prose saying a Python function “uses a function to generate a sum,” without valid code.

No stage produced a usable `def add(a, b): return a + b`-style solution.

### One-sentence sky explanation

- Base: repetitive `The Earth is the Earth’s surface.`
- SFT: `The sky is a beautiful and beautiful world...`
- DPO: collapsed into repeated `Earth's` tokens.

No stage gave the Rayleigh-scattering explanation requested.

## Interpretation

This model is **not a useful chat, reasoning, coding, or agent model**. It can generate locally English-like text, but it remains highly repetitive and unreliable on very simple tasks.

That negative result is consistent with the scale/training regime rather than, by itself, falsifying TAM v3:

- only ~25M parameters,
- only 300M pretraining tokens,
- no dedicated reasoning/coding corpus,
- small post-training budget,
- post-training cannot create world knowledge or reasoning ability absent from the base representation.

The behavioral result also shows that lower SFT loss and >50% DPO preference accuracy are **not sufficient evidence of useful capability**. In this run, SFT/DPO changed the model distribution but did not improve the small forced-choice benchmark; DPO was slightly worse there.

The raw language tradeoff was already visible: FineWeb NLL moved from 4.4131 at the pretrained base to 4.6162 after post-training. This is acceptable only if post-training buys meaningful instruction behavior; this first behavioral test does not show such a benefit.

## Scientific conclusion

Keep the architecture/scaling claims separate from this small-model behavior test:

- TAM v3 remains promising on matched **language-loss scaling experiments** at 25M and 50M.
- This fully trained 25M checkpoint is **not evidence of useful intelligence**.
- Do not call the model agent-capable.
- Do not use SFT/DPO metrics alone as a proxy for downstream competence.

## Next useful tests

1. Run the same post-training and behavioral suite on a **parameter/data-matched Transformer** to measure whether TAM itself affects downstream capability.
2. If compute becomes available, train at a larger scale/data budget before spending heavily on preference optimization.
3. Add established downstream benchmarks and larger task sets only once the base model is above chance on simple diagnostics.
4. Preserve all negative results; architecture selection should be based on quality/compute/capability, not only favorable loss curves.
