# CPW v4 — delayed associative recall

Scientific preregistration: issue #1275.

## Purpose

CPW-v2 sequence-only produced a strong 25M-token language Pareto result but has a hard temporal dependency horizon of roughly 15 tokens. CPW-v3 `world_last1` restores a full-prefix causal path with one recurrent WORLD layer and retained a 3M-token Pareto signal.

This benchmark asks whether that recurrent path can learn **usable associative memory**, rather than merely creating mathematical connectivity.

## Task

Every example independently samples eight unique keys and eight unique values and creates a fresh random bijection. The model sees all eight bindings, then random filler, then `QUERY, key`. Loss is applied only to the next-token prediction after the query key.

The queried source value is placed exactly 32, 64, 128, or 256 tokens before the query key. Key, value, filler, and structural vocabularies are disjoint. Because mappings are resampled for every example, the answer cannot be learned as a fixed key-to-value relation in model weights.

## Arms

- Transformer: 24,940,288 parameters.
- sequence_only: frozen CPW-v3 reference, 21,745,408 parameters.
- world_last1: frozen selected CPW-v3 successor, 21,745,408 parameters, recurrent WORLD at layer 14 only.

No architecture tuning is permitted from benchmark results.

## Controls

Zero-GPU contracts verify:

- deterministic generator replay;
- unique bindings and exact requested delay;
- queried target occurs exactly once in the input and never after its source;
- balanced training delay mixture;
- exact frozen parameter counts;
- no attention/router/memory-prefix module in CPW arms;
- sequence-only final logits cannot depend on a token >15 positions away;
- world_last1 final logits can;
- strict future-token causality;
- finite forward/backward.

## Scientific gate

The complete thresholds are frozen in issue #1275. Transformer must first establish task learnability. If it fails, the benchmark is undertrained/invalid rather than negative CPW evidence. If sequence-only performs well above chance, treat that as a leakage/integrity alarm.

A positive result permits only a proposal for a 25M language confirmation of the exact world_last1 successor. It is not a breakthrough claim.
