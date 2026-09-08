# CORTEX-S v0 — separate architecture research project

CORTEX-S is an experimental architecture track kept separate from TAM/AERA. The name expands to **Continual, Organised, Recurrent, Temporal, Experience-based eXecutive — Safe**.

This folder is intentionally self-contained so results, seeds, trigger namespaces, checkpoints, and claims cannot be confused with other architecture projects in this repository.

## Hypothesis

A useful successor to a purely autoregressive Transformer may need five properties that are native to brains and only bolted onto current LLM systems:

1. persistent recurrent state across interactions;
2. conditional/sparse internal computation;
3. explicit fast episodic memory that can update without rewriting base weights;
4. learned state/world dynamics rather than only next-token statistics;
5. an external deterministic authority boundary that the learner cannot modify.

The v0 code tests whether these mechanisms are coherent and trainable. It does **not** claim SSI, AGI, consciousness, alignment, or a breakthrough.

## No-credit first-principles gate

Before any Modal/GPU run, v0 must pass CPU-only checks:

- finite forward/backward gradients;
- persistent state actually changes future computation;
- router emits exactly `top_k` expert choices;
- fast memory stores/retrieves without modifying model weights;
- deterministic safety kernel rejects unauthorized capability/amount/target requests;
- tiny CORTEX-S and Transformer are within 2% parameter count;
- on the exploratory state-transition task, CORTEX-S must learn the recurrent rule and extrapolate to sequence length 32.

Run:

```bash
cd architectures/cortex_s
PYTHONPATH=. pytest -q tests/test_cpu_primitives.py
PYTHONPATH=. python state_machine.py --seed 123 --steps 600 --output exploratory_cpu_result.json
```

Seed `123` is explicitly **consumed exploratory engineering data**. It may never be reused as fresh scientific evidence. Seed `124` was also consumed during exploratory tuning and is recorded in the preregistration.

## Important systems caveat

`SparseExpertBlock` currently computes every expert and gathers the top-k outputs. That validates routing semantics but is **not true sparse GPU execution**. A paid systems experiment must implement actual sparse dispatch before throughput or efficiency can support a claim.

## Progression rule

CPU PASS authorizes only preparation of a separately preregistered paid seed-1 harness. It does not by itself authorize a breakthrough claim. The frozen scientific protocol is in `PREREGISTRATION.md`.
