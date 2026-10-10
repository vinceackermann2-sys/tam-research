# AY stage gate — negative fresh-seed outcome and missing checkpoint postmortem

Date: 2026-10-10. Scope: isolated RLT AX research topic, no direct write to `main` or `exp/rlt-colab`.

## Terminal attempt and provenance

AY one-shot CPU engineering comparison is **complete and consumed**. GitHub Actions [run 38073108161](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38073108161) completed successfully on trigger `633ad44b3e94b9fed2e75ffdf8fa806c7ed4798f`. Frozen result: `experiments/rlt/cpu/results/rlt-bounded-threeway-20261010-ay.json` at final evidence commit `2460bd409fdfaaf9aa5c639f0d916d227fde2d18`. The reserved job `rlt-cpu-bounded-threeway-20261010-ay` and original claim remain immutable; do **not rerun, resume, amend, or reuse** either.

All models used 128-token balanced binary last-write classification (256 held-out examples per gap), 15-second easy and 90-second long-phase CPU training budgets, new AY initialization/data seeds, and matched AdamW settings. 27,712 RLT parameters vs 27,680 Transformer parameters. These are equal **time**, not equal tokens, active FLOPs, inference budget, or architecture.

## Held-out accuracy

| Last-write distance | Carry RLT | Reset RLT | Two-layer sliding-window Transformer |
| --- | ---: | ---: | ---: |
| 1 | 97.656% | 96.484% | 100% |
| 4 | 100% | 80.469% | 100% |
| 16 | 50% | 51.563% | 50% |
| 64 | 50% | 50.391% | 50% |
| 127 | 50% | 45.313% | 50% |

Carry-RLT predicts **class 1 for every held-out example** at gaps 16, 64 and 127 (class-0 accuracy 0%; class-1 accuracy 100%), and has worse-than-chance binary NLLs 0.74875, 0.74668, 0.74248. This is a *negative* learned-persistent-memory outcome for AY, not evidence of successful long-horizon recall. AY reset likewise hovers near chance; the Transformer at these gaps was **provably information-limited** (two 8-token layers => a 14-token causal receptive distance). Its 50% is not an optimization failure. At gap 4 it achieved 100% near-perfect NLL (~0.000166).

The prior AW carry run attained 92.188% at 16 and 94.141% at 64, compared with AY 50% at both. AW and AY are different engineering protocols and seeds: this is strong concern about **seed/protocol sensitivity**, *not* a controlled scientific replication or a measured failure probability.

## Throughput and direct retention

AY long-phase throughput: carry 15,250.6 tokens/s (1,373,184 tokens, 1,341 steps); reset 15,669.9 (1,411,072 tokens, 1,378 steps); sliding Transformer 90,075.4 (8,107,008 tokens, 7,917 steps). The Transformer was ~5.91x faster than carry in this CPU implementation and learned the in-window task much more efficiently.

The runner logged trained-gate mean *direct* retention over the queried gap. By class the values were about 0.294/0.304 at 16, 0.079/0.082 at 64, and 0.0296/0.0292 at 127. Gate products suggest decaying direct memory pathways, but do **not** establish sole causation for the learned readout's failure. Hidden-state/counterfactual attribution needs actual trained weights.

## Forensic checkpoint preservation failure

The runner wrote and SHA-256-verified `carry.pt`, `reset.pt`, and `sliding_kv.pt` in the temporary GitHub Actions runner, but the repository's root `.gitignore` contains `*.pt`. The workflow executed `git add "$RESULT" "$CHECKPOINT"` without forcing ignored checkpoint addition. The final commit contains **only AY result JSON**, not checkpoint blobs. GitHub Actions run 38073108161 reports **zero artifacts** and no uploaded checkpoint archive. The result JSON's checkpoint paths/SHA-256 values are historical *runner-local* records, not proof of durable retention.

This is **evidence-preservation failure**, despite a successful workflow. The metrics/terminal provenance remain verified, but trained-weight tests cannot be recovered from the recorded sources. Do not suggest or perform AY replay. Maintain AY as consumed and annotate the missing-weight limitation.

## Next research gate (zero GPU)

1. Freeze AY postmortem with a separately triggered read-only AZ correctness audit; preserve failed/successful evidence, avoid re-triggering AY.
2. Before *any future* fresh engineering training attempt, add a fail-closed checkpoint retention mechanism: either `actions/upload-artifact@v4` with artifact ID/read-back and checksums, or explicit `git add -f` for small trusted checkpoints plus verification that each checkpoint appears in the **remote** tree after push. Retain persistent record on failure, never silently green.
3. Add trained-state counterfactual tests only once fresh, separately authorized engineering weights are durably preserved; do not recreate AY's missing weights by replay.
4. Prioritize investigating instability and long-horizon state dynamics, then compare a competitive equal-token and equal-budget baseline. No scientific replication, GPU, 250M/5B, or breakthrough declaration is authorized.

**Status:** AY complete / negative for long-distance memory; AX correctness green; checkpoint retention incomplete; no claim of superiority over Transformers.
