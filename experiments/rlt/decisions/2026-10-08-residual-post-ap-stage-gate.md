# Residual Gated-Scan RLT: post-AP scientific stage gate

**Decision: freeze the evaluated 15.1M residual variant as a stronger RLT baseline, but NOT a Transformer-beating language model. No breakthrough, automatic replication or 250M/5B scaling.**

Branch: `exp/rlt-colab`. No changes to `main`. Stage-gate scope: 15,129,344 parameters, FineWeb-Edu D, GPT-2 tokenizer, sequence length 64, batch 64, matched A100-80GB, postcompilation 60-second training windows. The 6M train and 500k validation source token counts define the pinned input corpus; timed training may reuse tokens.

## Frozen evidence: non-equivalent architecture comparisons

| Run | Classification | RLT model and final NLL | Transformer final NLL | RLT − Transformer |
| --- | --- | --- | ---: | ---: |
| AK (engineering) | Gated scan, constant LR 1e-3 | gated scan **5.672780** | 5.531180 | **+0.141600** |
| AM (fresh-seed science) | Gated scan, constant LR 1e-3 | gated scan **5.736319** | 5.586229 | **+0.150090** |
| AN (engineering) | Direct residual gated scan, constant LR 1e-3 | residual scan **5.620681**; gated scan 5.678387 | 5.524289 | **+0.096392** |
| AO (engineering tuning) | Residual only; 4 predeclared optimizer candidates | best cosine LR 1e-3 **5.601719** | not run | not comparable |
| AP (fresh-seed science) | Residual scan and Transformer, both cosine LR 1e-3 | residual scan **5.554478** | 5.522571 | **+0.031907** |

**Do not subtract losses across rows:** AK, AM, AN, AO and AP use distinct seeds. The direct AN *within-run* comparison is valid for that engineering seed: residual NLL is **0.057706 lower** than gated scan, with nearly the same throughput (282,626 vs 283,346 tokens/sec). AO *within-run* tuned residual optimizer to cosine LR 1e-3 over constant LR 3e-4, 1e-3, 3e-3; cosine achieved 5.601719 and constant 1e-3 5.636570. AO is optimizer selection, not comparative scientific evidence.

AP is the first separate scientific result with the new architecture and frozen AO optimizer. Verified results:

- Residual Gated-Scan RLT: **5.5544784814 NLL**, **304,104.69 tokens/sec**, 18,251,776 tokens seen, 60.0181 seconds training, 15,129,344 parameters.
- Transformer: **5.5225712508 NLL**, **334,177.73 tokens/sec**, 20,054,016 tokens seen, 60.0100 seconds training, 15,129,344 parameters.
- Residual minus Transformer NLL: **+0.0319072306**. Residual training throughput is **91.00%** of Transformer. Both have cosine LR 1e-3. Both use matched data and evaluation seeds and identical held-out eval token count 131,072. Source hashes match committed claims.
- Classification: `SCIENTIFIC_RESIDUAL_EQUALTIME_SINGLE_SEED_NO_QUALITY_ADVANTAGE`.
- AP GitHub workflow: https://github.com/vinceackermann2-sys/tam-research/actions/runs/37842900054.

The cross-experiment reduction from the AM quality gap (+0.150090) to the AP quality gap (+0.031907) is **suggestive** but **not a controlled cross-seed causal estimate**. The credible within-run architectural improvement is AN's 0.057706 lower NLL versus ordinary gated scan. AP remains worse than Transformer on the specified matched quality-per-second metric.

## Frozen result paths

- `experiments/rlt/modal/results/rlt-systems-gated-scan-realdata-modal-20261008-ak.json`
- `experiments/rlt/modal/results/rlt-systems-gated-scan-optimizer-qualitysec-modal-20261008-al.json`
- `experiments/rlt/modal/results/rlt-gated-scan-compute-matched-scientific-scale15m-60s-modal-20261008-am.json`
- `experiments/rlt/modal/results/rlt-systems-gated-scan-residual-realdata-modal-20261008-an.json`
- `experiments/rlt/modal/results/rlt-systems-residual-optimizer-qualitysec-modal-20261008-ao.json`
- `experiments/rlt/modal/results/rlt-residual-gated-scan-scientific-scale15m-60s-modal-20261008-ap.json`

## Implications

1. Parallel associative recurrence is feasible and materially faster than serial recurrence. The direct encoder residual preserves information that scan-only decoder initialization appears to discard under the tested setup.
2. The residual architecture is parameter neutral and passes 18 CPU scan/math/gradient/causality/compilation tests, not yet equivalent to Transformer in language NLL at 60s equal-GPU-time.
3. Stop further optimizer sweeps at this scale without a new preregistered hypothesis. The AN and AP outcomes do not justify a breakthrough claim, scientific replication using protected seeds, or 250M/5B scaling.
4. Investigate, **zero-GPU first**, whether direct/scan pathways can be combined adaptively **without adding parameters** and whether sequential memory improves explicit state-tracking tasks beyond short-context perplexity. Candidate mechanism: reuse the existing scan gate to choose between the local causal encoder state and the recurrent state; freeze gate semantics and compare prefix-causal gradients before any paid test.
5. A future paid screen would require a *new* job ID, source hashes, unused seeds, pre-registered model/optimizer/data/metric/timing, independent prereq CPU tests, claim before GPU, no auto retry. Positive results would still require a separately authorized replication. Long-context tests require equal effective context and equal compute rather than one model receiving more information.

## Governance

No mutation to `main`. Keep AK–AP claims and terminal records immutable. Do not reuse any consumed seed, rerun a claimed workflow, or infer scientific evidence from infrastructure failures. Record both negative and positive evidence. These notes grant **no** new GPU run, replication or scale authority.
