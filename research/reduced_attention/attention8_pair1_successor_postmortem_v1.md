# Attention-8 v2 Pair-1 successor: zero-GPU postmortem (v1)

**Classification:** \`SCIENTIFIC_ATTENTION8_V2_PAIR1_SUCCESSOR_SCREEN_FAIL\` — one completed, single-seed, preregistered 100M/2B scientific screen. **No breakthrough, replication, retry, 250M/5B, or further paid GPU authorization.** Seed \`60232\` is durably consumed.

## Frozen, independently verified evidence

- Authorized source: \`27a3a15cb8456523798163f4cfa5a829844477ed\`, tree \`027e0c1bbc845783d547f935f378889d298c9dab\`, original CI-tested harness \`5d9e1a170d9bbb93e4a02de6a47467049df10d55\`.
- Scientific first attempt: https://github.com/vinceackermann2-sys/tam-research/actions/runs/37748939335 ; issue https://github.com/vinceackermann2-sys/tam-research/issues/1300
- Independent CPU-only/read-only **primary Modal** volume snapshot: https://github.com/vinceackermann2-sys/tam-research/issues/1331 ; observer run https://github.com/vinceackermann2-sys/tam-research/actions/runs/37826511643
- Durable terminal \`RESULT.json\` SHA256: \`872daab1193742942ef01fd8dee77080a07272eee4b55aa0b9236cffb584d88d\`. Observer confirmed reservation, H100 starts, both COMPLETE model results, and terminal comparison. The **snapshot did not consume a new seed, write the volume, or allocate GPU**.
- \`attention8_pair1_successor_frozen_metrics_v1.json\` transcribes the frozen final and periodic metrics from the first-attempt Actions log; do not confuse this transcription with the authoritative Modal RESULT. CI tests check internal arithmetic and the pinned evidence IDs.

### Terminal preregistered comparison

| Metric | Fresh Transformer | Attention-8 dense v2 |
|---|---:|---:|
| Parameters | 101,803,520 | 101,795,328 |
| Attention layers (of 24) | 24 | 8: 3,6,9,12,15,18,21,24 |
| Training-token exposures | 2,000,027,648 | 2,000,027,648 |
| Optimizer steps | 30,518 | 30,518 |
| Final validation NLL | **2.722934193611145** | 2.7680209398269655 |
| Training tokens/s | 308,546.9337118847 | **331,686.4231479331** |
| Measured training seconds | 6,482.085639093039 | 6,029.874931323257 |
| Peak allocated VRAM (GiB) | 44.37207317352295 | 42.911231994628906 |

- Candidate-minus-Transformer **NLL = +0.04508674621582065**. Preregistered limit \`<= +0.015\`: **FAIL** (approximately 3.006 times the allowed positive delta).
- Candidate/Transformer training **throughput = 1.074995039353253** (+7.50%). Preregistered minimum \`>= 1.03\`: **PASS**.
- Combined progression gate **FAIL**. Runtime and VRAM refer to this particular H100/batch/compile setup, **not** general inference efficiency or production latency.
- Both models finished on **H100 80GB**, same 2B-exposure protocol and evaluation seed per matched phase. This is a **single scientific seed**, not a replicated estimate.

### Learning-curve analysis (descriptive, not a separate gate)

| Token exposures | Transformer periodic NLL | Attention-8 periodic NLL | Candidate minus baseline |
|---:|---:|---:|---:|
| 200,015,872 | 3.518708 | 3.525339 | +0.006631 |
| 400,031,744 | 3.181220 | 3.215852 | +0.034632 |
| 600,047,616 | 3.033336 | 3.075570 | +0.042235 |
| 800,063,488 | 2.946733 | 2.992791 | +0.046058 |
| 1,000,013,824 | 2.884781 | 2.930099 | +0.045318 |
| 1,200,029,696 | 2.837151 | 2.882313 | +0.045162 |
| 1,400,045,568 | 2.795453 | 2.841174 | +0.045722 |
| 1,600,061,440 | 2.765573 | 2.810107 | +0.044534 |
| 1,800,011,776 | 2.745483 | 2.790610 | +0.045127 |
| 2,000,027,648 | 2.734943 | 2.778247 | +0.043304 |

- Attention-8 is worse at **all 10 recorded periodic checkpoints**. The gap starts at +0.00663 near 200M, increases to +0.03463 by 400M and about +0.04606 near 800M, then stays around +0.043–0.046 through 2B; **there is no observed closing to within the +0.015 terminal threshold**.
- At 200M, periodic NLLs used the checkpoint-evaluation protocol. The final 2B evaluation uses a distinct, frozen final-evaluation seed; its exact gap is therefore not required to equal the 2B periodic gap (+0.043304455).
- The 200M *engineering* panel had ΔNLL \`−0.01449833869934114\` and TPS ratio \`1.0651255526571075\`. **Do not treat its early result as a replication or a controlled learning-curve point**: engineering seed \`2026092901\` differs from scientific \`60232\`, and the 200M panel trained with a 3,052-step cosine schedule that **ended** at 200M; the 2B pair used a 30,518-step schedule still in progress at 200M. The metrics are informative but not apples-to-apples.

## Mechanism candidates (hypotheses, not established causes)

1. **Fewer cross-token mixing opportunities:** frozen candidate uses attention in 8/24 blocks, leaving 16 FFN-only blocks (GELU is tokenwise). The baseline has attention in every block. Increasing FFN width from 2,048 to 2,731 preserves parameter count but does not preserve per-layer sequence-interaction capacity. **Plausible main bottleneck**, not causally isolated by this matched pair.
2. **Scheduled information flow:** first candidate attention occurs at layer 3; there are two FFN-only layers before any attention, and two FFN-only blocks between successive attention layers. The exact location/depth sensitivity is **not identified** from this one schedule.
3. **Budget/schedule mismatch with engineering:** engineering selected a fast 200M candidate using a different seed and shorter learning-rate schedule; the 2B phase falsifies its preregistered quality progression hypothesis, not the feasibility of other sparse-attention schedules. It does **not** distinguish structural loss from optimizer/horizon interaction.
4. **Fairness boundaries:** widths and parameter counts are close and the paired runs match exposure, dataset, optimizer settings, batch/sequence length and GPU class. The single seed cannot characterize variability or guarantee gains under other datasets, sequence lengths, inference setups or sizes.

## Decision / next safe work

- **Freeze** scientific attempt #1300 and seed \`60232\`: never rerun, resume or modify the evidence.
- **Do not** advance attention-8 v2 \`[3,6,9,12,15,18,21,24]\` to 250M/5B, replication, or breakthrough communications based on this screen.
- For a distinct future idea, draft a **new** zero-GPU, source-isolated hypothesis comparing attention placement or a cheap causal token mixer between sparse-attention blocks. Pre-register separate quality and systems gates, train-horizon assumptions, and seed governance before **requesting explicit GPU authorization**. This report grants **none**.
- Preserve engineering-vs-scientific discrepancies as experimental-design warnings; if proposing a 200M predictor, ensure its optimizer schedule matches the corresponding prefix of the longer training program.

## Static provenance for future audits

The unchanged experiment architecture is \`architectures/cortex_s/reduced_attention_200m_panel_v2.py\`, frozen candidate is \`architectures/cortex_s/reduced_attention_100m_v2_attention8.py\`, scientific successor protocol is \`architectures/cortex_s/reduced_attention_100m_v2_attention8_successor_protocol.py\`, and H100 runner is \`modal_cortex_reduced_attention_v2_attention8_pair1_successor_100m_2b_v1.py\`, all at authorized source commit. No experiment code or Modal volume files are modified by this postmortem.
