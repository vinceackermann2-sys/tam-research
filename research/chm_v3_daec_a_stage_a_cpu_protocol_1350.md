# CHM-v3 DAEC-A Stage A: address-target alignment toy (#1350)

Status: **CPU-only engineering demonstration, not scientific evidence**.

## Historical immutable evidence

Original DAEC Stage-C #1323 is `CHM_V3_100M_DAEC_STAGE_C_STOP`, consumed scientific seed `2013161`. The archived run recorded 0/384 correct copied answer tokens even though all 384 answers existed in memory. The original trained-checkpoint hard-flat parity proof was missing. No archived checkpoint is retrained, replayed, reclassified or modified here. #1341/#1345 are independent forensic paths; this design makes no claims on their unresolved questions.

## Question

Can an explicit address-position supervision signal, on small unrelated synthetic relations, improve the *ability to route two hard memory reads* compared with the same randomly initialized address parameters left untrained? This is a toy engineering question, **not** whether DAEC-A beats a Transformer on language modeling.

## Frozen synthetic input and control

- Pure PyTorch CPU; no `modal`, CUDA execution or checkpoint loading; no original aligned-v4 GPT-2 prompts/templates, original seed or vocab. Distinct training-only vector episodes.
- Memory lengths: exactly 128, 256, 512, 1024 entries. Every episode has a relation key pointing to a document identifier and a record key pointing from the document to an output token.
- Keys and queries are different fixed orthonormal views of the same underlying random identifier. The model learns two 24x24 address projection matrices. The two projections are reused for both hops; no hidden-state memory residual.
- Relation/record role masks are provided as **external oracle metadata** in this toy. They are **not learned** from natural language. Never transfer toy success to 100M scientific capability.
- Overwrite cases (index mod 4 = 1) duplicate a document-key record with a stale older value and an authoritative newer value. Deterministic recency tie break, not a learned overwrite solution.
- Negative cases (index mod 5 = 0) use absent query identifiers with a fixed similarity-based no-match option (threshold 0.50).
- Synthetic split seeds: train `1350011`, development `1350101`, test `1350201`, disjoint example ID ranges. 256/64/64 maximum episodes; generator seed additionally binds index and memory length.
- Training objective: first-hop relation-position cross-entropy (or no-match for negative) + second-hop record-position cross-entropy on positive cases. Training uses **teacher-forced gold first-hop payload for the second loss only**; evaluation may use only its own hard-selected first-hop payload.
- Control: same exact randomly initialized address transforms with no optimization. Optimized model: AdamW weight_decay=0, learning_rate=0.025, 96 sequential synthetic episodes, CPU only. The control is **untrained addressing**, not a Transformer or LM-loss-only matched control.
- Hard evaluation: argmax relation vs fixed no-match logit followed by argmax document record with recency tie break; score held-out development and test splits separately. No gold pointer routed during evaluation. Report pointer top1 and answer top1 with case counts, negatives, overwrite controls.
- Explicit independently constructed perfect-projection oracle tests prove **accessibility**, not learned reasoning. Ordinary 100M NLL, language abilities, real relation resolution, true parity of old trained checkpoint, and generalization to new corpora are OUT OF SCOPE.

## Gates and controls

Success for this Stage A: deterministic, finite CPU training and independently reproducible address/answer/oracle tests, no leakage, no unauthorized CUDA/GPU, no use of original scientific seed. The toy metrics must be reported regardless of whether optimization improves. No minimum accuracy was preregistered as proof of a model breakthrough.

Run `python scripts/chm_v3_daec_a_stage_a_cpu_report_1350.py` for CPU-only toy scores, or run repo `pytest -q` for tests. Future use of an engineering GPU or another scientific run requires a **new preregistration and authority**, not this issue.

## Potential confounders

The fixed synthetic view transformations and oracle role masks simplify retrieval; latest-overwrite is a deterministic tie-break; 96 steps of pointer supervision can memorize artificial alignments. The test split uses fresh vectors but the same generative mechanism. This can show an implementation is trainable, not that the 100M architecture solves natural-language memory.
