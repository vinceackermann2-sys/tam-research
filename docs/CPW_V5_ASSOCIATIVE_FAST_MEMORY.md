# CPW-v5: Associative Fast Memory

Scientific preregistration: [#1296](https://github.com/vinceackermann2-sys/tam-research/issues/1296). This is an **experimental memory-repair track**, not a demonstrated breakthrough and not a conversational/chat-trained model.

## Evidence motivating the change

CPW-v2 `sequence_only` performed well on a short-context 25M-token language benchmark with lower loss, fewer parameters and less measured H100 compute than the repository Transformer. However, its local receptive field is about 15 tokens at the configured depth.

CPW-v3 `world_last1` made the graph recurrent across the whole prefix but did not learn useful key->value recall. In the frozen CPW-v4 #1275 task, Transformer scored 100% at distances 64, 128 and 256 while `world_last1` scored about 0.95% long-distance accuracy. The preregistered result was STOP, not a breakthrough.

## Computational mechanism

CPW-v5 replaces **one** 49,152-parameter `sequence_only` predictor with a 49,152-parameter Associative Fast Memory (AFM). Other blocks and embeddings are unchanged.

At position t, with normalized token representation x:

- read query q_t = ELU(Wq x_t) + 1
- write key k_t = ELU(Wk x_(t-1)) + 1
- write value v_t = Wv x_t
- write gate g_t = sigmoid(wp x_(t-1) + wc x_t - 2)
- exclusive state M_(<t) = sum_(j<t) g_j k_j outer v_j
- exclusive key mass Z_(<t) = sum_(j<t) g_j k_j
- read r_t = q_t^T M_(<t) / (q_t^T Z_(<t) + epsilon)
- output y_t = Wo r_t

**Read before write** is essential: the final query token cannot write an answer into memory before retrieving it. At input position zero writes are disabled.

Ranks: key=32, value=63, d_model=256. Projectors and two gate vectors total exactly 49,152 parameters. Full candidate remains 21,745,408 parameters; Transformer baseline has 24,940,288.

A single compressed state vector is replaced by a matrix of distinct key/value traces. Memory capacity and ability to *learn to gate filler* are still hypotheses. The positive ELU features and normalized linear read may cause interference among many keys; this has not yet been validated empirically.

## Scientific test

Reuse CPW-v4's exact deterministic, leakage-resistant generator: 8 freshly randomized pairs per example, final query marker + key, filler disjoint, source-to-query delays 32/64/128/256. Score only the query target.

Arms: `transformer`, `sequence_only`, `afm_first1`; use paired seeds and equal steps. All pass/fail thresholds, seeds and namespace are frozen in #1296.

First complete zero-GPU contracts (causality, parameter count, functional multiple-binding recall with *constructed*, not learned features, finite gradients, absence of future-token leakage). Then a fresh H100 smoke and, only on integrity pass, 3 paired replication seeds. No retries or checkpoint reuse.

## Claim restrictions

Even if AFM-v5 learns delayed recall, it has not proved better language capability, reasoning or speed. Next gates would be integration into the FineWeb 25M matched language screen, evidence across a second scale/corpus, multi-task context tests and generation/chat post-training. A single synthetic benchmark is not a breakthrough.

- `breakthrough_claim_allowed=false`
- `scale_up_authorized=false`
