# CPW v3 — Sparse World-State

CPW-v2 sequence-only established a strong small-model language Pareto result, but its
one-step mixer gives a hard depth-limited context horizon. CPW-v3 preserves the same
parameter count and replaces the SEQUENCE predictor with the recurrent WORLD predictor
in only a few layers.

The design is intentionally minimal:

- most layers: previous-latent-state SEQUENCE predictor;
- sparse layers: causal recurrent WORLD state;
- one predictor per block;
- no self-attention;
- no learned router;
- no auxiliary latent loss;
- no intrinsic reward/emotion/desire system.

At d_model=256, both predictor types use exactly 49,152 parameters per block, so changing
a layer from SEQUENCE to WORLD does not increase model parameters.

Frozen panel arms:

- sequence_only: no recurrent layers
- world_last1: WORLD at layer 14
- world_2: WORLD at layers 7 and 14
- world_3: WORLD at layers 4, 9 and 14

The hypothesis is that sparse recurrent layers provide a full-prefix causal state path
while the majority fast local layers preserve CPW-v2's throughput advantage. This panel
is a development/falsification experiment, not a breakthrough claim.
