# CPW v1 — Lean Predictive Core

CPW-v1 is the evidence-driven successor to CPW-v0.

The v0 mechanism panel (#1247) showed that the full CPW mixer beat the Transformer at equal tokens, but its auxiliary prediction loss, sparse router, and attention branch were not responsible for that gain. CPW-v1 therefore physically removes them.

Each block has three causal predictive systems:

1. **World predictor** — a compressed recurrent state updated from the current causal representation.
2. **Sequence predictor** — a low-rank predictor driven by the immediately preceding latent representation.
3. **Memory predictor** — a low-rank predictor driven by the prior running-prefix summary.

Their outputs are fused with a parameter-free arithmetic mean and broadcast through a shared recurrent latent workspace across depth.

There is:
- no self-attention;
- no learned expert router;
- no top-k selection;
- no auxiliary predictive loss;
- no persistent reward maximizer or biological drive.

At the frozen 25M-class geometry, CPW-v1 has 22,974,208 parameters versus 24,940,288 for the Transformer, a reduction of about 7.9%.

This is still an experimental architecture. A positive 25M screen is not a breakthrough claim.
