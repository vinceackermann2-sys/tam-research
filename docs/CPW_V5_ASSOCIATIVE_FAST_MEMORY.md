# CPW-v5 — Associative Fast Memory (AFM)

CPW-v4 falsified the sparse WORLD vector state as a practical long-range
associative-memory mechanism: the Transformer learned random delayed
key/value retrieval perfectly while WORLD_LAST1 remained at chance.

CPW-v5 tests a different computational primitive. Fourteen blocks retain the
cheap one-step CPW sequence predictor. The final block uses a rank-32
associative fast-weight matrix.

For hidden state (x_t), the AFM uses one shared key/query projection,
a value projection, and a learned generic write gate. A write associates the
previous token's key feature with the current token's value feature. The
fixed-size memory matrix accumulates those bindings causally. The current
token queries that matrix through the same key projection.

This is attention-free in the usual quadratic sense. For fixed rank, memory
state size is constant with context length and scan work grows linearly with
sequence length.

The first scientific gate is deliberately narrow: the exact delayed
associative-recall task that WORLD_LAST1 failed. Passing that gate permits a
new proposal for a fresh-seed language Pareto panel; it does not authorize
25M language training or a breakthrough claim.
