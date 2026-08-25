# AERA-v24 VCEM implementation boundary

This note records the implementation boundary frozen by issue #347 before any controlled result.

- Base: main `a6942b375942f512e9d0221a786d636702fd7f73`.
- No GPU, production corpus, real-language seed, architecture freeze, S2, independent replication, 100M, or breakthrough claim is authorized by this implementation.
- V23 routing, experts, recurrent stream, chunking, q/k/v/out dimensions, pair-write gate parameter count, and K(C)=min(C,max(2,ceil(C/16))) physical write budget are preserved.
- V24 memory is a 48-slot/stage causal contextual episodic KV state. Real-language shape (4 stages, memory_dim=50, float32 keys/values/strengths plus bool validity) is 77,760 bytes/session.
- Context is `h[t] + mean(previous up to 8 normalized events)` and contains no future event.
- Stored keys are normalized k(context); stored values are tanh(v(next-context)); effective strengths are inherited write strengths.
- Selected entries are encoded in parallel. Newest near-duplicate contextual keys (cosine >=0.95) win within the incoming block and against prior state. No selected-candidate Python recurrence and no M/P Sherman-Morrison state remain.
- Retrieval score is `(cosine(q(context), stored_key) + log(clamp(strength,1e-4,1))) / 0.10`; at most four valid slots are softmax-combined and passed through inherited out.
- The controlled training auxiliary is frozen before results: observed `(current token, next token)` transition identity defines q/k positives; the next token is supervision only and is never query input. Payload-token CE remains decoder-aligned with detached backbone event representations and detached decoder weights.
- The controlled CPU gate keeps the established delayed-associative seed, LR=4e-3, 500 steps, matched stream-only control, behavioral thresholds, overwrite/stale safety, and session-isolation requirements. It adds the preregistered ambiguous-context mechanism control.

Only a full controlled PASS may authorize a separately preregistered no-training L4 systems benchmark.