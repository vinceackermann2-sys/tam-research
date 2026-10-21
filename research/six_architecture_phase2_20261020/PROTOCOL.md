# Phase two: larger, longer architecture decisions

Frozen before language training. The original phase-one sources and results
remain unchanged. This is development research, not a frontier benchmark.

Models: full attention, attention8 parent and Conv7 use width512, 24 blocks,
8 heads, learned1024-position embeddings and tied GPT2 vocabulary50257. Full
attention FFN1968; parent/Conv7 FFN2640 (floor to16 for tensor-core alignment).
Actual parameter counts:99,837,440 /99,558,912 /99,616,272. Counts differ by
less than0.3%; no inactive padding parameters. Convolution7, left padding6,
zero scalar residual gains in the16 blocks without attention, unchanged block
equations from phase one. CPW local uses the canonical sequence-only equations,
15 blocks, width704, FFN2816, low rank192,99,674,432 parameters. This is a
width/rank-scaled local mixer, not CPW-AFM or the complete original three-path
CPW. It has a maximum15-token temporal receptive field. Scaling that property
does not test distant-memory capability.

All arms: BF16, A100, required flash SDPA for attention; context512, batch16,
AdamW fused, betas.9/.95, weight decay.1, global norm clip1, peak LR.0003,
warmup2,000,000 exposures, cosine to.00003 at the fixed2B horizon. One pass
through the same source-ordered token positions, identical minibatches per arm;
no window replacement. No dropout. Seed202610291, same seed per arm but native
module construction consumes RNG differently; this is not identical initial
weights. The first100M checkpoint is a prefix of the2B schedule, not a
separately optimized100M recipe. No per-arm LR tuning. Corpus selection and
split are mechanical, frozen in prepare_corpus.py at FineWeb-Edu revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Training excludes document hash
buckets0/1; validation and test come from a separate later shard. Exact document
dedup only. Test stays closed until an arm terminates.

The actual upper horizon is floor(2,000,000,000/8192)*8192 =1,999,994,880
training exposures per model; all repeated optimizer exposures count. Evaluation
forwards are separate. Hardware-only discarded models each use65,536 synthetic
exposures and do not share weights with research models. A rejected nearest-
rounding Conv7 count exceeded100M and was blocked before any optimizer update;
the selected floor-rounding model is below the cap.

Stages at floor(100M/8192), floor(300M/8192), floor(1B/8192), floor(2B/8192).
All four arms run the first stage. At100M, continue a candidate to300M if its
validation NLL <=Transformer+.05, OR its training TPS >=1.03*Transformer and
NLL <=Transformer+.10. Conv7 must additionally beat parent by at least.005 NLL.
After300M and1B, continue to the next stage only if NLL <=Transformer-.03 OR
(NLL <=Transformer+.015 AND TPS >=1.03*Transformer). Conv7 still requires.005
benefit over its parent. Parent continues only while Conv7 is eligible. The
Transformer continues while any candidate remains eligible. An early stop
means low allocation value for this tested recipe, not impossibility of the
architecture. No best intermediate checkpoint selection. All attempts retained.

Validation uses the first524,288 reserved tokens, contiguous nonoverlap512-token
blocks, token-weighted NLL. Test uses all2,097,152 reserved tokens, opened once
at termination. Report per-block losses and paired block-bootstrap differences
as approximate uncertainty; adjacent blocks/documents can remain dependent.
One seed at each long horizon cannot establish robust superiority. A passing
candidate earns independent-seed replication before recommending beyond2B or
claiming a breakthrough. Quality/cost measured separately; throughput excludes
evaluation/export/checkpoints, while total wall time is also reported.

Every segment has an exclusive claim, immutable metric result, full optimizer
checkpoint split into SHA256-verified64MiB chunks, and explicit update/exposure
count. Background local export preserves finished segments. No automatic
rerun of an ambiguous interrupted segment; a continuation must conservatively
charge its whole reserved segment budget before authorizing further updates.
Never exceed either user ceiling. No paid account upgrades or new purchases.

Other four designs receive targeted bounded repairs/diagnostics before any
large scaling: valid query-specific retrieval controls for PGW, unseen-address
and rejection learning for CHM, planning-state factual training for SI, and
seed/length robustness for RLT. Their existing failures remain in the record.
