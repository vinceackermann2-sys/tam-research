# Terminal-checkpoint implementation benchmark

Frozen while language training is active, before viewing any language endpoint.
After the training queue finishes, benchmark every terminal checkpoint on the
same A100/runtime. No optimizer updates. BF16, flash SDPA, no gradients; no
compilation or cached KV implementation is introduced for these timings.

Full forward/prefill at context128/512/1024, batch1/8, five warmups and20 timed
iterations, CUDA synchronization around each measured forward. Report median
and p90 wall latency, tokens/sec, peak allocated memory. These measurements
include output projection, exclude tokenization, and apply to the implemented
PyTorch code. They do not represent optimized production serving throughput.

Naive greedy generation:128-token prompt,8 new tokens, batch1, full prefix
recomputed every step, five timed sequences after one warmup. All arms use the
same public synthetic prompt. Label this explicitly as generation WITHOUT KV
caching; do not infer production decoding gains from it.

For terminal Conv7, replay the exact development NLL then zero the16 scalar
convolution gains and reevaluate. This is an inference-time dependence test,
not a clean attribution of the training benefit because coadaptation can make
erasure harmful. Restore gains before latency benchmarks. CPW must have exact
last-position output invariance when changing tokens more than15 positions
back; this verifies its limited receptive field on the actual checkpoint.
