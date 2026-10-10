# Conv7 200M pilot: reusable CPU optimizer and model-builder contract

**Status: synthetic CPU-only executable contract; NOT a live GPU runner.** No new engineering seed, spending cap, Modal workflow, training trigger, dataset download, scientific result or authorization is present.

The frozen 100M/2B scientific runner uses two gradient-accumulation microbatches (64 x 512 tokens each) per optimizer step, float32 cross-entropy, AdamW (betas 0.9 and 0.95, weight decay 0.1), gradient clip 1.0, and a cosine learning-rate schedule with 610 warmup updates over 30,518 full-horizon steps. It separately seeds training with seed+10,000 and the final 50x32x512 held-out evaluation stream with seed+30,000. A compile warm-up trains only a throwaway model and then reseeds/rebuilds the measured model.

The new core registers exact original Transformer, original Attention-8 and Conv7 builders, verifying parameter counts **101,803,520**, **101,795,328**, and **101,803,536** with meta tensors only. The registration rejects any unrecognized model, while the repository's existing source lock and authority checks remain required.

An independently tested CPU function runs one synthetic optimizer update on caller-provided tiny toy-token microbatches, matching loss normalization, accumulation, gradient clipping and the full-2B cosine schedule **prefix**. It checks that all tensors and parameters are CPU-based, rejects missing microbatches and non-finite loss/gradients, and performs no file operations. One test compares its AdamW update with a mathematically equivalent mean-loss reference. This covers optimizer semantics without allocating 100M real CPU weights.

The function is explicitly CPU-only, has no Modal client or dispatch hook, and cannot replace the H100 scientific runner. Actual BF16 autocast, torch.compile performance, H100 convolution kernels, corpus matching, hidden-state behavior, GPU memory, numerical stability at scale and 200M loss/perplexity remain untested.

## Next milestone

A separately reviewed and source-bound Modal engineering runner can adopt this tested comparator/optimizer contract, but must additionally enforce durable remote reservations, one-shot job IDs, an authorized unique engineering seed, strict GPU spending policy, data SHA256, frozen source/tree/harness, retries=0 and no resume. Until the user gives explicit paid authorization, the engineering run remains blocked. Prior scientific seed 60232 is consumed permanently. No scientific replication or 250M/5B is authorized.
