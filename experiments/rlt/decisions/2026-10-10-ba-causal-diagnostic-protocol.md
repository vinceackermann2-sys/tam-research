# BA — preregistration draft and preflight gate

**Not yet authorized for execution by source alone.** Source and pure preflight tests only. Any eventual one-shot CPU launch requires a committed unique job, consumed-claim guard, successful source-bound CPU preflight, and fail-closed immutable checkpoint verification.

**Question:** In a freshly trained carry-RLT, do opposite earlier SET instructions leave distinguishable final recurrent states, and does the final model readout respond to those differences? Distinguish intermediate-state information retention from failure to use the information for classification.

Model: `BoundedChunkGatedRLT`, 27,712 trainable parameters, chunk size 8, 128-token last-write sequences. No more than one 15-second easy + 90-second long CPU training run on newly reserved seeds. AW and AY attempts are never rerun. The diagnostic does not mutate their evidence.

Method: construct balanced heldout counterfactual input pairs whose last SET token is flipped, with the suffix after that SET unchanged. Each write flip reverses the correct label. At every 8-token chunk boundary, record the recurrent-state L2 difference and relative L2 difference; report actual mean gate/direct retention, class-balanced scores and NLL. At off-chunk distances, feed identical last-chunk tokens with two legitimately reached recurrent states to test whether the logits change. Save the trained model weight bytes and SHA-256.

**Failure is informative:** a state difference may survive without influencing the final answer, or the state itself may lose the difference. If classification stays at chance, preserve the failure rather than tuning the same seed. Do not claim the observed mechanism explains historical AW or AY, whose trained checkpoints are missing.

Model/data access caveat: last-write SET/QUERY semantics are a synthetic supervised inductive-bias task, not language quality. This is one tiny CPU engineering seed, not scientific replication, a competitive general Transformer baseline, or a breakthrough.

**Hard preservation requirement:** checkpoint is gitignored by root `*.pt`. The future run must force-add the specific checkpoint with `git add -f`, verify staged bytes by hash, then read back the exact bytes from the remote committed ref after push. The result must never be marked durably complete unless checkpoint readback succeeds. No automatic or manual replay, no GPU use.
