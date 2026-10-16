# SI original v8: local timeout / CPU attempt accounting (2026-10-16)

Source/protocol already frozen: `scripts/si_original_belief_transfer_v8.py` local Git blob `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`, SHA256 `b3d0350bfbde917c8ce61060a8e22b4d9b680dbe1196ba97594859faf9ee413d`; `docs/SI_ORIGINAL_BELIEF_TRANSFER_V8_PROTOCOL_2026-10-16.md`.

First predeclared engineering seed **100211** was launched with 240 steps, batch64, 8192 evaluation worlds per family. Its container-backed command was interrupted by a 120-second tool execution timeout, and it left **0 bytes** in `result_100211.json`. There is no surviving worker process and no complete score. **Do not rerun 100211, interpret it as a model failure, or impute a metric.**

**BEFORE any further confirmation attempt** designate fresh replacement **100214** for the consumed 100211; unstarted originally preregistered **100212 and 100213** remain valid. Confirmation engineering set will consist of completed seeds 100212/100213/100214 if successful. Keep exact frozen source, data-generation streams, 240 AdamW updates, batch64, and 8192 eval worlds per family. Use a monitored long-running local process with persisted output and per-seed explicit exit status instead of an outer tool call that terminates all processes after 120 seconds. **No change to code, hyperparameters, metrics, family mix, or gate after observing any results.** The replacement means this cannot be labeled unamended preregistered scientific replication; it is a CPU engineering screen with a documented interrupted attempt.

No GPU, Modal, Colab credits, new scientific seed reuse, retries, main writes, or automation. This addendum was committed BEFORE starting seed 100214.
