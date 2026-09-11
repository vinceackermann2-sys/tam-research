# CHM-v1 / EIEM small-LM implementation

This namespace implements the zero-credit code boundary of issue #854.  It is not a scientific result and contains no GPU trigger.

## Matched training geometry

The repository's established small real-language token accounting is preserved exactly:

- GPT-2 vocabulary: 50,257
- train tokens/model: 8,388,608
- dense local attention window: 512 tokens
- contiguous training session: 1,024 tokens = two 512-token local chunks
- microbatch: 4 sessions
- gradient accumulation: 4
- global tokens/optimizer step: `4 * 1024 * 4 = 16,384`, equal to the inherited `8 * 512 * 4`
- optimizer steps: 512
- AdamW `(0.9, 0.95)`, weight decay 0.1, peak LR 3e-4, cosine decay, 2% warmup, grad clip 1.0

LOCAL processes the two chunks with no cross-chunk state.  EIEM's differentiable training path may query exact hidden values written by earlier chunks through exhaustive learned-address scoring.  Current-chunk representations are appended only after that chunk's logits are formed.

No synthetic examples replace the frozen FineWeb-Edu training bytes in this implementation.  Long-memory generators are evaluation probes; any later proposal to mix synthetic training data would be a protocol change that must be preregistered before observing scientific results.

## Parameter accounting

The inherited Transformer backbone is 24,940,288 trainable parameters.  EIEM adds:

- query address projection: `256 * 32 = 8,192`
- key address projection: `256 * 32 = 8,192`
- channel-wise integration gate: `256`

Total EIEM increment: 16,640 parameters; about 0.0667% over LOCAL, below the frozen 1% ceiling without dummy/dead parameters.

Values are the full causal hidden representations available at write time.  They are not separately projected or compressed.

## Exact retrieval contract

Inference stores learned normalized address vectors and exact hidden values.  Both exhaustive and indexed search order candidates by:

`(squared Euclidean distance, monotonic insertion ID)`

For normalized addresses this has the same nearest-neighbour ordering as maximum cosine similarity.  Insertion IDs are store metadata used only to resolve exact distance ties; they are not generator IDs, entity IDs, labels, answers, or other oracle information.

The branch-and-bound index stores only address bounds/membership plus exact source evidence.  A subtree is pruned only when its geometric lower bound is strictly greater than the current best distance; equal-distance subtrees remain searchable so the shared tie contract can be honored.

## Current systems boundary

`tam_research/chm_v1_exact_index.py` is a deterministic NumPy correctness/reference implementation.  It measures exact address vectors distance-scored and directory nodes visited separately.  It is deliberately not presented as a production GPU speedup.  Before any practical efficiency claim, the preregistered gate must measure build/update/search wall-clock and show that indexing overhead does not erase the read reduction.

## Files

- `tam_research/chm_v1_exact_index.py`: exact flat + branch-and-bound search.
- `tam_research/chm_v1_small_lm.py`: LOCAL/EIEM models, episodic session state, differentiable flat path, exact stateful inference.
- `tam_research/chm_v1_small_lm_protocol.py`: frozen data/optimizer/token geometry and reusable training/evaluation functions; no launcher.
- `architectures/chm_v1/small_lm/smoke.py`: non-scientific seed-12345 CPU smoke.
- `tests/test_chm_v1_small_lm*.py`: zero-credit invariant coverage.

## Spend boundary

Scientific seeds 8611/8612/8613 remain reserved and unused.  A scientific GPU attempt requires a separate post-merge authorization that cites the exact merged SHA, GPU class, and budget.  No automatic retry is permitted.
