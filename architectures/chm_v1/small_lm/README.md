# CHM-v1 / EIEM small-LM implementation

This namespace implements the zero-credit code boundary of issue #854. It is not a scientific result and contains no GPU trigger.

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

LOCAL processes the chunks with no cross-chunk state. EIEM's differentiable training path may query exact hidden values written by earlier chunks through exhaustive learned-address scoring. Current-chunk representations are appended only after that chunk's logits are formed.

No synthetic examples replace the frozen FineWeb-Edu training bytes in this implementation. Long-memory generators are evaluation probes only; any later proposal to mix synthetic training data would be a protocol change that must be preregistered before observing scientific results.

## Frozen EIEM implementation choices

These choices are fixed before any scientific seed is allocated:

- learned address dimension: 32
- exact retrievals per token when prior episodic state exists: **2 sequential hops**
- items selected per hop: **top-1 exact nearest neighbour**
- index leaf size: 16
- stored address/value precision: float32 source evidence
- comparison arithmetic in the correctness index: float64 squared distances/bounds over the exact stored float32 values
- memory value: full causal 256-dimensional hidden representation at write time
- integration: the same learned 256-channel gate is applied after each hop
- hop 2 query is formed from the representation after integrating hop 1, rather than from the original hidden state

The two-hop choice is necessary for the preregistered dynamic two-hop family: a one-read design that integrates memory only at the LM head cannot genuinely use a first retrieved record to address a second remote record.

## Parameter accounting

The inherited Transformer backbone is 24,940,288 trainable parameters. EIEM adds:

- query address projection: `256 * 32 = 8,192`
- key address projection: `256 * 32 = 8,192`
- channel-wise integration gate: `256`

Total EIEM increment: 16,640 parameters; about 0.0667% over LOCAL, below the frozen 1% ceiling without dummy/dead parameters. Two-hop retrieval reuses these same parameters and adds no capacity.

Values are the full causal hidden representations available at write time. They are not separately projected or compressed.

## Exact retrieval contract

Inference stores learned normalized address vectors and exact hidden values. Both exhaustive and indexed search order candidates by:

`(squared Euclidean distance, monotonic insertion ID)`

For normalized addresses this has the same nearest-neighbour ordering as maximum cosine similarity. Insertion IDs are store metadata used only to resolve exact distance ties; they are not generator IDs, entity IDs, labels, answers, or other oracle information.

The branch-and-bound index stores only address bounds/membership plus exact source evidence. A subtree is pruned only when its geometric lower bound is strictly greater than the current best distance; equal-distance subtrees remain searchable so the shared tie contract can be honored. Both exhaustive candidate distance and index lower-bound arithmetic are evaluated in float64 from the exact stored float32 vectors to reduce pruning-boundary rounding risk.

Exactness applies independently to both sequential retrieval hops. Any indexed-vs-flat item mismatch at either hop is a gate failure.

## Held-out long-memory probes

`tam_research/chm_v1_long_memory_eval.py` generates deterministic natural-language probes with a caller-supplied frozen GPT-2 encoder. Generator IDs, family labels, candidate sets, expected answers, and stale-answer sets are evaluator metadata only and never enter the model input.

The probe families are:

- rare fact: one target fact among distractors;
- overwrite: multiple visible updates, answer the final authoritative value and score stale-value errors;
- two-hop: one visible record identifies a second visible remote record that contains the answer;
- local negative control: the necessary fact remains within 128 tokens of the query.

For rare fact, overwrite, and two-hop, the **end of the latest evidence required to answer** must be at least 513 encoded tokens before the scored query position. This prevents a nominal long-memory win from being caused by evidence that merely fell across a 512-token chunk boundary while remaining within the intended local horizon.

The current generator is evaluation-only and versioned `chm-v1-heldout-natural-v1`. Production evaluation supplies the already-frozen GPT-2 tokenizer; CPU CI uses a deterministic toy encoder only to validate generator geometry and metadata isolation.

## Current systems boundary

`tam_research/chm_v1_exact_index.py` is a deterministic NumPy correctness/reference implementation. It measures exact address vectors distance-scored and directory nodes visited separately. It is deliberately not presented as a production GPU speedup. Before any practical efficiency claim, the preregistered gate must measure build/update/search wall-clock and show that indexing overhead does not erase the read reduction.

The reference indexed path currently crosses GPU/CPU boundaries when used with a GPU model. Therefore an indexed wall-clock speedup is **not assumed**; if that overhead erases the practical benefit, #854's stop condition must fire even if algorithmic vector-read reduction passes.

## Files

- `tam_research/chm_v1_exact_index.py`: exact flat + branch-and-bound search.
- `tam_research/chm_v1_small_lm.py`: LOCAL/EIEM models, exact episodic state, two-hop differentiable flat path, exact stateful inference.
- `tam_research/chm_v1_small_lm_protocol.py`: frozen data/optimizer/token geometry and reusable training/evaluation functions; no launcher.
- `tam_research/chm_v1_long_memory_eval.py`: held-out >512-token rare-fact/overwrite/two-hop/local-control probes and scoring helpers.
- `architectures/chm_v1/small_lm/smoke.py`: non-scientific seed-12345 CPU smoke.
- `tests/test_chm_v1_small_lm*.py` and `tests/test_chm_v1_long_memory_eval.py`: zero-credit invariant coverage.

## Spend boundary

Scientific seeds 8611/8612/8613 remain reserved and unused. A scientific GPU attempt requires a separate post-merge authorization that cites the exact merged SHA, GPU class, and budget. No automatic retry is permitted.
