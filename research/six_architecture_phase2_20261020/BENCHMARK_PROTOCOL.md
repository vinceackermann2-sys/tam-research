# Public language task benchmarks

Frozen before any large-model endpoint is available. Run once per terminal
checkpoint after the inference timing queue, without optimizer updates. Sources
are pinned in BENCHMARK_SOURCES.json. No selection or hyperparameter tuning uses
these outcomes. PIQA uses the original repository's valid.jsonl and
valid-labels.lst; LAMBADA uses EleutherAI's default English lambada_test.jsonl.
The default file is selected prospectively, not compared against translations.

PIQA prompt: Question: {goal}\nAnswer: followed by a space and each solution.
Rank solutions by mean continuation-token NLL (primary), also report summed-NLL
and character-normalized-NLL rankings. This is our explicitly specified harness;
do not present the token-normalized score as interchangeable with every public
PIQA result. LAMBADA: context is all words before the last space; target is that
space plus the final word. Report exact all-subtoken greedy correctness and
mean summed target-word NLL/perplexity. Contexts left-truncated only if needed
to fit1024; preserve full continuation. GPT2 tiktoken, BF16, batch8, causal
teacher forcing. Exact greedy-word correctness equals all teacher-forced target
positions correct because any first incorrect token already fails the word.

Save source-file SHA256s, item counts, predictions/scores by item, actual token
forwards and checkpoint identifiers. Public web training can contain benchmark
text; no contamination audit was performed. These are auxiliary development
benchmarks and cannot establish uncontaminated reasoning or frontier novelty.
Earlier stopping can leave different terminal token budgets; compare them
explicitly instead of calling those comparisons equal-training-budget results.
