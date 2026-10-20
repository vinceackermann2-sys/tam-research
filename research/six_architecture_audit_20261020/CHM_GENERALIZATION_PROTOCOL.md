# Frozen checkpoint diagnostic: unseen optimizer entities

Created before scoring this panel. No additional training or checkpoint selection.
Evaluate all three existing capacity checkpoints on native TRAIN entities 0–63
except entity 48, the only entity used by their optimizer. Use every eight-way
counterfactual rotation for direct, overwrite and two-hop families, and one
no-match example per entity. Reserved DEVELOPMENT and TEST remain unopened.

Report each entity's cluster mean, source-address correctness, payload-source
correctness, abstention and final answer. An evaluator-only exact oracle verifies
the task labels; it is not a trained fair architecture baseline. Query/visible
memory alone enter the neural forward pass. This is an exploratory out-of-sample
entity diagnostic within the native TRAIN namespace, not a confirmatory held-out
test or proof of architecture superiority. All checkpoints and input panels are
hashed. No feedback affects weights. Zero training exposures are added.
