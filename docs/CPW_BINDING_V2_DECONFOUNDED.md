# CPW binding-required delayed recall (benchmark v2)

Preregistered zero-GPU design: issue #1307. Source benchmark v1 is frozen and unchanged.

## Why version 1 was confounded

In CPW-v4's original task, the target VALUE is always at one of four offsets (32, 64, 128, 256) and *only one* such candidate offset is occupied by a VALUE token. A decoder that never reads the queried key can recover the target from value-range membership. See issue #1301.

## New generator

Every example contains eight fresh key/value pairs with unique keys, unique values and independently randomized slot assignments.

Each example always places a valid value at every one of eight positions 32, 64, 96, 128, 160, 192, 224, and 256 tokens before the final queried key. Its source key is immediately before that value and a PAIR separator immediately after.

The queried key is sampled from one of the four scored slots (32, 64, 128, 256). Under balanced mixed-delay sampling, each of those four slots is equally likely to be the target, so selecting a fixed offset without consulting the key gets exactly 25%. Merely detecting the VALUE vocabulary at an offset cannot distinguish the correct one because all four candidate offsets always contain values.

An oracle that uses the query key, scans for its unique earlier key and returns the following value obtains 100%.

The strongest counterfactual invariant is: **change only the final query key; keep all 319 preceding tokens fixed; the correct target must change.** A query-blind decoder must return the same prediction for these paired inputs and therefore cannot be correct for both.

## Caveats and next safeguards

This closes the specific discovered positional/vocabulary shortcut. It does *not* automatically rule out every shortcut in arbitrary generative data. The benchmark must be augmented with key-blind neural baselines, query-swap accuracy, and held-out randomized pair/slot distributions before paid scientific experiments.

Per-distance evaluations should always be interpreted with mixed-delay performance. A decoder told the external test bucket (e.g. delay=64) could copy that position without a key; the model must not receive an explicit bucket label.

No current results demonstrate that an AFM model can learn the deconfounded task. Both prior H100 datasets remain historical and scientifically consumed.

## No GPU authority

This issue only provides a generator and zero-GPU validity tests. It neither consumes scientific seeds nor authorizes H100, scaling, chat-model claims or a breakthrough claim.
