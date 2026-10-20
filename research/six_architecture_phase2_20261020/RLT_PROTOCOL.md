# RLT retention repair screen

Prospective fresh seeds202610311/312/313. Native bounded affine-scan RLT source
6dd09e019a65ea307f9027f6aabd4f31b11a5b3c, tiny27,712-parameter geometry,
unchanged inference equations. This is not the published nonlinear-feedback RLT.
Pair native carry with a training-only public-command gate regularizer. Both
receive exactly2100 easy8-token updates at LR.003 then1200 mixed128-token
updates at LR.001, batch8, AdamW weight decay.01, norm clip1. Same examples,
initial weights and optimizer. New namespaces70,000,000 and71,000,000.

The regularized arm adds.01*BCEWithLogits(gate projection+bias2, keep_target),
where keep_target=0 at public SET0/SET1 instructions,1 at distractor/query
tokens, across every state dimension. This supplies a task-specific write/retain
prior from visible instruction types, not hidden target bits. It is extra
supervision and compute; success does not establish generic language-model
superiority or an architecture-only advance. Native arm also computes the gate
projection and multiplies its loss by zero to match this auxiliary compute.
No encoder recomputation: capture the actual memory activations used in forward.

Evaluate fresh128 paired prefixes (256examples) at lengths128/256/512/1024,
two opposite SET bits followed by identical distractors, gap length-1. Same
checkpoint with interchunk carry erased must score50%. Require both classes
>=95% at every length in all3 seeds to earn a broader task evaluation. All models
receive1,363,200 input exposures; auxiliary projections do not reconsume inputs.
Save every checkpoint and outcomes. No retuning or failed-seed replacement.
