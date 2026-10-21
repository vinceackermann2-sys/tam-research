# Development-only same-checkpoint convolution erasure

Frozen before the longer Conv7 candidate's final outcome is read. Wait until
all timed GPU training arms finish. Load the single20M-class recovery Conv7
checkpoint and verify its source/data/checkpoint hashes. Re-evaluate the same
fixed65,536 validation-token exposures, then set only its16 convolution residual
gains to zero and re-evaluate. No training, checkpoint choice or new test split.
Record both NLLs, original gains and numerical replay difference from the saved
terminal NLL. A deterioration shows dependence on the trained convolution path;
co-adaptation/distribution shift means it is not a clean training attribution or
independent confirmation of superiority. Do not promote this to a breakthrough.
