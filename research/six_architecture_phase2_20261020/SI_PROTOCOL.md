# SI factual planning-head repair

Frozen before new training. Use exact user-selected V2 source commit
9298c117e736676269107374a402519e45e52f71. Three fresh seeds202610301/302/303.
Width24/layers4/experts4/max reason3, native modality encoders and hybrid blocks.
Train direct policy for2000 AdamW updates, batch32, LR.001, no decay, gradient
clip1, balance penalty.001, source-authored public cue environment. New data
namespace60,000,000. No research test benchmark or realistic multimodal claim.

Then freeze the entire representation/policy and train only transition/reward/
value/verifier heads for1000 updates, batch32, LR.001, executed uniform random
actions independent of targets. The environment returns reward1 iff the action
is correct,0 otherwise, and a terminal public text observation100+reward. Each
trajectory is a factual ONE-STEP episode. The logged before and terminal states
come from the same frozen forward planning coordinate system. Value targets are
the actually received reward before,0 at terminal. Train transition MSE, reward
MSE, value MSE and.1 verifier calibration at the actual current state. No
counterfactual action reward labels or imagined returns are used. Save executed
action/reward ledger digest, full weights, all persistent state, and optimizer.

Evaluate before and after head repair on the same fresh2048 cue episodes,
namespace62,000,000. Compare direct policy, native gated depth2 search, forced
depth2 search and forced one-step reward search, and their reward calibration.
Memory/fast disabled for this isolated diagnostic; those mechanisms require
separate complete-system evidence. Head repair cannot alter the direct logits;
verify that equality. Native depth2 search extrapolates past this task's terminal
state, so one-step search is a task-aligned diagnostic, not a validated replacement
for general multistep planning. Report that limitation rather than claiming a
complete SI fix. Success gate: forced one-step search accuracy >=direct-.02 and
reward MSE improved in all3 seeds. Native depth2 must independently match direct
before recommending it. Failure pauses scaling of that tested component.

Training exposures count5 modality positions per policy episode, plus5 before
and6 terminal positions per logged factual episode. Heads also consume two
state positions per trajectory: total2000*32*5+1000*32*(11+2)=736,000 per model,
well below2B. Raw image/audio/video scalar elements are not language tokens.
