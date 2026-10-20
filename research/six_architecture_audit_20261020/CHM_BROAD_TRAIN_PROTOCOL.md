# First broad TRAIN / reserved DEVELOPMENT V4 study

Frozen before new initialization, training or development scoring. Previous
single-entity capacity checkpoints overfit and did not generalize to the other
TRAIN entities. This new finite study changes the training distribution explicitly;
it never rescues or overwrites those attempts. User authorization is the current
request to continue/training/test the six architectures. Native TEST stays closed.

Use the unchanged V4 neural hard-copy pointer on pinned main. Fresh initialization
seeds202610271/272/273, reset tensor modules as in the completed capacity study.
For each model:4096 single-episode optimizer updates, AdamW .005, wd0, clip1.
Each2048-update epoch contains all64 TRAIN entities, all four families and all
eight variants, shuffled with fixed data-order seed202610270+epoch. No-match's
eight equivalent variants are deliberate repeated exposures for25% negative
family balance. All models receive identical order/data/supervision. Source-address
CE is used only on TRAIN as implemented by the native loss; count it as extra
supervision, not a fair ordinary-decoder baseline.

After the fixed last update, score all64 TRAIN entities and open the native16
DEVELOPMENT entities per family for the first time:8 rotations per positive
entity, one independent no-match/entity. Report cluster means, both source hops,
payload copy, presence/abstention and final answers. Hash panels and checkpoints.
Do not tune on DEVELOPMENT or choose a checkpoint. A preliminary learnability
gate requires >=80% accuracy in every positive family and >=90% no-match rejection
in every initialization.16 negatives cannot certify a low population FPR.

No matched trained comparator or hierarchical acceleration is claimed: the native
V4 pointer is a flat Transformer-based address/copy prototype. Passing establishes
some reusable binding under this supervision; it does not establish a CHM
breakthrough. Failing after TRAIN success identifies generalization as the next
blocker. Hard ceilings100M actual parameters/2B exposures,15 minutes/model, zero
retries/resume. All negative/incomplete outcomes must remain recorded.
